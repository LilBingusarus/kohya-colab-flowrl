import os, re, math, random, json
from collections import Counter, deque, defaultdict
from typing import List, Dict, Optional, Iterable
import torch
from torch.utils.data import Sampler

_SPLIT_PAT = re.compile(r"[,\n]")  # comma or newline split for tags

def _read_tags(txt_path: str, tag_regex: Optional[str]) -> List[str]:
    try:
        with open(txt_path, "r", encoding="utf-8") as f:
            raw = f.read().strip().lower()
    except Exception:
        return []
    if tag_regex:
        m = re.findall(tag_regex, raw)
        if m:
            return [t.strip().lower() for t in m if t.strip()]
    # default: split by commas/newlines; also accept space-separated if no commas present
    if "," in raw or "\n" in raw:
        parts = _SPLIT_PAT.split(raw)
    else:
        parts = raw.split()
    return [p.strip() for p in parts if p.strip()]

def _kl(q: Dict[str, float], p: Dict[str, float]) -> float:
    s = 0.0
    for k, qv in q.items():
        if qv <= 0.0: 
            continue
        pv = p.get(k, 1e-12)
        s += qv * (math.log(max(qv, 1e-12)) - math.log(max(pv, 1e-12)))
    return s

def _norm(counter: Counter) -> Dict[str, float]:
    tot = float(sum(counter.values())) or 1.0
    return {k: v / tot for k, v in counter.items() if v > 0}

def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    inter = len(a & b)
    union = len(a | b) or 1
    return inter / union

class FlowRLBatchedSampler(Sampler[List[int]]):
    """
    FlowRL-style smart batch sampler:
      - Builds concept/tag vocab from caption .txt files next to images
      - Keeps a sliding window histogram of recently used tags
      - For each step, draws C candidate batches and scores each by Delta-KL
        (KL_before - KL_after); adds a small diversity bonus inside the batch
      - Samples the winning batch according to p*(a) ∝ exp(r(a)/T)
    """
    def __init__(
        self,
        dataset,
        batch_size: int,
        *,
        indices: Optional[List[int]] = None,
        caption_extension: str = ".txt",
        flowrl_temperature: float = 0.7,
        flowrl_candidates: int = 8,
        flowrl_window: int = 2048,
        flowrl_target: Optional[Dict[str, float]] = None,
        flowrl_tag_regex: Optional[str] = None,
        steps_per_epoch: Optional[int] = None,
        seed: int = 42,
        diversity_bonus: float = 0.05
    ):
        self.dataset = dataset
        self.bs = int(batch_size)
        self.T = float(flowrl_temperature)
        self.C = int(flowrl_candidates)
        self.W = int(flowrl_window)
        self.tag_regex = flowrl_tag_regex
        self.caption_ext = caption_extension
        self.diversity_bonus = float(diversity_bonus)
        self.rng = random.Random(seed)

        # dataset indices we’re allowed to sample from
        if indices is None:
            self.indices = list(range(len(self.dataset)))
        else:
            self.indices = list(indices)

        # Build per-item tag sets by discovering the paired caption path.
        # We try common patterns used by kohya datasets: the underlying dataset
        # usually exposes an 'image_path' or similar attribute; if not, we fall back
        # to guessing from a 'filepath' field or str(item).
        self.item_tags: List[set] = []
        for i in self.indices:
            txt = self._find_caption_for_item(i)
            tags = set(_read_tags(txt, self.tag_regex)) if txt else set()
            self.item_tags.append(tags)

        # Build target distribution over tags (uniform if not provided)
        tag_counts = Counter()
        for tags in self.item_tags:
            tag_counts.update(tags)
        vocab = sorted(tag_counts.keys())
        if flowrl_target is None or flowrl_target == "uniform":
            self.p_target = {t: 1.0 / max(1, len(vocab)) for t in vocab}
        else:
            # normalize provided weights
            tot = sum(max(0.0, float(w)) for w in flowrl_target.values()) or 1.0
            self.p_target = {k.lower(): max(0.0, float(v)) / tot for k, v in flowrl_target.items()}

        # Sliding window of the last W samples’ tags
        self.window_idx = deque(maxlen=self.W)
        self.window_tags = Counter()

        # Define length (how many batches per epoch)
        if steps_per_epoch is None:
            n = len(self.indices)
            self.steps_per_epoch = math.ceil(n / self.bs)
        else:
            self.steps_per_epoch = int(steps_per_epoch)

    def _find_caption_for_item(self, i: int) -> Optional[str]:
        # Try to infer the path to the caption .txt
        item = self.dataset[i]
        # common: item may be a dict with 'image', 'image_path', or 'filepath'
        candidates = []
        for key in ("image_path", "image", "filepath", "path"):
            if isinstance(item, dict) and key in item:
                candidates.append(item[key])
        # fallback: sometimes item is (image, something) or exposes .image_path attr
        if not candidates:
            if hasattr(item, "image_path"):
                candidates.append(getattr(item, "image_path"))
            elif isinstance(item, (list, tuple)) and len(item) > 0 and isinstance(item[0], str):
                candidates.append(item[0])
        for p in candidates:
            if not isinstance(p, str):
                continue
            root, _ = os.path.splitext(p)
            txt_path = root + self.caption_ext
            if os.path.exists(txt_path):
                return txt_path
        return None

    def __len__(self) -> int:
        return self.steps_per_epoch

    def __iter__(self) -> Iterable[List[int]]:
        # iterate a full "epoch"
        for _ in range(self.steps_per_epoch):
            # draw C candidate batches
            candidates = []
            rewards = []
            for _c in range(self.C):
                batch = [self.rng.choice(self.indices) for _ in range(self.bs)]
                r = self._reward_for_batch(batch)
                candidates.append(batch)
                rewards.append(r)

            # FlowRL-style target distribution over candidates: p* ∝ exp(r/T)
            max_r = max(rewards) if rewards else 0.0
            # subtract max for stability
            exps = [math.exp((r - max_r) / max(self.T, 1e-6)) for r in rewards]
            Z = sum(exps) or 1.0
            probs = [e / Z for e in exps]

            # sample one candidate according to p*
            pick = self._sample_from(probs)
            chosen = candidates[pick]

            # update sliding window state
            self._add_to_window(chosen)

            yield chosen

    def _reward_for_batch(self, batch: List[int]) -> float:
        # current KL
        q_now = _norm(self.window_tags)
        kl_now = _kl(q_now, self.p_target)

        # simulate adding this batch
        tmp_counts = Counter(self.window_tags)
        for idx in batch:
            tmp_counts.update(self.item_tags[self.indices.index(idx)])

        q_next = _norm(tmp_counts)
        kl_next = _kl(q_next, self.p_target)

        # coverage reward: KL reduction (positive is good)
        coverage_gain = kl_now - kl_next

        # simple diversity bonus inside the batch (avg 1 - Jaccard)
        sets = [self.item_tags[self.indices.index(i)] for i in batch]
        pairs = 0
        jd_sum = 0.0
        for i in range(len(sets)):
            for j in range(i + 1, len(sets)):
                pairs += 1
                jd_sum += 1.0 - _jaccard(sets[i], sets[j])
        div_bonus = (jd_sum / pairs) if pairs > 0 else 0.0

        return coverage_gain + self.diversity_bonus * div_bonus

    def _add_to_window(self, batch: List[int]):
        # remove old
        while len(self.window_idx) + len(batch) > self.W and self.window_idx:
            old = self.window_idx.popleft()
            self.window_tags.subtract(self.item_tags[self.indices.index(old)])
            # clean zeros
            for k in list(self.window_tags.keys()):
                if self.window_tags[k] <= 0:
                    del self.window_tags[k]
        # add new
        for idx in batch:
            self.window_idx.append(idx)
            self.window_tags.update(self.item_tags[self.indices.index(idx)])

    def _sample_from(self, probs: List[float]) -> int:
        r = self.rng.random()
        acc = 0.0
        for i, p in enumerate(probs):
            acc += p
            if r <= acc:
                return i
        return len(probs) - 1
