import os, re, math, random, json
from collections import Counter, deque, defaultdict
from typing import List, Dict, Optional, Iterable, Tuple, Any
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



class FlowRLSampler:
    """
    Plans an epoch ordering over pre-baked batches (buckets_indices) to improve
    tag/Concept coverage. Safe: if we can't resolve batch images, returns the
    original order unchanged.
    """

    def __init__(self,
                 caption_extension: str = ".txt",
                 temperature: float = 0.9,
                 candidates: int = 6,
                 window: int = 1024,
                 diversity_bonus: float = 0.03,
                 tag_regex: Optional[str] = None,
                 seed: int = 42):
        self.caption_ext = caption_extension
        self.T = float(temperature)
        self.C = int(candidates)
        self.W = int(window)
        self.diversity_bonus = float(diversity_bonus)
        self.tag_regex = tag_regex
        self.rng = random.Random(seed)

        # rolling window state
        self._window_batches: deque = deque(maxlen=self.W)
        self._window_tags: Counter = Counter()

    @classmethod
    def from_args(cls, args):
        # tolerate missing attributes (wrapper may inject them)
        return cls(
            caption_extension=getattr(args, "caption_extension", ".txt"),
            temperature=getattr(args, "flowrl_temperature", 0.9),
            candidates=getattr(args, "flowrl_candidates", 6),
            window=getattr(args, "flowrl_window", 1024),
            diversity_bonus=getattr(args, "flowrl_diversity_bonus", 0.03),
            tag_regex=getattr(args, "flowrl_tag_regex", None),
            seed=getattr(args, "seed", 42),
        )

    # ---- public API used by train_util.py ----
    def build_epoch_indices(self,
                            *,
                            buckets_indices: List[Tuple[Any, ...]],
                            bucket_manager: Any,
                            image_data: Optional[dict],
                            epoch: int) -> List[Tuple[Any, ...]]:
        self.rng.seed(epoch)
        self._window_batches.clear()
        self._window_tags.clear()

        # Precompute per-batch tag sets
        batch_tags: List[set] = []
        for bi in buckets_indices:
            tags = self._tags_for_bucket_item(bi, bucket_manager, image_data)
            batch_tags.append(tags)

        # If we couldn't recover any tags, bail out safely
        if not any(len(t) for t in batch_tags):
            return list(buckets_indices)

        # Uniform target over observed vocab
        vocab = sorted({t for s in batch_tags for t in s})
        p_target = {t: 1.0 / max(1, len(vocab)) for t in vocab}

        # Plan order using FlowRL-style soft selection over C candidates each step
        remaining = list(range(len(buckets_indices)))
        planned_idx: List[int] = []

        while remaining:
            # pick C candidates without replacement
            cand_ids = self._sample_without_replacement(remaining, k=min(self.C, len(remaining)))
            rewards = []
            for cid in cand_ids:
                rewards.append(self._reward_if_add(batch_tags[cid], p_target))

            # soft select
            pick_local = self._soft_pick(rewards)
            chosen_id = cand_ids[pick_local]
            planned_idx.append(chosen_id)

            # update window
            self._add_to_window(batch_tags[chosen_id])
            remaining.remove(chosen_id)

        # return reordered buckets_indices
        return [buckets_indices[i] for i in planned_idx]

    # ---- internals ----

    def _tags_for_bucket_item(self, item, bucket_manager, image_data) -> set:
        """
        Try to map a (bucket_index, batch_size, batch_index) item to the list of image paths
        for that batch, then read their caption .txt files to form a tag set.
        Works across common kohya forks; falls back to {} if we can't resolve.
        """
        paths = []
        try:
            # Common shape: (bucket_idx, batch_size, batch_idx)
            if isinstance(item, (tuple, list)) and len(item) >= 3:
                bidx, _, bbatch = item[0], item[1], item[2]
            else:
                # sometimes it's an object with attributes
                bidx = getattr(item, "bucket_index", None)
                bbatch = getattr(item, "batch_index", None)
            # Heuristic 1: bucket_manager has .buckets[bidx].batches[bbatch] -> [image_keys]
            bm = bucket_manager
            if hasattr(bm, "buckets"):
                bucket = bm.buckets[bidx]
                cand = getattr(bucket, "batches", None) or getattr(bucket, "batch_images", None)
                if cand:
                    imgs = cand[bbatch]
                    paths = self._image_keys_to_paths(imgs, image_data)
            # Heuristic 2: direct helper
            if not paths and hasattr(bm, "get_batch_image_keys"):
                imgs = bm.get_batch_image_keys(bidx, bbatch)
                paths = self._image_keys_to_paths(imgs, image_data)
        except Exception:
            paths = []

        tags = set()
        for p in paths:
            root, _ = os.path.splitext(p)
            txt_path = root + self.caption_ext
            if os.path.exists(txt_path):
                tags.update(_read_tags(txt_path, self.tag_regex))
        return tags

    def _image_keys_to_paths(self, imgs, image_data) -> List[str]:
        out = []
        if not imgs:
            return out
        # imgs may be strings (paths) or int keys into image_data
        for x in imgs:
            if isinstance(x, str):
                out.append(x)
            elif isinstance(x, int) and image_data:
                # image_data may be a dict keyed by int -> object with .image_path / .filepath
                rec = image_data.get(x)
                if isinstance(rec, dict):
                    for k in ("image_path", "filepath", "path"):
                        if k in rec and isinstance(rec[k], str):
                            out.append(rec[k])
                            break
                else:
                    for k in ("image_path", "filepath", "path"):
                        if hasattr(rec, k):
                            out.append(getattr(rec, k))
                            break
        return out

    def _reward_if_add(self, batch_tagset: set, p_target: Dict[str, float]) -> float:
        q_now = _norm(self._window_tags)
        kl_now = _kl(q_now, p_target)
        tmp = Counter(self._window_tags)
        tmp.update(batch_tagset)
        q_next = _norm(tmp)
        kl_next = _kl(q_next, p_target)
        coverage_gain = kl_now - kl_next

        # light intra-batch diversity proxy (tag count)
        div_bonus = self.diversity_bonus * math.log(1 + len(batch_tagset))
        return coverage_gain + div_bonus

    def _add_to_window(self, batch_tagset: set):
        self._window_batches.append(batch_tagset)
        self._window_tags.update(batch_tagset)
        # prune if over window (deque already prunes by size)

    def _soft_pick(self, rewards: List[float]) -> int:
        if not rewards:
            return 0
        m = max(rewards)
        exps = [math.exp((r - m) / max(self.T, 1e-6)) for r in rewards]
        z = sum(exps) or 1.0
        probs = [e / z for e in exps]
        r = self.rng.random()
        acc = 0.0
        for i, p in enumerate(probs):
            acc += p
            if r <= acc:
                return i
        return len(probs) - 1

    def _sample_without_replacement(self, pool: List[int], k: int) -> List[int]:
        pool = list(pool)
        self.rng.shuffle(pool)
        return pool[:k]
