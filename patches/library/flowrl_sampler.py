# sd-scripts/library/flowrl_sampler.py
import os, re, math, random
from collections import Counter, deque
from typing import List, Dict, Optional, Any, Tuple

_SPLIT_PAT = re.compile(r"[,\n]")

def _read_tags(txt_path: str, tag_regex: Optional[str]) -> List[str]:
    try:
        with open(txt_path, "r", encoding="utf-8") as f:
            raw = f.read().strip().lower()
    except Exception:
        return []
    if tag_regex:
        found = re.findall(tag_regex, raw)
        if found:
            return [t.strip().lower() for t in found if t.strip()]
    parts = _SPLIT_PAT.split(raw) if ("," in raw or "\n" in raw) else raw.split()
    return [p.strip() for p in parts if p.strip()]

def _kl(q: Dict[str, float], p: Dict[str, float]) -> float:
    s = 0.0
    for k, qv in q.items():
        if qv <= 0: continue
        pv = p.get(k, 1e-12)
        s += qv * (math.log(max(qv, 1e-12)) - math.log(max(pv, 1e-12)))
    return s

def _norm(counter: Counter) -> Dict[str, float]:
    tot = float(sum(counter.values())) or 1.0
    return {k: v / tot for k, v in counter.items() if v > 0}

class FlowRLSampler:
    """
    Plans an epoch ordering over sd-scripts' pre-baked batches (buckets_indices)
    to improve tag/concept coverage. If it can't read any tags, it leaves order unchanged.
    """

    def __init__(self,
                 caption_extension: str = ".txt",
                 temperature: float = 1.0,
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
        self._window_tags: Counter = Counter()
        self._window_batches: deque = deque(maxlen=self.W)

    @classmethod
    def from_env(cls):
        e = os.getenv
        return cls(
            caption_extension=e("FLOWRL_CAPTION_EXT", ".txt"),
            temperature=float(e("FLOWRL_TEMPERATURE", "1.0")),
            candidates=int(e("FLOWRL_CANDIDATES", "6")),
            window=int(e("FLOWRL_WINDOW", "1024")),
            diversity_bonus=float(e("FLOWRL_DIVERSITY", "0.03")),
            tag_regex=e("FLOWRL_TAG_REGEX") or None,
            seed=int(e("FLOWRL_SEED", "42")),
        )

    def build_epoch_indices(self,
                            *,
                            buckets_indices: List[Tuple[Any, ...]],
                            bucket_manager: Any,
                            image_data: Optional[dict],
                            epoch: int) -> List[Tuple[Any, ...]]:
        self.rng.seed(epoch)
        self._window_tags.clear()
        self._window_batches.clear()

        # collect a tag-set per batch
        per_batch_tags: List[set] = []
        for item in buckets_indices:
            per_batch_tags.append(self._tags_for_bucket_item(item, bucket_manager, image_data))

        if not any(per_batch_tags):
            # couldn't read tags; leave order unchanged
            return list(buckets_indices)

        vocab = sorted({t for s in per_batch_tags for t in s})
        p_target = {t: 1.0 / max(1, len(vocab)) for t in vocab}

        remaining = list(range(len(buckets_indices)))
        planned: List[int] = []

        while remaining:
            # propose C candidates without replacement
            self.rng.shuffle(remaining)
            cand_ids = remaining[:min(self.C, len(remaining))]

            # score each candidate with FlowRL-style coverage gain + light diversity
            rewards = [self._reward_if_add(per_batch_tags[cid], p_target) for cid in cand_ids]

            # soft pick proportional to exp(r/T)
            m = max(rewards)
            exps = [math.exp((r - m) / max(self.T, 1e-6)) for r in rewards]
            z = sum(exps) or 1.0
            probs = [e / z for e in exps]
            r = self.rng.random()
            acc = 0.0
            pick = len(probs) - 1
            for i, p in enumerate(probs):
                acc += p
                if r <= acc:
                    pick = i
                    break

            chosen_id = cand_ids[pick]
            planned.append(chosen_id)
            remaining.remove(chosen_id)
            self._add_to_window(per_batch_tags[chosen_id])

        return [buckets_indices[i] for i in planned]

    # ---------- helpers ----------

    def _tags_for_bucket_item(self, item, bucket_manager, image_data) -> set:
        # Try common layouts in sd-scripts to recover image paths for this batch.
        paths: List[str] = []
        try:
            if isinstance(item, (tuple, list)) and len(item) >= 3:
                bidx, _, bbatch = item[0], item[1], item[2]
            else:
                bidx = getattr(item, "bucket_index", None)
                bbatch = getattr(item, "batch_index", None)

            bm = bucket_manager
            # Most forks: bm.buckets[bidx].batches[bbatch] -> [image_keys]
            if hasattr(bm, "buckets"):
                bucket = bm.buckets[bidx]
                cand = getattr(bucket, "batches", None) or getattr(bucket, "batch_images", None)
                if cand:
                    imgs = cand[bbatch]
                    paths = self._image_keys_to_paths(imgs, image_data)
            # Fallback helper hook
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
        out: List[str] = []
        if not imgs:
            return out
        for x in imgs:
            if isinstance(x, str):
                out.append(x)
            elif isinstance(x, int) and image_data:
                rec = image_data.get(x)
                if isinstance(rec, dict):
                    for k in ("image_path", "filepath", "path"):
                        if isinstance(rec.get(k), str):
                            out.append(rec[k]); break
                else:
                    for k in ("image_path", "filepath", "path"):
                        if hasattr(rec, k):
                            out.append(getattr(rec, k)); break
        return out

    def _reward_if_add(self, batch_tags: set, p_target: Dict[str, float]) -> float:
        q_now = _norm(self._window_tags)
        kl_now = _kl(q_now, p_target)
        tmp = Counter(self._window_tags); tmp.update(batch_tags)
        q_next = _norm(tmp)
        kl_next = _kl(q_next, p_target)
        coverage_gain = kl_now - kl_next
        div_bonus = self.diversity_bonus * math.log(1 + len(batch_tags))
        return coverage_gain + div_bonus

    def _add_to_window(self, batch_tags: set):
        self._window_batches.append(batch_tags)
        self._window_tags.update(batch_tags)
