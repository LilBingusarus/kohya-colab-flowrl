import os, re, math, random
from collections import Counter, deque
from typing import List, Dict, Optional, Any, Tuple

_SPLIT_PAT = re.compile(r"[,\n]")

def _read_tags(txt_path: str, tag_regex: Optional[str]) -> List[str]:
    try:
        raw = open(txt_path, "r", encoding="utf-8").read().strip().lower()
    except Exception:
        return []
    if tag_regex:
        found = re.findall(tag_regex, raw)
        return [t.strip().lower() for t in found if t.strip()]
    if "," in raw or "\n" in raw:
        parts = _SPLIT_PAT.split(raw)
    else:
        parts = raw.split()
    return [p.strip() for p in parts if p.strip()]

def _kl(q: Dict[str, float], p: Dict[str, float]) -> float:
    s = 0.0
    for k, qv in q.items():
        if qv <= 0: 
            continue
        pv = p.get(k, 1e-12)
        s += qv * (math.log(max(qv,1e-12)) - math.log(max(pv,1e-12)))
    return s

def _norm(counter: Counter) -> Dict[str, float]:
    tot = float(sum(counter.values())) or 1.0
    return {k: v / tot for k, v in counter.items() if v > 0}

class FlowRLSampler:
    """
    Plans a per-epoch ordering over pre-baked bucket batches to improve concept/tag coverage.
    Safe fallback: if tags can't be read, returns the original order unchanged.
    """

    def __init__(
        self,
        caption_extension: str = ".txt",
        temperature: float = 0.9,
        candidates: int = 6,
        window: int = 1024,
        diversity_bonus: float = 0.03,
        tag_regex: Optional[str] = None,
        seed: int = 42,
    ):
        self.caption_ext = caption_extension
        self.T = float(temperature)
        self.C = int(candidates)
        self.W = int(window)
        self.diversity_bonus = float(diversity_bonus)
        self.tag_regex = tag_regex
        self.rng = random.Random(seed)
        self._win_tags = Counter()
        self._win_batches = deque(maxlen=self.W)

    @classmethod
    def from_args(cls, args):
        return cls(
            caption_extension=getattr(args, "caption_extension", ".txt"),
            temperature=getattr(args, "flowrl_temperature", 0.9),
            candidates=getattr(args, "flowrl_candidates", 6),
            window=getattr(args, "flowrl_window", 1024),
            diversity_bonus=getattr(args, "flowrl_diversity_bonus", 0.03),
            tag_regex=getattr(args, "flowrl_tag_regex", None),
            seed=getattr(args, "seed", 42),
        )

    # ---- public API called by BaseDataset.shuffle_buckets ----
    def build_epoch_indices(
        self,
        *,
        buckets_indices: List[Tuple[Any, ...]],
        bucket_manager: Any,
        image_data: Optional[dict],
        epoch: int,
    ) -> List[Tuple[Any, ...]]:
        self.rng.seed(epoch)
        self._win_tags.clear()
        self._win_batches.clear()

        # Read per-batch tag sets
        batch_tags: List[set] = []
        for bi in buckets_indices:
            tags = self._tags_for_bucket_item(bi, bucket_manager, image_data)
            batch_tags.append(tags)

        # If we couldn't read any tags, do nothing special
        if not any(len(s) for s in batch_tags):
            return list(buckets_indices)

        # Uniform target over observed tags
        vocab = sorted({t for s in batch_tags for t in s})
        p_target = {t: 1.0 / max(1, len(vocab)) for t in vocab}

        remaining = list(range(len(buckets_indices)))
        planned = []

        while remaining:
            cand_ids = self._pick_candidates(remaining, k=min(self.C, len(remaining)))
            rewards = [self._reward_if_add(batch_tags[cid], p_target) for cid in cand_ids]
            pick = self._soft_pick(rewards)
            chosen = cand_ids[pick]
            planned.append(chosen)
            self._add_to_window(batch_tags[chosen])
            remaining.remove(chosen)

        return [buckets_indices[i] for i in planned]

    # ---- helpers ----
    def _tags_for_bucket_item(self, item, bucket_manager, image_data) -> set:
        paths = []
        try:
            # Expect (bucket_idx, batch_size, batch_idx) tuple
            if isinstance(item, (tuple, list)) and len(item) >= 3:
                bidx, _, bbatch = item[0], item[1], item[2]
            else:
                bidx = getattr(item, "bucket_index", None)
                bbatch = getattr(item, "batch_index", None)

            # Heuristic: bucket manager holds batches -> image keys
            bucket = getattr(bucket_manager, "buckets", [])[bidx]
            batches = getattr(bucket, "batches", None) or getattr(bucket, "batch_images", None)
            if batches:
                image_keys = batches[bbatch]
                paths = self._image_keys_to_paths(image_keys, image_data)

            if not paths and hasattr(bucket_manager, "get_batch_image_keys"):
                image_keys = bucket_manager.get_batch_image_keys(bidx, bbatch)
                paths = self._image_keys_to_paths(image_keys, image_data)
        except Exception:
            paths = []

        tags = set()
        for p in paths:
            root, _ = os.path.splitext(p)
            txt = root + self.caption_ext
            if os.path.exists(txt):
                tags.update(_read_tags(txt, self.tag_regex))
        return tags

    def _image_keys_to_paths(self, keys, image_data) -> List[str]:
        out = []
        if not keys:
            return out
        for k in keys:
            if isinstance(k, str):
                out.append(k)
            elif isinstance(k, int) and image_data:
                rec = image_data.get(k)
                if isinstance(rec, dict):
                    for name in ("image_path", "filepath", "path"):
                        if name in rec and isinstance(rec[name], str):
                            out.append(rec[name])
                            break
                else:
                    for name in ("image_path", "filepath", "path"):
                        if hasattr(rec, name):
                            out.append(getattr(rec, name))
                            break
        return out

    def _reward_if_add(self, batch_tagset: set, p_target: Dict[str, float]) -> float:
        q_now = _norm(self._win_tags)
        kl_now = _kl(q_now, p_target)
        tmp = Counter(self._win_tags)
        tmp.update(batch_tagset)
        q_next = _norm(tmp)
        kl_next = _kl(q_next, p_target)
        coverage_gain = kl_now - kl_next
        div_bonus = self.diversity_bonus * math.log(1 + len(batch_tagset))
        return coverage_gain + div_bonus

    def _add_to_window(self, tagset: set):
        self._win_batches.append(tagset)
        self._win_tags.update(tagset)

    def _pick_candidates(self, pool: List[int], k: int) -> List[int]:
        pool = list(pool)
        self.rng.shuffle(pool)
        return pool[:k]

    def _soft_pick(self, rewards: List[float]) -> int:
        if not rewards:
            return 0
        m = max(rewards)
        exps = [math.exp((r - m) / max(self.T, 1e-6)) for r in rewards]
        z = sum(exps) or 1.0
        r = self.rng.random()
        acc = 0.0
        for i, p in enumerate(exps):
            acc += p / z
            if r <= acc:
                return i
        return len(rewards) - 1
