"""Synthetic examples that teach SIMPLE where the bottom half of its scale is (paper §4.2).

Regression anchors (targets on the (-1, 1) scale):

    copy            (s, s)               ->  0.0   doing nothing scores zero
    random_other    (s, unrelated)       -> -1.0   unrelated text is maximally wrong
    shuffled        (s, shuffled words)  -> -0.8
    word_dropout    (s, ~50% of words)   -> -0.5
    truncated       (s, first half)      -> -0.4

and, around human-rated data, flipped pairs ``(candidate, source) -> -target`` for every
rated simplification with target > 0.1, plus a copy and a cross-source pair per rated source.
"""

from __future__ import annotations

import random
from collections.abc import Sequence

ADVERSARIAL_TARGETS = {
    "random_other": -1.0,
    "shuffled": -0.8,
    "word_dropout": -0.5,
    "truncated": -0.4,
}


def derangement_partner(items: Sequence[str], rng: random.Random) -> dict[str, str]:
    """Map each distinct item to a different item: shuffle, then pair each item with its
    successor in the shuffled order (a single cycle, so never with itself when n > 1)."""
    pool = list(dict.fromkeys(items))
    rng.shuffle(pool)
    return {s: pool[(i + 1) % len(pool)] for i, s in enumerate(pool)}


def copy_anchors(sources: Sequence[str], n: int, seed: int) -> list[dict]:
    """``n`` (source, source) -> 0.0 examples drawn from ``sources``."""
    pool = sorted(set(sources))
    random.Random(seed).shuffle(pool)
    return [{"source": s, "candidate": s, "target": 0.0, "type": "copy"} for s in pool[:n]]


def adversarial_anchors(sources: Sequence[str], n: int, seed: int) -> list[dict]:
    """Up to four degenerate candidates for each of ``n`` sampled sources."""
    rng = random.Random(seed)
    unique = sorted(set(sources))
    rng.shuffle(unique)
    sampled = unique[:n]
    partner = derangement_partner(sampled, rng)

    out: list[dict] = []
    for src in sampled:
        words = src.split()
        out.append({"source": src, "candidate": partner[src],
                    "target": ADVERSARIAL_TARGETS["random_other"], "type": "random_other"})
        if len(words) >= 5:
            w = words[:]
            rng.shuffle(w)
            out.append({"source": src, "candidate": " ".join(w),
                        "target": ADVERSARIAL_TARGETS["shuffled"], "type": "shuffled"})
        if len(words) >= 6:
            out.append({"source": src, "candidate": " ".join(words[: max(1, len(words) // 2)]),
                        "target": ADVERSARIAL_TARGETS["truncated"], "type": "truncated"})
        if len(words) >= 6:
            kept = [w for w in words if rng.random() > 0.5] or [words[0]]
            out.append({"source": src, "candidate": " ".join(kept),
                        "target": ADVERSARIAL_TARGETS["word_dropout"], "type": "word_dropout"})
    return out


def rating_anchors(rated: Sequence[dict], seed: int, flip_threshold: float = 0.1) -> list[dict]:
    """Copy, flipped and cross-source anchors around human-rated examples."""
    sources = sorted({r["source"] for r in rated})
    out = [{"source": s, "candidate": s, "target": 0.0, "type": "copy"} for s in sources]
    for r in rated:
        if r["target"] > flip_threshold:
            out.append({"source": r["candidate"], "candidate": r["source"],
                        "target": round(-r["target"], 4), "type": "flipped"})
    partner = derangement_partner(sources, random.Random(seed))
    out.extend({"source": s, "candidate": partner[s], "target": -1.0, "type": "random_other"}
               for s in sources)
    return out
