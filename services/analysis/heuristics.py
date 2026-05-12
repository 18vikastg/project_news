from __future__ import annotations

import re
from typing import Tuple


_KANNADA_RE = re.compile(r"[\u0C80-\u0CFF]")


def infer_translation_quality(kannada: str, english: str) -> str:
    k = (kannada or "").strip()
    e = (english or "").strip()
    if not e:
        return "poor"
    low = e.lower()
    if any(w in low for w in ("translation", "timed out", "timeout", "error", "unavailable")):
        return "poor"
    kn_ratio = len(_KANNADA_RE.findall(e)) / max(len(e), 1)
    if kn_ratio > 0.15:
        return "fair"
    if len(e) < max(8, len(k) * 0.08):
        return "fair"
    if e.lower() == k.lower():
        return "poor"
    return "good"


def infer_summary(english: str, max_len: int = 280) -> str:
    e = (english or "").strip()
    if not e:
        return "No English summary available."
    return e if len(e) <= max_len else e[: max_len - 1].rstrip() + "…"


def infer_category(english: str) -> Tuple[str, float]:
    t = (english or "").lower()
    if not t.strip():
        return "other", 0.35

    sports_kw = ("cricket", "football", "match", "ipl", "player", "game", "sport", "olympic", "cup")
    tech_kw = ("google", "apple", "software", "app", "phone", "internet", "data", "computer", "ai", "tech")
    ent_kw = ("film", "movie", "actor", "cinema", "music", "celebrity", "serial", "show", "award")

    scores = {
        "sports": sum(1 for w in sports_kw if w in t),
        "tech": sum(1 for w in tech_kw if w in t),
        "entertainment": sum(1 for w in ent_kw if w in t),
    }
    best = max(scores, key=scores.get)
    if scores[best] == 0:
        return "other", 0.45
    conf = min(0.55 + 0.1 * scores[best], 0.85)
    return best, conf
