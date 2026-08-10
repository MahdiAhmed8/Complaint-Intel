"""Fast, explainable Arabic/French language identification."""

from __future__ import annotations

import re

ARABIC_RE = re.compile(r"[\u0600-\u06ff\u0750-\u077f\u08a0-\u08ff]")
LATIN_RE = re.compile(r"[A-Za-zÀ-ÿ]")


def detect_language(text: str) -> str:
    """Return ``ar``, ``fr`` or ``unknown`` from Unicode-script counts.

    Tunisian Arabic frequently contains French words. Arabic wins when at least
    30% of detected alphabetic characters use Arabic script; this is deliberate
    because a short Arabic frame can contain a French product or technical term.
    """
    text = text or ""
    arabic = len(ARABIC_RE.findall(text))
    latin = len(LATIN_RE.findall(text))
    total = arabic + latin
    if total == 0:
        return "unknown"
    return "ar" if arabic / total >= 0.30 else "fr"


def language_name(code: str) -> str:
    return {"ar": "Arabic", "fr": "French", "unknown": "Unknown"}.get(code, code)

