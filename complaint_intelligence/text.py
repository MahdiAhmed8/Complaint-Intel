"""Conservative normalization shared by training and inference."""

from __future__ import annotations

import re
import unicodedata

URL_RE = re.compile(r"https?://\S+|www\.\S+")
SPACE_RE = re.compile(r"\s+")
ARABIC_DIACRITICS_RE = re.compile(r"[\u0617-\u061A\u064B-\u0652\u0670\u06D6-\u06ED]")


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", str(text or "")).lower()
    text = URL_RE.sub(" URL ", text)
    text = ARABIC_DIACRITICS_RE.sub("", text)
    text = text.translate(str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ى": "ي", "ة": "ه", "ـ": ""}))
    text = re.sub(r"[^\w\sÀ-ÿ\u0600-\u06ff!?]", " ", text)
    return SPACE_RE.sub(" ", text).strip()

