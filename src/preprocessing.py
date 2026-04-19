from __future__ import annotations

import re
from typing import Optional

def basic_clean(text: str) -> str:
    """Lightweight text cleaning suitable for TF-IDF baselines.
    Keeps it simple and reproducible for a capstone.
    """
    if text is None:
        return ""
    text = str(text)
    text = text.lower()
    # remove urls
    text = re.sub(r"http\S+|www\S+", " ", text)
    # keep letters/numbers and basic punctuation spacing
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    # collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text
