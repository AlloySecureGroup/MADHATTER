from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Any


def text_diff(original: str, adversarial: str) -> dict[str, Any]:
    """Return a whitespace-preserving word diff and a descriptive similarity."""

    tokenize = lambda text: re.findall(r"\s+|\w+|[^\w\s]", text)
    original_tokens = tokenize(original)
    adversarial_tokens = tokenize(adversarial)
    matcher = SequenceMatcher(None, original_tokens, adversarial_tokens)
    original_segments: list[dict[str, Any]] = []
    adversarial_segments: list[dict[str, Any]] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        changed = tag != "equal"
        if i1 != i2:
            original_segments.append({
                "text": "".join(original_tokens[i1:i2]),
                "changed": changed,
            })
        if j1 != j2:
            adversarial_segments.append({
                "text": "".join(adversarial_tokens[j1:j2]),
                "changed": changed,
            })
    return {
        "similarity": matcher.ratio(),
        "exact_match": original == adversarial,
        "original": original_segments,
        "adversarial": adversarial_segments,
    }
