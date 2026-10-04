from __future__ import annotations

import string
import unicodedata
from typing import Any

_ASCII_NON_LETTERS = frozenset(string.digits + string.punctuation + " \t\n\r")


def is_latin_replacement(text: str) -> bool:
    """Allow Latin letters plus ASCII digits, punctuation, and whitespace."""
    if not text:
        return False

    for char in text:
        if char in _ASCII_NON_LETTERS:
            continue
        if char.isascii():
            if char.isalpha():
                continue
            return False
        if unicodedata.category(char).startswith("L"):
            if "LATIN" in unicodedata.name(char, ""):
                continue
        return False
    return True


def latin_replacement_token_ids(tokenizer: Any, vocab_size: int) -> list[int]:
    allowed: list[int] = []
    special_ids = set(tokenizer.all_special_ids or [])
    for token_id in range(vocab_size):
        if token_id in special_ids:
            continue
        text = tokenizer.decode(
            [token_id],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        if is_latin_replacement(text):
            allowed.append(token_id)
    return allowed
