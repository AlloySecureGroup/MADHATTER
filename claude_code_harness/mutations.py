from __future__ import annotations

import hashlib
import random
import re
from collections.abc import Callable

Mutation = Callable[[str, int], str]


def spaced(text: str, seed: int = 0) -> str:
    """Add deterministic extra whitespace without changing word order."""
    rng = random.Random(seed)
    return re.sub(r"\s+", lambda _: " " * rng.randint(1, 3), text).strip()


def alternating_case(text: str, seed: int = 0) -> str:
    """Change case while preserving punctuation and content."""
    upper = bool(seed % 2)
    chars: list[str] = []
    for char in text:
        if char.isalpha():
            chars.append(char.upper() if upper else char.lower())
            upper = not upper
        else:
            chars.append(char)
    return "".join(chars)


def delimited(text: str, seed: int = 0) -> str:
    marker = hashlib.sha256(f"{seed}:{text}".encode()).hexdigest()[:8]
    return f"<untrusted-{marker}>\n{text}\n</untrusted-{marker}>"


MUTATIONS: dict[str, Mutation] = {
    "spaced": spaced,
    "alternating_case": alternating_case,
    "delimited": delimited,
}


def mutate(text: str, mutation_id: str, seed: int = 0) -> str:
    try:
        mutation = MUTATIONS[mutation_id]
    except KeyError as exc:
        raise ValueError(f"unknown mutation: {mutation_id}") from exc
    return mutation(text, seed)
