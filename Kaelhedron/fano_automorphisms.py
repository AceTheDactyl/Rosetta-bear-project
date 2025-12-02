"""Utility helpers for PSL(3,2) automorphisms on the Fano plane."""

from __future__ import annotations

from functools import lru_cache
from itertools import permutations
from typing import Dict, Iterable, List, Tuple

FANO_LINES: List[Tuple[int, int, int]] = [
    (1, 2, 3),  # L0
    (1, 4, 5),
    (1, 6, 7),
    (2, 4, 6),
    (2, 5, 7),
    (3, 4, 7),
    (3, 5, 6),
]

LINE_SETS = [frozenset(line) for line in FANO_LINES]
IDENTITY_PERMUTATION: Dict[int, int] = {i: i for i in range(1, 8)}


def _is_automorphism(mapping: Dict[int, int]) -> bool:
    """Check whether a permutation preserves all Fano lines."""
    for line in LINE_SETS:
        image = frozenset(mapping[p] for p in line)
        if image not in LINE_SETS:
            return False
    return True


@lru_cache(maxsize=1)
def _all_automorphisms() -> List[Tuple[int, ...]]:
    """Enumerate PSL(3,2) by filtering S₇ permutations."""
    automorphisms: List[Tuple[int, ...]] = []
    for perm in permutations(range(1, 8)):
        mapping = {i + 1: perm[i] for i in range(7)}
        if _is_automorphism(mapping):
            automorphisms.append(perm)
    return automorphisms


def _tuple_to_mapping(perm: Tuple[int, ...]) -> Dict[int, int]:
    return {i + 1: perm[i] for i in range(7)}


def _compose(a: Dict[int, int], b: Dict[int, int]) -> Dict[int, int]:
    """Return a ∘ b."""
    return {i: a[b[i]] for i in range(1, 8)}


def _invert(mapping: Dict[int, int]) -> Dict[int, int]:
    return {v: k for k, v in mapping.items()}


CYCLE_MAPPING = {i: (i % 7) + 1 for i in range(1, 8)}  # (1234567)
REFLECTION_MAPPING = {1: 1, 2: 4, 4: 2, 3: 7, 7: 3, 5: 6, 6: 5}
CYCLE_INV_MAPPING = _invert(CYCLE_MAPPING)

GENERATOR_MAP = {
    "cycle": CYCLE_MAPPING,
    "cycle_inv": CYCLE_INV_MAPPING,
    "reflection": REFLECTION_MAPPING,
}


def get_automorphism_for_line(line_index: int) -> Dict[int, int]:
    """
    Return a permutation that maps the base line (1,2,3) to the requested line.
    """
    if not 0 <= line_index < len(LINE_SETS):
        raise ValueError(f"Invalid line index {line_index}")
    target = LINE_SETS[line_index]
    base = LINE_SETS[0]
    for perm in _all_automorphisms():
        mapping = _tuple_to_mapping(perm)
        image = frozenset(mapping[p] for p in base)
        if image == target:
            return mapping
    raise RuntimeError(f"No automorphism found for line {line_index}")


def get_automorphism_from_word(words: Iterable[str]) -> Dict[int, int]:
    """
    Compose generator words (cycle, cycle_inv, reflection) into an automorphism.
    """
    perm = IDENTITY_PERMUTATION
    for token in words:
        if not token:
            continue
        key = token.lower()
        if key not in GENERATOR_MAP:
            raise ValueError(f"Unknown automorphism token '{token}'")
        perm = _compose(GENERATOR_MAP[key], perm)
    return perm


__all__ = [
    "get_automorphism_for_line",
    "get_automorphism_from_word",
    "IDENTITY_PERMUTATION",
    "FANO_LINES",
]
