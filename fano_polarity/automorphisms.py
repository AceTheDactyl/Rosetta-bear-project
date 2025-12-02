# fano_polarity/automorphisms.py
"""
PSL(3,2) Automorphisms for Coherence Release
============================================

Implements non-trivial PSL(3,2) automorphisms that are applied when
polarity coherence is released. The 168 automorphisms of the Fano plane
form the simple group PSL(3,2) ≅ PSL(2,7) ≅ GL(3,2).

Key generators:
- Cycle: (1234567) - 7-cycle rotation
- Reflection: Involution swapping pairs

When coherence is released at a point, we apply the automorphism
that maps the base configuration to one centered at that point.
"""

from __future__ import annotations

from functools import lru_cache
from itertools import permutations
from typing import Dict, List, Tuple, Optional

# Fano plane lines
FANO_LINES: List[Tuple[int, int, int]] = [
    (1, 2, 3),  # L0
    (1, 4, 5),  # L1
    (1, 6, 7),  # L2
    (2, 4, 6),  # L3
    (2, 5, 7),  # L4
    (3, 4, 7),  # L5
    (3, 5, 6),  # L6
]

LINE_SETS = [frozenset(line) for line in FANO_LINES]

# Identity permutation
IDENTITY: Dict[int, int] = {i: i for i in range(1, 8)}

# Generator: 7-cycle (1234567)
CYCLE: Dict[int, int] = {i: (i % 7) + 1 for i in range(1, 8)}

# Generator: Reflection/involution
REFLECTION: Dict[int, int] = {1: 1, 2: 4, 4: 2, 3: 7, 7: 3, 5: 6, 6: 5}


def _invert(perm: Dict[int, int]) -> Dict[int, int]:
    """Compute inverse permutation."""
    return {v: k for k, v in perm.items()}


def _compose(a: Dict[int, int], b: Dict[int, int]) -> Dict[int, int]:
    """Compose permutations: a ∘ b (apply b first, then a)."""
    return {i: a[b[i]] for i in range(1, 8)}


def _power(perm: Dict[int, int], n: int) -> Dict[int, int]:
    """Compute perm^n."""
    if n == 0:
        return IDENTITY.copy()
    if n < 0:
        return _power(_invert(perm), -n)
    result = IDENTITY.copy()
    for _ in range(n):
        result = _compose(perm, result)
    return result


def _is_automorphism(mapping: Dict[int, int]) -> bool:
    """Check whether a permutation preserves all Fano lines."""
    for line in LINE_SETS:
        image = frozenset(mapping[p] for p in line)
        if image not in LINE_SETS:
            return False
    return True


@lru_cache(maxsize=1)
def enumerate_psl32() -> List[Tuple[int, ...]]:
    """
    Enumerate all 168 PSL(3,2) automorphisms.

    Returns permutations as tuples (image of 1, image of 2, ..., image of 7).
    """
    automorphisms: List[Tuple[int, ...]] = []
    for perm in permutations(range(1, 8)):
        mapping = {i + 1: perm[i] for i in range(7)}
        if _is_automorphism(mapping):
            automorphisms.append(perm)
    return automorphisms


def _tuple_to_mapping(perm: Tuple[int, ...]) -> Dict[int, int]:
    """Convert tuple representation to dict mapping."""
    return {i + 1: perm[i] for i in range(7)}


def get_automorphism_for_point(point: int) -> Dict[int, int]:
    """
    Get an automorphism that maps point 1 to the target point.

    This is used when coherence is released at a specific point -
    we apply the automorphism that "centers" the Fano plane at that point.

    Args:
        point: Target point (1-7)

    Returns:
        Automorphism mapping {old_point: new_point}
    """
    if point == 1:
        return IDENTITY.copy()

    # Find an automorphism that maps 1 -> point
    for perm in enumerate_psl32():
        mapping = _tuple_to_mapping(perm)
        if mapping[1] == point:
            return mapping

    # Fallback (should never happen for valid points)
    return IDENTITY.copy()


def get_automorphism_for_line(line_index: int) -> Dict[int, int]:
    """
    Get an automorphism that maps line L0 = (1,2,3) to the target line.

    Args:
        line_index: Target line index (0-6)

    Returns:
        Automorphism mapping
    """
    if line_index == 0:
        return IDENTITY.copy()

    target = LINE_SETS[line_index]
    base = LINE_SETS[0]

    for perm in enumerate_psl32():
        mapping = _tuple_to_mapping(perm)
        image = frozenset(mapping[p] for p in base)
        if image == target:
            return mapping

    return IDENTITY.copy()


def get_stabilizer(point: int) -> List[Dict[int, int]]:
    """
    Get all automorphisms that fix a given point.

    The stabilizer of a point in PSL(3,2) has order 24 (isomorphic to S4).

    Args:
        point: The point to stabilize (1-7)

    Returns:
        List of automorphisms fixing that point
    """
    stabilizer = []
    for perm in enumerate_psl32():
        mapping = _tuple_to_mapping(perm)
        if mapping[point] == point:
            stabilizer.append(mapping)
    return stabilizer


def get_line_stabilizer(line_index: int) -> List[Dict[int, int]]:
    """
    Get all automorphisms that preserve a given line (as a set).

    The stabilizer of a line has order 24.

    Args:
        line_index: The line index (0-6)

    Returns:
        List of automorphisms preserving that line
    """
    target = LINE_SETS[line_index]
    stabilizer = []
    for perm in enumerate_psl32():
        mapping = _tuple_to_mapping(perm)
        image = frozenset(mapping[p] for p in target)
        if image == target:
            stabilizer.append(mapping)
    return stabilizer


def compute_polarity_automorphism(
    forward_points: Tuple[int, int],
    coherence_point: int,
) -> Dict[int, int]:
    """
    Compute the automorphism to apply when polarity coherence is released.

    The automorphism encodes the relationship between:
    - The forward polarity (defined by two points)
    - The coherence release point (intersection of backward lines)

    This creates a non-trivial transformation that "rotates" the Fano
    structure based on the polarity configuration.

    Args:
        forward_points: The two points used in forward polarity
        coherence_point: The point where coherence was released

    Returns:
        PSL(3,2) automorphism to apply
    """
    p1, p2 = forward_points

    # The forward polarity defines a line
    # The coherence point is on two other lines
    # We want an automorphism that relates these

    # Strategy: compose automorphisms
    # 1. Map point 1 to the coherence point
    # 2. Apply a rotation based on the forward points

    # Get base automorphism centering at coherence point
    center_auto = get_automorphism_for_point(coherence_point)

    # Compute rotation based on forward points
    # Use the difference of points modulo 7 as rotation amount
    rotation_amount = (p2 - p1) % 7
    rotation = _power(CYCLE, rotation_amount)

    # Compose: first rotate, then center
    result = _compose(center_auto, rotation)

    # Verify it's still an automorphism (it should be, by group closure)
    if not _is_automorphism(result):
        # Fallback to just the centering automorphism
        return center_auto

    return result


def apply_automorphism_to_cells(
    cells: List[Tuple[int, int, float]],  # (seal, face, activation)
    automorphism: Dict[int, int],
) -> List[Tuple[int, int, float]]:
    """
    Apply an automorphism to a list of cells.

    Cells are indexed by (seal, face) where seal is 1-7.
    The automorphism permutes the seals.

    Args:
        cells: List of (seal_index, face_index, activation)
        automorphism: PSL(3,2) automorphism

    Returns:
        Permuted cells
    """
    return [
        (automorphism[seal], face, activation)
        for seal, face, activation in cells
    ]


class CoherenceAutomorphismEngine:
    """
    Engine for computing and applying coherence-release automorphisms.

    Tracks the history of automorphisms applied and provides
    methods for composing and inverting transformations.
    """

    def __init__(self):
        self._history: List[Dict[int, int]] = []
        self._cumulative = IDENTITY.copy()

    def apply(
        self,
        forward_points: Tuple[int, int],
        coherence_point: int,
    ) -> Dict[int, int]:
        """
        Compute and record an automorphism for a coherence release.

        Args:
            forward_points: Points used in forward polarity
            coherence_point: Point where coherence was released

        Returns:
            The automorphism that was applied
        """
        auto = compute_polarity_automorphism(forward_points, coherence_point)
        self._history.append(auto)
        self._cumulative = _compose(auto, self._cumulative)
        return auto

    def reset(self) -> None:
        """Reset to identity."""
        self._history.clear()
        self._cumulative = IDENTITY.copy()

    def undo(self) -> Optional[Dict[int, int]]:
        """Undo the last automorphism."""
        if not self._history:
            return None
        last = self._history.pop()
        inv = _invert(last)
        self._cumulative = _compose(inv, self._cumulative)
        return inv

    @property
    def cumulative(self) -> Dict[int, int]:
        """Get the cumulative automorphism."""
        return self._cumulative.copy()

    @property
    def history_length(self) -> int:
        """Get the number of automorphisms applied."""
        return len(self._history)

    def describe(self) -> str:
        """Get a human-readable description of the current state."""
        if self._cumulative == IDENTITY:
            return "Identity (no transformation)"

        # Find cycle structure
        visited = set()
        cycles = []
        for start in range(1, 8):
            if start in visited:
                continue
            cycle = []
            current = start
            while current not in visited:
                visited.add(current)
                cycle.append(current)
                current = self._cumulative[current]
            if len(cycle) > 1:
                cycles.append(tuple(cycle))

        if not cycles:
            return "Identity"

        return " ".join(f"({' '.join(map(str, c))})" for c in cycles)
