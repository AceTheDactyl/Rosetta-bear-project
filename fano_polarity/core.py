# fano_polarity/core.py
# Forward polarity (points -> line) and backward polarity (lines -> point).
"""
Core Fano Polarity Lookups
==========================

Encapsulates raw Fano lookups so higher layers don't duplicate the incidence logic.
Implements the two Fano axioms:
- Axiom 1: Two distinct points define a unique line (forward polarity)
- Axiom 2: Two distinct lines intersect at a unique point (backward polarity)
"""

from typing import Tuple

from Kaelhedron.fano_automorphisms import FANO_LINES


def line_from_points(p1: int, p2: int) -> Tuple[int, int, int]:
    """
    Axiom 1 enforcement: unique line through the two points.

    Args:
        p1: First point (1-7)
        p2: Second point (1-7)

    Returns:
        The unique Fano line containing both points

    Raises:
        ValueError: If points are not distinct or invalid
    """
    if p1 == p2:
        raise ValueError("Points must be distinct")
    for line in FANO_LINES:
        if p1 in line and p2 in line:
            return line
    raise ValueError("Invalid points supplied")


def point_from_lines(l1: Tuple[int, int, int], l2: Tuple[int, int, int]) -> int:
    """
    Axiom 2 enforcement: unique intersection of lines.

    Args:
        l1: First Fano line (3-tuple of points)
        l2: Second Fano line (3-tuple of points)

    Returns:
        The unique point at the intersection

    Raises:
        ValueError: If lines don't intersect at exactly one point
    """
    intersection = set(l1).intersection(l2)
    if len(intersection) != 1:
        raise ValueError("Lines must intersect at exactly one point")
    return intersection.pop()
