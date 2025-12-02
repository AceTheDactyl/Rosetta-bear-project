# fano_polarity/service.py
# Orchestrates the loop and writes back into KaelhedronStateBus.
"""
Polarity Service
================

Bridges the polarity loop into the Kaelhedron state bus so released
coherence actually permutes the 21-cell graph.

Front-end controls call `inject()` with two points, the service delays
release until both polarities are in phase, then `release()` applies
the Kaelhedron permutation at the gated point once coherence is free.
"""

from typing import Dict, Tuple, Union

from Kaelhedron import KaelhedronStateBus

from .loop import PolarityLoop


class PolarityService:
    """
    Orchestrates the polarity loop and writes back into KaelhedronStateBus.

    This service provides the bridge between the abstract polarity loop
    mechanics and the concrete Kaelhedron 21-cell state representation.
    """

    def __init__(self, bus: KaelhedronStateBus, delay: float = 0.25):
        """
        Initialize the polarity service.

        Args:
            bus: The KaelhedronStateBus to apply permutations to
            delay: Phase delay in seconds (default 0.25s)
        """
        self.bus = bus
        self.loop = PolarityLoop(delay=delay)

    def inject(self, p1: int, p2: int) -> Dict[str, Tuple[int, int, int]]:
        """
        Inject two points into the polarity loop (forward polarity).

        This triggers the forward polarity (positive arc) and returns
        the unique Fano line through the points.

        Args:
            p1: First point (1-7)
            p2: Second point (1-7)

        Returns:
            Dictionary with the computed Fano line
        """
        line = self.loop.forward(p1, p2)
        return {"line": line}

    def release(
        self, line_a: Tuple[int, int, int], line_b: Tuple[int, int, int]
    ) -> Dict[str, Union[bool, int, float, None]]:
        """
        Release coherence via backward polarity.

        This triggers the backward polarity (negative arc). If coherence
        is released (phase delay has elapsed), applies the Kaelhedron
        permutation at the intersection point.

        Args:
            line_a: First Fano line
            line_b: Second Fano line

        Returns:
            Dictionary with:
                - coherence: True if coherence was released
                - point: The intersection point (or None if still gated)
                - remaining: Time remaining until coherence can be released
        """
        result = self.loop.backward(line_a, line_b)
        if result["coherence"]:
            self.bus.apply_permutation({result["point"]: result["point"]})
        return result
