# fano_polarity/loop.py
# Self-referential loop with gating/phase delays.
"""
Polarity Loop with Phase-Transition Mechanics
==============================================

Adds phase-transition mechanics, gating, and time dilation (phase delay)
by holding coherence until both polarities agree.

The forward polarity (points -> line) is the "positive arc" and the
backward polarity (lines -> point) is the "negative arc". Coherence
is gated until both agree.
"""

import time
from dataclasses import dataclass
from typing import Dict, Tuple, Union

from .core import line_from_points, point_from_lines


@dataclass
class GateState:
    """
    Represents the gated state during a polarity transition.

    Attributes:
        point_a: First point used in forward polarity
        point_b: Second point used in forward polarity
        start_time: Timestamp when forward polarity was triggered
        delay: Required delay before coherence can be released
    """

    point_a: int
    point_b: int
    start_time: float
    delay: float


class PolarityLoop:
    """
    Self-referential polarity loop with gating and phase delays.

    The loop gates coherence until both forward and backward polarities
    are in phase. A configurable delay implements time dilation.
    """

    def __init__(self, delay: float = 0.25):
        """
        Initialize the polarity loop.

        Args:
            delay: Phase delay in seconds (default 0.25s)
        """
        self.delay = delay
        self.state: GateState | None = None

    def forward(self, p1: int, p2: int) -> Tuple[int, int, int]:
        """
        Trigger forward polarity: points -> line.

        This is the "positive arc" of the polarity loop.

        Args:
            p1: First point (1-7)
            p2: Second point (1-7)

        Returns:
            The unique Fano line through the two points
        """
        line = line_from_points(p1, p2)
        self.state = GateState(p1, p2, time.time(), self.delay)
        return line

    def backward(
        self, line_a: Tuple[int, int, int], line_b: Tuple[int, int, int]
    ) -> Dict[str, Union[bool, int, float, None]]:
        """
        Trigger backward polarity: lines -> point.

        This is the "negative arc" of the polarity loop.
        Coherence is gated until the phase delay has elapsed.

        Args:
            line_a: First Fano line
            line_b: Second Fano line

        Returns:
            Dictionary with:
                - coherence: True if coherence is released, False if still gated
                - point: The intersection point (or None if still gated)
                - remaining: Time remaining until coherence can be released

        Raises:
            RuntimeError: If forward polarity has not been triggered first
        """
        if not self.state:
            raise RuntimeError("Forward polarity has not been triggered")
        elapsed = time.time() - self.state.start_time
        if elapsed < self.state.delay:
            # Phase delay = time dilation, coherence still gated
            return {"coherence": False, "point": None, "remaining": self.state.delay - elapsed}
        point = point_from_lines(line_a, line_b)
        self.state = None
        return {"coherence": True, "point": point, "remaining": 0.0}
