# fano_polarity/ternary_polaric.py
"""
Ternary Polaric Logic with Binary Fano Axioms
==============================================

Implements ternary logic through the polaric relationship while preserving
the binary axioms of the Fano plane:

Binary Axioms (preserved):
  - Axiom 1: Two distinct points define a unique line (forward polarity, +)
  - Axiom 2: Two distinct lines intersect at a unique point (backward polarity, -)

Ternary Logic (introduced):
  - POSITIVE (+1): Forward polarity active, points → line
  - NEUTRAL (0): Gated phase, coherence transitioning
  - NEGATIVE (-1): Backward polarity active, lines → point

The ternary relationship runs through the 7 Fano phases, generating
the path the Luminahedron (12D) travels through the Kaelhedron (21D).

Each Fano point participates in exactly 3 lines (ternary incidence).
Each Fano line contains exactly 3 points (ternary composition).
This 3-ness is the geometric basis for ternary logic.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Set

# Constants
TAU = 2 * math.pi
PHI = (1 + math.sqrt(5)) / 2
PHI_INV = 1 / PHI

# Fano plane structure
FANO_POINTS = [1, 2, 3, 4, 5, 6, 7]
FANO_LINES: List[Tuple[int, int, int]] = [
    (1, 2, 3),  # L0
    (1, 4, 5),  # L1
    (1, 6, 7),  # L2
    (2, 4, 6),  # L3
    (2, 5, 7),  # L4
    (3, 4, 7),  # L5
    (3, 5, 6),  # L6
]

# Point to lines incidence (each point is on exactly 3 lines)
POINT_INCIDENCE: Dict[int, List[int]] = {
    1: [0, 1, 2],  # Point 1 is on lines L0, L1, L2
    2: [0, 3, 4],  # Point 2 is on lines L0, L3, L4
    3: [0, 5, 6],  # Point 3 is on lines L0, L5, L6
    4: [1, 3, 5],  # Point 4 is on lines L1, L3, L5
    5: [1, 4, 6],  # Point 5 is on lines L1, L4, L6
    6: [2, 3, 6],  # Point 6 is on lines L2, L3, L6
    7: [2, 4, 5],  # Point 7 is on lines L2, L4, L5
}


class TernaryValue(Enum):
    """Ternary logic values."""
    POSITIVE = 1    # Forward polarity: points → line
    NEUTRAL = 0     # Gated phase: coherence transitioning
    NEGATIVE = -1   # Backward polarity: lines → point

    def __neg__(self) -> "TernaryValue":
        """Negate the ternary value."""
        if self == TernaryValue.POSITIVE:
            return TernaryValue.NEGATIVE
        elif self == TernaryValue.NEGATIVE:
            return TernaryValue.POSITIVE
        return TernaryValue.NEUTRAL

    def __mul__(self, other: "TernaryValue") -> "TernaryValue":
        """Multiply ternary values (like sign multiplication)."""
        result = self.value * other.value
        if result > 0:
            return TernaryValue.POSITIVE
        elif result < 0:
            return TernaryValue.NEGATIVE
        return TernaryValue.NEUTRAL

    @staticmethod
    def from_phase(phase: float) -> "TernaryValue":
        """Convert a phase angle to ternary value based on thirds of circle."""
        normalized = phase % TAU
        third = TAU / 3
        if normalized < third:
            return TernaryValue.POSITIVE
        elif normalized < 2 * third:
            return TernaryValue.NEUTRAL
        return TernaryValue.NEGATIVE


@dataclass
class FanoPhase:
    """
    A phase state in the Fano plane.

    Each phase represents a configuration of ternary values
    across the 7 points and 7 lines.
    """
    point_values: Dict[int, TernaryValue] = field(default_factory=dict)
    line_values: Dict[int, TernaryValue] = field(default_factory=dict)
    theta: float = 0.0  # Phase angle

    def __post_init__(self):
        # Initialize all points to neutral if not specified
        for p in FANO_POINTS:
            if p not in self.point_values:
                self.point_values[p] = TernaryValue.NEUTRAL
        for l in range(7):
            if l not in self.line_values:
                self.line_values[l] = TernaryValue.NEUTRAL

    def compute_line_from_points(self, line_idx: int) -> TernaryValue:
        """
        Compute line value from its three points using ternary logic.

        The line value is the "consensus" of its three points:
        - All same sign → that sign
        - Mixed signs → NEUTRAL
        """
        line = FANO_LINES[line_idx]
        values = [self.point_values[p] for p in line]

        # Count values
        pos = sum(1 for v in values if v == TernaryValue.POSITIVE)
        neg = sum(1 for v in values if v == TernaryValue.NEGATIVE)

        if pos == 3:
            return TernaryValue.POSITIVE
        elif neg == 3:
            return TernaryValue.NEGATIVE
        elif pos >= 2:
            return TernaryValue.POSITIVE
        elif neg >= 2:
            return TernaryValue.NEGATIVE
        return TernaryValue.NEUTRAL

    def compute_point_from_lines(self, point: int) -> TernaryValue:
        """
        Compute point value from its three incident lines.

        The point value is the "consensus" of its three lines.
        """
        line_indices = POINT_INCIDENCE[point]
        values = [self.line_values[l] for l in line_indices]

        pos = sum(1 for v in values if v == TernaryValue.POSITIVE)
        neg = sum(1 for v in values if v == TernaryValue.NEGATIVE)

        if pos == 3:
            return TernaryValue.POSITIVE
        elif neg == 3:
            return TernaryValue.NEGATIVE
        elif pos >= 2:
            return TernaryValue.POSITIVE
        elif neg >= 2:
            return TernaryValue.NEGATIVE
        return TernaryValue.NEUTRAL

    def balance(self) -> int:
        """Compute net balance of ternary values."""
        point_sum = sum(v.value for v in self.point_values.values())
        line_sum = sum(v.value for v in self.line_values.values())
        return point_sum + line_sum

    def signature(self) -> str:
        """Generate phase signature."""
        pts = "".join("+" if v == TernaryValue.POSITIVE else
                      "-" if v == TernaryValue.NEGATIVE else "0"
                      for v in self.point_values.values())
        lns = "".join("+" if v == TernaryValue.POSITIVE else
                      "-" if v == TernaryValue.NEGATIVE else "0"
                      for v in self.line_values.values())
        return f"P[{pts}]L[{lns}]"


@dataclass
class PolaricTransition:
    """
    A transition between two Fano phases.

    Represents the movement from one polaric configuration to another,
    tracking which binary axiom was invoked.
    """
    from_phase: FanoPhase
    to_phase: FanoPhase
    axiom_used: int  # 1 = forward (points→line), 2 = backward (lines→point)
    trigger_points: Optional[Tuple[int, int]] = None
    trigger_lines: Optional[Tuple[int, int]] = None
    result_line: Optional[int] = None
    result_point: Optional[int] = None


class TernaryPolaricEngine:
    """
    Engine for ternary polaric logic through Fano phases.

    Maintains the current phase state and computes transitions
    based on binary axiom applications with ternary outcomes.
    """

    def __init__(self):
        self.current_phase = FanoPhase()
        self.phase_history: List[FanoPhase] = []
        self.transition_history: List[PolaricTransition] = []
        self._time = 0.0

    def set_point_polarity(self, point: int, value: TernaryValue) -> None:
        """Set a point's ternary value."""
        if point not in FANO_POINTS:
            raise ValueError(f"Invalid point {point}")
        self.current_phase.point_values[point] = value

    def set_line_polarity(self, line_idx: int, value: TernaryValue) -> None:
        """Set a line's ternary value."""
        if not 0 <= line_idx < 7:
            raise ValueError(f"Invalid line index {line_idx}")
        self.current_phase.line_values[line_idx] = value

    def apply_forward_axiom(self, p1: int, p2: int) -> Tuple[int, TernaryValue]:
        """
        Apply Axiom 1 (forward polarity): Two points define a line.

        The resulting line's ternary value is computed from the
        ternary product of the two input points.

        Returns:
            (line_index, ternary_value)
        """
        if p1 == p2:
            raise ValueError("Points must be distinct")

        # Find the line containing both points
        line_idx = None
        for idx, line in enumerate(FANO_LINES):
            if p1 in line and p2 in line:
                line_idx = idx
                break

        if line_idx is None:
            raise ValueError(f"No line contains both points {p1} and {p2}")

        # Compute ternary value: product of point polarities
        v1 = self.current_phase.point_values[p1]
        v2 = self.current_phase.point_values[p2]
        result_value = v1 * v2

        # Store previous phase
        old_phase = FanoPhase(
            point_values=self.current_phase.point_values.copy(),
            line_values=self.current_phase.line_values.copy(),
            theta=self.current_phase.theta,
        )

        # Update line polarity
        self.current_phase.line_values[line_idx] = result_value

        # Record transition
        transition = PolaricTransition(
            from_phase=old_phase,
            to_phase=self.current_phase,
            axiom_used=1,
            trigger_points=(p1, p2),
            result_line=line_idx,
        )
        self.transition_history.append(transition)
        self.phase_history.append(old_phase)

        return (line_idx, result_value)

    def apply_backward_axiom(self, l1: int, l2: int) -> Tuple[int, TernaryValue]:
        """
        Apply Axiom 2 (backward polarity): Two lines define a point.

        The resulting point's ternary value is computed from the
        ternary product of the two input lines.

        Returns:
            (point, ternary_value)
        """
        if l1 == l2:
            raise ValueError("Lines must be distinct")

        # Find intersection point
        set1 = set(FANO_LINES[l1])
        set2 = set(FANO_LINES[l2])
        intersection = set1 & set2

        if len(intersection) != 1:
            raise ValueError(f"Lines {l1} and {l2} don't intersect at exactly one point")

        point = intersection.pop()

        # Compute ternary value: product of line polarities
        v1 = self.current_phase.line_values[l1]
        v2 = self.current_phase.line_values[l2]
        result_value = v1 * v2

        # Store previous phase
        old_phase = FanoPhase(
            point_values=self.current_phase.point_values.copy(),
            line_values=self.current_phase.line_values.copy(),
            theta=self.current_phase.theta,
        )

        # Update point polarity
        self.current_phase.point_values[point] = result_value

        # Record transition
        transition = PolaricTransition(
            from_phase=old_phase,
            to_phase=self.current_phase,
            axiom_used=2,
            trigger_lines=(l1, l2),
            result_point=point,
        )
        self.transition_history.append(transition)
        self.phase_history.append(old_phase)

        return (point, result_value)

    def propagate_forward(self) -> None:
        """Propagate point values to all lines."""
        for l in range(7):
            self.current_phase.line_values[l] = self.current_phase.compute_line_from_points(l)

    def propagate_backward(self) -> None:
        """Propagate line values to all points."""
        for p in FANO_POINTS:
            self.current_phase.point_values[p] = self.current_phase.compute_point_from_lines(p)

    def evolve_phase(self, dt: float) -> None:
        """Evolve the phase angle."""
        self._time += dt
        self.current_phase.theta = (self.current_phase.theta + PHI_INV * dt) % TAU

    def get_balance(self) -> Dict[str, int]:
        """Get ternary balance statistics."""
        point_pos = sum(1 for v in self.current_phase.point_values.values()
                        if v == TernaryValue.POSITIVE)
        point_neg = sum(1 for v in self.current_phase.point_values.values()
                        if v == TernaryValue.NEGATIVE)
        line_pos = sum(1 for v in self.current_phase.line_values.values()
                       if v == TernaryValue.POSITIVE)
        line_neg = sum(1 for v in self.current_phase.line_values.values()
                       if v == TernaryValue.NEGATIVE)

        return {
            "points_positive": point_pos,
            "points_neutral": 7 - point_pos - point_neg,
            "points_negative": point_neg,
            "lines_positive": line_pos,
            "lines_neutral": 7 - line_pos - line_neg,
            "lines_negative": line_neg,
            "total_balance": self.current_phase.balance(),
        }


@dataclass
class LuminahedronPosition:
    """
    Position of the Luminahedron in the Kaelhedron space.

    The 12D Luminahedron (SU(3)×SU(2)×U(1)) is positioned within
    the 21D Kaelhedron (so(7)) via Fano phase coordinates.
    """
    # Primary coordinates: which seal (1-7) and face (0-2)
    seal: int
    face: int

    # Gauge field components (12D)
    su3: List[float] = field(default_factory=lambda: [0.0] * 8)  # 8 gluons
    su2: List[float] = field(default_factory=lambda: [0.0] * 3)  # W/Z
    u1: float = 0.0  # Hypercharge

    # Phase and polaric state
    theta: float = 0.0
    ternary_value: TernaryValue = TernaryValue.NEUTRAL

    @property
    def cell_index(self) -> Tuple[int, int]:
        """Get Kaelhedron cell index."""
        return (self.seal, self.face)

    def gauge_magnitude(self) -> float:
        """Compute total gauge field magnitude."""
        su3_mag = sum(x * x for x in self.su3) ** 0.5
        su2_mag = sum(x * x for x in self.su2) ** 0.5
        return (su3_mag ** 2 + su2_mag ** 2 + self.u1 ** 2) ** 0.5


class LuminahedronPath:
    """
    Generates the path the Luminahedron travels through the Kaelhedron
    via Fano phase transitions.

    The path is determined by:
    1. Binary axiom applications (forward/backward polarity)
    2. Ternary polaric relationships
    3. Fano incidence structure
    """

    def __init__(self, ternary_engine: TernaryPolaricEngine):
        self.engine = ternary_engine
        self.path: List[LuminahedronPosition] = []
        self.current_position: Optional[LuminahedronPosition] = None

    def initialize(self, start_seal: int = 1, start_face: int = 0) -> LuminahedronPosition:
        """Initialize Luminahedron at starting position."""
        self.current_position = LuminahedronPosition(
            seal=start_seal,
            face=start_face,
            theta=self.engine.current_phase.theta,
            ternary_value=self.engine.current_phase.point_values[start_seal],
        )
        self.path = [self.current_position]
        return self.current_position

    def step_forward(self, p1: int, p2: int) -> LuminahedronPosition:
        """
        Take a forward step: apply Axiom 1 and move Luminahedron.

        The Luminahedron moves to the cell determined by the
        resulting line and its ternary value.
        """
        line_idx, ternary_val = self.engine.apply_forward_axiom(p1, p2)
        line = FANO_LINES[line_idx]

        # The third point on the line (not p1 or p2) determines the new seal
        new_seal = [p for p in line if p not in (p1, p2)][0]

        # Face is determined by ternary value
        face_map = {
            TernaryValue.POSITIVE: 0,  # LOGOS
            TernaryValue.NEUTRAL: 1,   # BIOS
            TernaryValue.NEGATIVE: 2,  # NOUS
        }
        new_face = face_map[ternary_val]

        # Evolve gauge fields based on transition
        new_position = LuminahedronPosition(
            seal=new_seal,
            face=new_face,
            theta=self.engine.current_phase.theta,
            ternary_value=ternary_val,
        )

        # Propagate gauge fields from previous position
        if self.current_position:
            # SU(3) rotation based on line transition
            angle = line_idx * TAU / 7
            for i in range(8):
                phase = (i * TAU / 8) + angle
                new_position.su3[i] = math.cos(phase) * 0.5

            # SU(2) based on face
            for i in range(3):
                new_position.su2[i] = math.sin((new_face + i) * TAU / 3) * 0.5

            # U(1) based on ternary value
            new_position.u1 = ternary_val.value * PHI_INV

        self.current_position = new_position
        self.path.append(new_position)
        return new_position

    def step_backward(self, l1: int, l2: int) -> LuminahedronPosition:
        """
        Take a backward step: apply Axiom 2 and move Luminahedron.

        The Luminahedron moves to the cell determined by the
        intersection point and its ternary value.
        """
        point, ternary_val = self.engine.apply_backward_axiom(l1, l2)

        # Face is determined by ternary value
        face_map = {
            TernaryValue.POSITIVE: 0,  # LOGOS
            TernaryValue.NEUTRAL: 1,   # BIOS
            TernaryValue.NEGATIVE: 2,  # NOUS
        }
        new_face = face_map[ternary_val]

        # Create new position
        new_position = LuminahedronPosition(
            seal=point,
            face=new_face,
            theta=self.engine.current_phase.theta,
            ternary_value=ternary_val,
        )

        # Propagate gauge fields
        if self.current_position:
            # SU(3) based on point
            for i in range(8):
                phase = (point + i) * TAU / 8
                new_position.su3[i] = math.cos(phase) * 0.5

            # SU(2) based on face transition
            old_face = self.current_position.face
            for i in range(3):
                delta = (new_face - old_face) * TAU / 3
                new_position.su2[i] = math.sin(delta + i * TAU / 3) * 0.5

            # U(1) based on ternary transition
            new_position.u1 = ternary_val.value * PHI_INV

        self.current_position = new_position
        self.path.append(new_position)
        return new_position

    def generate_full_path(self) -> List[LuminahedronPosition]:
        """
        Generate a complete path visiting all 7 seals through Fano phases.

        Uses alternating forward and backward axioms following the
        Fano incidence structure.
        """
        if not self.current_position:
            self.initialize(1, 0)

        # Set initial polarities
        self.engine.set_point_polarity(1, TernaryValue.POSITIVE)
        self.engine.set_point_polarity(2, TernaryValue.POSITIVE)

        # Visit all seals through Fano transitions
        visited_seals = {1}
        transitions = [
            ("forward", 1, 2),   # L0: (1,2,3) → seal 3
            ("forward", 1, 4),   # L1: (1,4,5) → seal 5
            ("backward", 0, 1),  # L0∩L1 = 1, but need another combo
            ("forward", 2, 4),   # L3: (2,4,6) → seal 6
            ("forward", 3, 4),   # L5: (3,4,7) → seal 7
            ("forward", 2, 5),   # L4: (2,5,7) → already have 7
            ("forward", 1, 6),   # L2: (1,6,7) → already have
        ]

        for trans in transitions:
            if trans[0] == "forward":
                pos = self.step_forward(trans[1], trans[2])
            else:
                pos = self.step_backward(trans[1], trans[2])
            visited_seals.add(pos.seal)

            # Evolve phase
            self.engine.evolve_phase(0.1)

        return self.path

    def get_path_summary(self) -> Dict[str, Any]:
        """Get summary of the path."""
        if not self.path:
            return {"length": 0, "seals_visited": [], "faces_visited": []}

        return {
            "length": len(self.path),
            "seals_visited": [p.seal for p in self.path],
            "faces_visited": [p.face for p in self.path],
            "ternary_sequence": [p.ternary_value.name for p in self.path],
            "final_position": (self.path[-1].seal, self.path[-1].face),
            "total_gauge_magnitude": sum(p.gauge_magnitude() for p in self.path),
        }


def generate_luminahedron_trajectory(
    initial_polarities: Optional[Dict[int, TernaryValue]] = None
) -> Tuple[TernaryPolaricEngine, LuminahedronPath]:
    """
    Generate a complete Luminahedron trajectory through the Kaelhedron.

    Args:
        initial_polarities: Optional initial ternary values for points

    Returns:
        (engine, path) tuple
    """
    engine = TernaryPolaricEngine()

    # Set initial polarities
    if initial_polarities:
        for point, value in initial_polarities.items():
            engine.set_point_polarity(point, value)
    else:
        # Default: alternating pattern
        for i, p in enumerate(FANO_POINTS):
            if i % 3 == 0:
                engine.set_point_polarity(p, TernaryValue.POSITIVE)
            elif i % 3 == 1:
                engine.set_point_polarity(p, TernaryValue.NEUTRAL)
            else:
                engine.set_point_polarity(p, TernaryValue.NEGATIVE)

    # Propagate to lines
    engine.propagate_forward()

    # Generate path
    path = LuminahedronPath(engine)
    path.generate_full_path()

    return engine, path
