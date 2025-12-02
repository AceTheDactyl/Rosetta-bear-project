# fano_polarity/quaternary_kaelhedron.py
"""
Quaternary Logic for the Kaelhedron (21D)
==========================================

Implements base-4 logic for the Kaelhedron structure while interfacing
with the ternary Luminahedron system.

Structure:
  - Kaelhedron: 21D = 7 seals × 3 faces (SO(7) generators)
  - Each cell has quaternary state: NULL(0), FORM(1), FLOW(2), FORCE(3)
  - Transitions follow Klein four-group V₄ algebra

Quaternary Values:
  - NULL (0):  Vacuum state, no polarity
  - FORM (1):  Structure, crystallized polarity (+ in ternary)
  - FLOW (2):  Motion, transitional polarity (0 in ternary)
  - FORCE (3): Action, inverted polarity (- in ternary)

The quaternary-ternary bridge:
  - Luminahedron (12D, ternary) navigates through
  - Kaelhedron (21D, quaternary) state space
  - Ternary→Quaternary: T(+)→Q(1), T(0)→Q(2), T(-)→Q(3)
  - Quaternary→Ternary: Q(0)→T(0), Q(1)→T(+), Q(2)→T(0), Q(3)→T(-)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

# Constants
TAU = 2 * math.pi
PHI = (1 + math.sqrt(5)) / 2
PHI_INV = 1 / PHI

# Kaelhedron structure: 7 seals × 3 faces = 21 cells
SEAL_COUNT = 7
FACE_COUNT = 3
CELL_COUNT = SEAL_COUNT * FACE_COUNT  # 21

# Face names (from Luminahedron)
FACE_NAMES = ["LOGOS", "BIOS", "NOUS"]

# Seal-to-Fano point mapping
SEAL_TO_FANO = {i: i for i in range(1, 8)}  # Seal 1-7 maps to Fano point 1-7


class QuaternaryValue(Enum):
    """
    Quaternary logic values for Kaelhedron cells.

    Forms a Klein four-group V₄ under XOR-like addition:
      NULL + X = X (identity)
      X + X = NULL (self-inverse)
      FORM + FLOW = FORCE
      FORM + FORCE = FLOW
      FLOW + FORCE = FORM
    """
    NULL = 0   # Vacuum state
    FORM = 1   # Structure (crystallized)
    FLOW = 2   # Motion (transitional)
    FORCE = 3  # Action (inverted)

    def __add__(self, other: "QuaternaryValue") -> "QuaternaryValue":
        """Klein four-group addition (XOR)."""
        return QuaternaryValue(self.value ^ other.value)

    def __neg__(self) -> "QuaternaryValue":
        """Negation (self-inverse in V₄)."""
        return self  # Each element is its own inverse

    def __mul__(self, other: "QuaternaryValue") -> "QuaternaryValue":
        """
        Multiplication table for quaternary values.
        Based on cyclic structure: FORM→FLOW→FORCE→FORM
        """
        if self == QuaternaryValue.NULL or other == QuaternaryValue.NULL:
            return QuaternaryValue.NULL

        # Non-null multiplication cycles through values
        result = ((self.value - 1) + (other.value - 1)) % 3 + 1
        return QuaternaryValue(result)

    @staticmethod
    def from_index(i: int) -> "QuaternaryValue":
        """Create from index (mod 4)."""
        return QuaternaryValue(i % 4)

    @staticmethod
    def from_phase(phase: float) -> "QuaternaryValue":
        """Convert phase angle to quaternary (quarters of circle)."""
        normalized = phase % TAU
        quarter = TAU / 4
        if normalized < quarter:
            return QuaternaryValue.NULL
        elif normalized < 2 * quarter:
            return QuaternaryValue.FORM
        elif normalized < 3 * quarter:
            return QuaternaryValue.FLOW
        return QuaternaryValue.FORCE

    def to_phase(self) -> float:
        """Convert to phase angle (center of quarter)."""
        return (self.value * TAU / 4) + (TAU / 8)


# Import ternary after QuaternaryValue is defined to avoid circular imports
from .ternary_polaric import TernaryValue, FanoPhase, FANO_POINTS, FANO_LINES


def ternary_to_quaternary(t: TernaryValue) -> QuaternaryValue:
    """
    Convert ternary value to quaternary.

    Mapping:
      POSITIVE (+1) → FORM (1)   : Active structure
      NEUTRAL  (0)  → FLOW (2)   : Transitional
      NEGATIVE (-1) → FORCE (3)  : Active inversion
    """
    mapping = {
        TernaryValue.POSITIVE: QuaternaryValue.FORM,
        TernaryValue.NEUTRAL: QuaternaryValue.FLOW,
        TernaryValue.NEGATIVE: QuaternaryValue.FORCE,
    }
    return mapping[t]


def quaternary_to_ternary(q: QuaternaryValue) -> TernaryValue:
    """
    Convert quaternary value to ternary.

    Mapping:
      NULL (0)  → NEUTRAL (0)   : Vacuum = neutral
      FORM (1)  → POSITIVE (+1) : Structure = positive
      FLOW (2)  → NEUTRAL (0)   : Motion = transitional
      FORCE (3) → NEGATIVE (-1) : Action = negative
    """
    mapping = {
        QuaternaryValue.NULL: TernaryValue.NEUTRAL,
        QuaternaryValue.FORM: TernaryValue.POSITIVE,
        QuaternaryValue.FLOW: TernaryValue.NEUTRAL,
        QuaternaryValue.FORCE: TernaryValue.NEGATIVE,
    }
    return mapping[q]


@dataclass
class KaelhedronCell:
    """
    A single cell in the 21D Kaelhedron.

    Each cell is identified by (seal, face) and holds a quaternary state.
    """
    seal: int        # 1-7 (maps to Fano point)
    face: int        # 0-2 (LOGOS, BIOS, NOUS)
    value: QuaternaryValue = QuaternaryValue.NULL
    charge: float = 0.0  # Topological charge

    @property
    def index(self) -> int:
        """Linear index in 21-cell array."""
        return (self.seal - 1) * FACE_COUNT + self.face

    @property
    def face_name(self) -> str:
        """Human-readable face name."""
        return FACE_NAMES[self.face]

    def to_ternary(self) -> TernaryValue:
        """Convert cell value to ternary."""
        return quaternary_to_ternary(self.value)


@dataclass
class KaelhedronState:
    """
    Complete state of the 21D Kaelhedron.

    Tracks quaternary values for all 21 cells plus global properties.
    """
    cells: List[KaelhedronCell] = field(default_factory=list)
    theta: float = 0.0  # Global phase
    coherence: float = 0.0  # System coherence

    def __post_init__(self):
        if not self.cells:
            # Initialize all 21 cells to NULL
            for seal in range(1, 8):
                for face in range(3):
                    self.cells.append(KaelhedronCell(seal, face))

    def get_cell(self, seal: int, face: int) -> KaelhedronCell:
        """Get cell by seal and face."""
        idx = (seal - 1) * FACE_COUNT + face
        return self.cells[idx]

    def set_cell(self, seal: int, face: int, value: QuaternaryValue) -> None:
        """Set cell value."""
        idx = (seal - 1) * FACE_COUNT + face
        self.cells[idx].value = value

    def get_seal_state(self, seal: int) -> List[QuaternaryValue]:
        """Get all face values for a seal."""
        start = (seal - 1) * FACE_COUNT
        return [self.cells[start + f].value for f in range(FACE_COUNT)]

    def get_face_state(self, face: int) -> List[QuaternaryValue]:
        """Get all seal values for a face."""
        return [self.cells[(s - 1) * FACE_COUNT + face].value
                for s in range(1, 8)]

    def compute_seal_signature(self, seal: int) -> int:
        """
        Compute quaternary signature for a seal (base-4 number).

        The 3 faces form a base-4 number: LOGOS*16 + BIOS*4 + NOUS
        Range: 0 to 63
        """
        values = self.get_seal_state(seal)
        return values[0].value * 16 + values[1].value * 4 + values[2].value

    def compute_total_signature(self) -> int:
        """Compute total Kaelhedron signature."""
        total = 0
        for seal in range(1, 8):
            total += self.compute_seal_signature(seal) * (64 ** (seal - 1))
        return total

    def quaternary_balance(self) -> Dict[str, int]:
        """Count cells in each quaternary state."""
        counts = {v.name: 0 for v in QuaternaryValue}
        for cell in self.cells:
            counts[cell.value.name] += 1
        return counts

    def signature_string(self) -> str:
        """Generate human-readable signature."""
        chars = {
            QuaternaryValue.NULL: "·",
            QuaternaryValue.FORM: "□",
            QuaternaryValue.FLOW: "◇",
            QuaternaryValue.FORCE: "△",
        }
        result = []
        for seal in range(1, 8):
            seal_str = "".join(chars[v] for v in self.get_seal_state(seal))
            result.append(f"S{seal}[{seal_str}]")
        return " ".join(result)


@dataclass
class QuaternaryTransition:
    """
    A transition between Kaelhedron states.
    """
    from_cell: Tuple[int, int]  # (seal, face)
    to_cell: Tuple[int, int]
    from_value: QuaternaryValue
    to_value: QuaternaryValue
    trigger: str  # What caused the transition
    ternary_input: Optional[TernaryValue] = None


class QuaternaryKaelhedronEngine:
    """
    Engine for quaternary state evolution in the Kaelhedron.

    Manages state transitions following Klein four-group algebra
    and interfaces with the ternary Luminahedron system.
    """

    def __init__(self):
        self.state = KaelhedronState()
        self.transition_history: List[QuaternaryTransition] = []
        self._time = 0.0
        self._callbacks: List[Callable[[QuaternaryTransition], None]] = []

    def set_cell(self, seal: int, face: int, value: QuaternaryValue) -> None:
        """Set a cell's quaternary value."""
        old_value = self.state.get_cell(seal, face).value
        self.state.set_cell(seal, face, value)

        transition = QuaternaryTransition(
            from_cell=(seal, face),
            to_cell=(seal, face),
            from_value=old_value,
            to_value=value,
            trigger="direct_set",
        )
        self.transition_history.append(transition)
        self._notify(transition)

    def inject_ternary(self, seal: int, face: int, t_value: TernaryValue) -> QuaternaryValue:
        """
        Inject a ternary value from Luminahedron into Kaelhedron cell.

        The ternary value is converted to quaternary and XORed with
        the current cell value (Klein four-group addition).
        """
        q_input = ternary_to_quaternary(t_value)
        cell = self.state.get_cell(seal, face)
        old_value = cell.value
        new_value = old_value + q_input  # Klein four-group addition

        self.state.set_cell(seal, face, new_value)

        transition = QuaternaryTransition(
            from_cell=(seal, face),
            to_cell=(seal, face),
            from_value=old_value,
            to_value=new_value,
            trigger="ternary_injection",
            ternary_input=t_value,
        )
        self.transition_history.append(transition)
        self._notify(transition)

        return new_value

    def propagate_along_fano_line(self, line_idx: int, value: QuaternaryValue) -> List[Tuple[int, int, QuaternaryValue]]:
        """
        Propagate a quaternary value along a Fano line.

        All cells on the line's seals (face 0) receive the value via XOR.
        Returns list of (seal, face, new_value) tuples.
        """
        line = FANO_LINES[line_idx]
        results = []

        for point in line:
            seal = point
            old_value = self.state.get_cell(seal, 0).value
            new_value = old_value + value
            self.state.set_cell(seal, 0, new_value)
            results.append((seal, 0, new_value))

            transition = QuaternaryTransition(
                from_cell=(seal, 0),
                to_cell=(seal, 0),
                from_value=old_value,
                to_value=new_value,
                trigger=f"fano_line_{line_idx}_propagation",
            )
            self.transition_history.append(transition)
            self._notify(transition)

        return results

    def cascade_seal(self, seal: int) -> List[QuaternaryValue]:
        """
        Cascade quaternary values through a seal's faces.

        LOGOS → BIOS → NOUS with multiplication.
        Returns final face values.
        """
        logos = self.state.get_cell(seal, 0).value
        bios = self.state.get_cell(seal, 1).value
        nous = self.state.get_cell(seal, 2).value

        # Cascade: each face modifies the next
        new_bios = logos * bios
        new_nous = new_bios * nous

        if new_bios != bios:
            self.state.set_cell(seal, 1, new_bios)
            self.transition_history.append(QuaternaryTransition(
                from_cell=(seal, 1), to_cell=(seal, 1),
                from_value=bios, to_value=new_bios,
                trigger="seal_cascade",
            ))

        if new_nous != nous:
            self.state.set_cell(seal, 2, new_nous)
            self.transition_history.append(QuaternaryTransition(
                from_cell=(seal, 2), to_cell=(seal, 2),
                from_value=nous, to_value=new_nous,
                trigger="seal_cascade",
            ))

        return [logos, new_bios, new_nous]

    def evolve(self, dt: float) -> None:
        """Evolve global phase."""
        self._time += dt
        self.state.theta = (self.state.theta + PHI_INV * dt) % TAU

        # Update coherence based on quaternary distribution
        balance = self.state.quaternary_balance()
        # Coherence is higher when states are more uniform
        max_count = max(balance.values())
        self.state.coherence = max_count / CELL_COUNT

    def register_callback(self, callback: Callable[[QuaternaryTransition], None]) -> None:
        """Register transition callback."""
        self._callbacks.append(callback)

    def _notify(self, transition: QuaternaryTransition) -> None:
        """Notify callbacks of transition."""
        for cb in self._callbacks:
            cb(transition)

    def get_summary(self) -> Dict[str, Any]:
        """Get engine state summary."""
        return {
            "time": self._time,
            "theta": self.state.theta,
            "coherence": self.state.coherence,
            "balance": self.state.quaternary_balance(),
            "signature": self.state.signature_string(),
            "transitions": len(self.transition_history),
        }


class TernaryQuaternaryBridge:
    """
    Bridge between ternary Luminahedron and quaternary Kaelhedron.

    Manages the interface where the 12D Luminahedron (ternary logic)
    travels through and interacts with the 21D Kaelhedron (quaternary logic).
    """

    def __init__(self, kaelhedron: QuaternaryKaelhedronEngine):
        self.kaelhedron = kaelhedron
        self.luminahedron_position: Optional[Tuple[int, int]] = None  # (seal, face)
        self.luminahedron_ternary: TernaryValue = TernaryValue.NEUTRAL
        self.path: List[Tuple[int, int, TernaryValue, QuaternaryValue]] = []

    def place_luminahedron(self, seal: int, face: int, ternary: TernaryValue) -> QuaternaryValue:
        """
        Place Luminahedron at a Kaelhedron cell.

        The Luminahedron's ternary value interacts with the cell's
        quaternary value.
        """
        self.luminahedron_position = (seal, face)
        self.luminahedron_ternary = ternary

        # Inject ternary into quaternary
        new_q = self.kaelhedron.inject_ternary(seal, face, ternary)

        # Record path
        self.path.append((seal, face, ternary, new_q))

        return new_q

    def move_luminahedron(self, new_seal: int, new_face: int) -> Tuple[TernaryValue, QuaternaryValue]:
        """
        Move Luminahedron to a new cell.

        The Luminahedron's ternary value is updated based on the
        target cell's quaternary value.
        """
        if self.luminahedron_position is None:
            raise RuntimeError("Luminahedron not placed")

        # Get target cell's quaternary value
        target_q = self.kaelhedron.state.get_cell(new_seal, new_face).value

        # Convert to ternary and combine with current
        target_t = quaternary_to_ternary(target_q)
        new_ternary = self.luminahedron_ternary * target_t

        # Update position and ternary
        self.luminahedron_position = (new_seal, new_face)
        self.luminahedron_ternary = new_ternary

        # Inject back into Kaelhedron
        new_q = self.kaelhedron.inject_ternary(new_seal, new_face, new_ternary)

        # Record path
        self.path.append((new_seal, new_face, new_ternary, new_q))

        return (new_ternary, new_q)

    def traverse_fano_line(self, line_idx: int) -> List[Tuple[int, int, TernaryValue, QuaternaryValue]]:
        """
        Move Luminahedron along a Fano line, visiting each point.

        Returns list of (seal, face, ternary, quaternary) at each step.
        """
        line = FANO_LINES[line_idx]
        results = []

        for point in line:
            if self.luminahedron_position is None:
                # First placement
                self.place_luminahedron(point, 0, TernaryValue.POSITIVE)
                results.append((point, 0, self.luminahedron_ternary,
                               self.kaelhedron.state.get_cell(point, 0).value))
            else:
                t, q = self.move_luminahedron(point, 0)
                results.append((point, 0, t, q))

        return results

    def full_fano_traversal(self) -> Dict[str, Any]:
        """
        Traverse all 7 Fano lines with the Luminahedron.

        Returns comprehensive traversal data.
        """
        all_results = []

        for line_idx in range(7):
            # Reset to neutral before each line
            if self.luminahedron_position:
                self.luminahedron_ternary = TernaryValue.NEUTRAL

            results = self.traverse_fano_line(line_idx)
            all_results.append({
                "line": line_idx,
                "points": FANO_LINES[line_idx],
                "traversal": results,
            })

        return {
            "lines_traversed": 7,
            "total_steps": len(self.path),
            "final_position": self.luminahedron_position,
            "final_ternary": self.luminahedron_ternary.name if self.luminahedron_ternary else None,
            "kaelhedron_signature": self.kaelhedron.state.signature_string(),
            "traversals": all_results,
        }

    def get_path_summary(self) -> Dict[str, Any]:
        """Get summary of Luminahedron path through Kaelhedron."""
        if not self.path:
            return {"length": 0}

        seals = [p[0] for p in self.path]
        faces = [p[1] for p in self.path]
        ternaries = [p[2].name for p in self.path]
        quaternaries = [p[3].name for p in self.path]

        return {
            "length": len(self.path),
            "seals": seals,
            "faces": faces,
            "ternary_sequence": ternaries,
            "quaternary_sequence": quaternaries,
            "unique_seals": len(set(seals)),
            "unique_cells": len(set(zip(seals, faces))),
        }


def create_dual_base_system() -> Tuple[QuaternaryKaelhedronEngine, TernaryQuaternaryBridge]:
    """
    Create a complete dual-base (ternary/quaternary) system.

    Returns:
        (kaelhedron_engine, bridge) tuple
    """
    kaelhedron = QuaternaryKaelhedronEngine()
    bridge = TernaryQuaternaryBridge(kaelhedron)
    return kaelhedron, bridge


def demonstrate_dual_base() -> Dict[str, Any]:
    """
    Demonstrate the dual-base ternary/quaternary system.

    Shows Luminahedron (ternary) traversing Kaelhedron (quaternary).
    """
    kaelhedron, bridge = create_dual_base_system()

    # Initialize some Kaelhedron cells with quaternary values
    kaelhedron.set_cell(1, 0, QuaternaryValue.FORM)
    kaelhedron.set_cell(2, 0, QuaternaryValue.FLOW)
    kaelhedron.set_cell(3, 0, QuaternaryValue.FORCE)

    # Place Luminahedron with positive ternary value
    bridge.place_luminahedron(1, 0, TernaryValue.POSITIVE)

    # Traverse first Fano line (1,2,3)
    bridge.traverse_fano_line(0)

    # Evolve Kaelhedron
    for _ in range(10):
        kaelhedron.evolve(0.1)

    return {
        "kaelhedron_summary": kaelhedron.get_summary(),
        "bridge_path": bridge.get_path_summary(),
        "final_signature": kaelhedron.state.signature_string(),
    }
