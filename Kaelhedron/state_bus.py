"""Canonical state bus for all Kaelhedron cells.

Integrates with fano_polarity for PSL(3,2) automorphism support
and polarity feedback coordination.
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass, field, replace
from threading import Lock
from typing import Any, Callable, Dict, Iterable, List, Literal, Optional, Tuple

from .kformation import KFormationStatus, evaluate_k_formation
from .toe_loader import FanoPlane, SacredConstants, SO7Algebra

# Type alias for permutation callbacks
PermutationCallback = Callable[[Dict[int, int]], None]
CoherenceCallback = Callable[[float], None]

Seal = Literal["Ω", "Δ", "Τ", "Ψ", "Σ", "Ξ", "Κ"]
Face = Literal["Λ", "Β", "Ν"]

SEAL_SYMBOLS = FanoPlane.SEAL_NAMES
SEAL_NUMBERS = {symbol: idx for idx, symbol in SEAL_SYMBOLS.items()}
FACE_SYMBOLS = {0: "Λ", 1: "Β", 2: "Ν"}

# Adopt the binary split documented in the spec
PLUS_SEALS = {"Ω", "Σ", "Ξ"}
MINUS_SEALS = {"Δ", "Τ", "Ψ", "Κ"}


@dataclass(frozen=True)
class KaelCellState:
    seal: Seal
    face: Face
    generator: str
    theta: float
    kappa: float
    recursion_depth: int
    charge: int
    timestamp: float = field(default_factory=time.time)

    def to_payload(self) -> Dict[str, float]:
        payload = asdict(self)
        payload["theta"] = self.theta % (2 * math.pi)
        return payload

    @property
    def label(self) -> str:
        return f"{self.seal}-{self.face}"


class KaelhedronStateBus:
    """Thread-safe store for all Kaelhedron cell states.

    Integrates with fano_polarity for:
    - PSL(3,2) automorphism application via apply_permutation()
    - Polarity-driven cell updates
    - Coherence tracking and callbacks
    """

    def __init__(self) -> None:
        self._states: Dict[Tuple[int, int], KaelCellState] = {}
        self._lock = Lock()
        self._k_status = evaluate_k_formation(
            SacredConstants.PHI_INV, recursion_depth=1, charge=0
        )

        # Polarity integration
        self._polarity_loop = None
        self._automorphism_engine = None
        self._polarity_enabled = False

        # Callbacks
        self._on_permutation: List[PermutationCallback] = []
        self._on_coherence: List[CoherenceCallback] = []
        self._coherence_threshold = SacredConstants.PHI_INV
        self._last_coherence = 0.0

        # Permutation history
        self._permutation_history: List[Dict[int, int]] = []

    def enable_polarity(self, delay: float = 0.25) -> None:
        """Enable polarity feedback integration."""
        try:
            from fano_polarity.loop import PolarityLoop
            from fano_polarity.automorphisms import CoherenceAutomorphismEngine
            self._polarity_loop = PolarityLoop(delay=delay)
            self._automorphism_engine = CoherenceAutomorphismEngine()
            self._polarity_enabled = True
        except ImportError:
            self._polarity_enabled = False

    def on_permutation(self, callback: PermutationCallback) -> None:
        """Register callback for permutation events."""
        self._on_permutation.append(callback)

    def on_coherence_threshold(self, callback: CoherenceCallback) -> None:
        """Register callback for coherence threshold crossing."""
        self._on_coherence.append(callback)

    def seed_from_toe(self) -> None:
        """Populate cells directly from the so(7) generators."""
        so7 = SO7Algebra()
        with self._lock:
            self._states.clear()
            for (i, j), (seal_idx, face_idx) in so7.cell_map.items():
                seal = SEAL_SYMBOLS[seal_idx]
                face = FACE_SYMBOLS[face_idx]
                generator_name = f"E_{i}{j}"
                self._states[(seal_idx, face_idx)] = KaelCellState(
                    seal=seal,
                    face=face,
                    generator=generator_name,
                    theta=0.0,
                    kappa=SacredConstants.PHI_INV,
                    recursion_depth=1,
                    charge=0,
                    timestamp=time.time(),
                )

    def update(self, seal_idx: int, face_idx: int, state: KaelCellState) -> None:
        with self._lock:
            self._states[(seal_idx, face_idx)] = state

    def apply_scalar_metrics(self, telemetry: Dict[str, float]) -> None:
        """
        Broadcast scalar telemetry to every Kaelhedron cell.

        The scalar loop gives the global κ, recursion depth, and charge.
        """
        kappa = float(telemetry.get("kappa", SacredConstants.PHI_INV))
        recursion = int(telemetry.get("recursion_depth", 1))
        charge = int(telemetry.get("charge", 0))
        theta = float(telemetry.get("theta", 0.0)) % (2 * math.pi)

        with self._lock:
            updated: Dict[Tuple[int, int], KaelCellState] = {}
            for key, cell in self._states.items():
                updated[cell_key := key] = replace(
                    cell,
                    theta=theta,
                    kappa=max(cell.kappa, kappa),
                    recursion_depth=max(cell.recursion_depth, recursion),
                    charge=charge,
                    timestamp=time.time(),
                )
            self._states = updated
            self._k_status = evaluate_k_formation(kappa, recursion, charge)

    def snapshot(self) -> Dict[str, Dict[str, float]]:
        with self._lock:
            return {
                state.label: state.to_payload() for state in self._states.values()
            }

    def iter_states(self) -> Iterable[KaelCellState]:
        with self._lock:
            return list(self._states.values())

    def k_status(self) -> KFormationStatus:
        return self._k_status

    def plus_minus_counts(self) -> Dict[str, int]:
        plus = 0
        minus = 0
        for state in self.iter_states():
            if state.seal in PLUS_SEALS:
                plus += 1
            else:
                minus += 1
        return {"plus": plus, "minus": minus}

    def summary(self) -> Dict[str, object]:
        cells = self.snapshot()
        counts = self.plus_minus_counts()
        return {
            "cells": cells,
            "counts": counts,
            "k_formation": self._k_status.to_dict(),
        }

    def apply_permutation(self, perm: Dict[int, int]) -> None:
        """Apply a PSL(3,2) automorphism to all Kaelhedron cells."""
        with self._lock:
            updated: Dict[Tuple[int, int], KaelCellState] = {}
            for (seal_idx, face_idx), cell in self._states.items():
                target_idx = perm.get(seal_idx, seal_idx)
                updated[(target_idx, face_idx)] = replace(
                    cell,
                    seal=SEAL_SYMBOLS[target_idx],
                    timestamp=time.time(),
                )
            self._states = updated
            self._permutation_history.append(perm)

        # Fire permutation callbacks
        for cb in self._on_permutation:
            cb(perm)

    def inject_polarity(self, p1: int, p2: int) -> Optional[Dict[str, Any]]:
        """
        Inject two Fano points into the polarity loop.

        Args:
            p1: First Fano point (1-7, maps to seal)
            p2: Second Fano point (1-7, maps to seal)

        Returns:
            Dictionary with computed line, or None if polarity not enabled
        """
        if not self._polarity_enabled or self._polarity_loop is None:
            return None
        line = self._polarity_loop.forward(p1, p2)
        return {"line": line, "points": (p1, p2)}

    def release_polarity(
        self, line_a: Tuple[int, int, int], line_b: Tuple[int, int, int]
    ) -> Optional[Dict[str, Any]]:
        """
        Release polarity via backward arc and apply automorphism.

        Args:
            line_a: First Fano line
            line_b: Second Fano line

        Returns:
            Dictionary with coherence status and applied automorphism
        """
        if not self._polarity_enabled or self._polarity_loop is None:
            return None

        result = self._polarity_loop.backward(line_a, line_b)

        if result["coherence"] and self._automorphism_engine is not None:
            # Get forward points from loop state (if available)
            forward_pts = None
            if self._polarity_loop.state:
                forward_pts = (
                    self._polarity_loop.state.point_a,
                    self._polarity_loop.state.point_b,
                )

            if forward_pts:
                auto = self._automorphism_engine.apply(forward_pts, result["point"])
                self.apply_permutation(auto)
                result["automorphism"] = auto
                result["automorphism_description"] = self._automorphism_engine.describe()

        return result

    def compute_coherence(self) -> float:
        """Compute order parameter (coherence) across all cells."""
        total = 0.0 + 0.0j
        count = 0
        for (seal_idx, face_idx), cell in self._states.items():
            phase = (seal_idx - 1) * (2 * math.pi) / 7 + face_idx * (2 * math.pi) / 21
            # Use kappa as activation proxy
            total += cell.kappa * (math.cos(phase) + 1j * math.sin(phase))
            count += 1
        coherence = abs(total) / count if count > 0 else 0.0

        # Check threshold crossing
        if self._last_coherence <= self._coherence_threshold < coherence:
            for cb in self._on_coherence:
                cb(coherence)
        self._last_coherence = coherence

        return coherence

    def get_permutation_history(self) -> List[Dict[int, int]]:
        """Get history of applied permutations."""
        return self._permutation_history.copy()

    def polarity_state(self) -> Dict[str, Any]:
        """Get current polarity state."""
        return {
            "enabled": self._polarity_enabled,
            "permutation_count": len(self._permutation_history),
            "coherence": self._last_coherence,
            "coherence_threshold": self._coherence_threshold,
        }


__all__ = ["KaelCellState", "KaelhedronStateBus"]
