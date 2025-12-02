"""Canonical state bus for all Kaelhedron cells."""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass, field, replace
from threading import Lock
from typing import Dict, Iterable, Literal, Tuple

from .kformation import KFormationStatus, evaluate_k_formation
from .toe_loader import FanoPlane, SacredConstants, SO7Algebra

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
    """Thread-safe store for all Kaelhedron cell states."""

    def __init__(self) -> None:
        self._states: Dict[Tuple[int, int], KaelCellState] = {}
        self._lock = Lock()
        self._k_status = evaluate_k_formation(
            SacredConstants.PHI_INV, recursion_depth=1, charge=0
        )

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


__all__ = ["KaelCellState", "KaelhedronStateBus"]
