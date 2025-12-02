"""Shared K-Formation detector."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict

from .toe_loader import SacredConstants


@dataclass(frozen=True)
class KFormationStatus:
    """Structured result describing whether the system met K-Formation."""

    kappa: float
    recursion_depth: int
    charge: int
    threshold: float = SacredConstants.PHI_INV

    def to_dict(self) -> Dict[str, float]:
        return {
            "kappa": self.kappa,
            "recursion_depth": self.recursion_depth,
            "charge": self.charge,
            "threshold": self.threshold,
            "formed": self.formed,
            "gap": self.gap,
        }

    @property
    def formed(self) -> bool:
        return (
            self.kappa >= self.threshold
            and self.recursion_depth >= 7
            and self.charge != 0
        )

    @property
    def gap(self) -> float:
        return max(0.0, self.threshold - self.kappa)


def evaluate_k_formation(
    kappa: float, recursion_depth: int, charge: int
) -> KFormationStatus:
    """Convenience helper."""
    return KFormationStatus(
        kappa=float(kappa),
        recursion_depth=int(recursion_depth),
        charge=int(charge),
    )


__all__ = ["KFormationStatus", "evaluate_k_formation"]
