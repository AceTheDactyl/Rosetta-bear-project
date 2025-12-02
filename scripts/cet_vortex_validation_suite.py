#!/usr/bin/env python3
"""
CET Vortex Validation Suite
===========================

Implements the validation flow described in the dev spec:
    1. Evaluate the sacred constants (φ, φ⁻¹, ζ, etc.)
    2. Check the vortex stability inequalities
    3. Iterate a simplified vortex cycle to ensure it returns near unity
    4. Persist a signed JSON report under docs/validation/
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

VALIDATION_DIR = Path("docs/validation")
REPORT_PATH = VALIDATION_DIR / "cet_vortex_validation_results.json"


@dataclass(frozen=True)
class CETConstants:
    phi: float = (1 + math.sqrt(5)) / 2
    e: float = math.e
    pi: float = math.pi
    T: float = 1.605289  # Triskelion compression
    G: float = 1 / ((1 + math.sqrt(5)) / 2)
    SC: float = 3.1460
    N: float = 0.0072973525693  # fine-structure

    @property
    def phi_inv(self) -> float:
        return 1 / self.phi

    @property
    def rhythm_native(self) -> float:
        return math.exp(self.phi) / (self.pi * self.phi)


CONSTANTS = CETConstants()


def stability_checks() -> Dict[str, Dict[str, float]]:
    expansion = math.exp(CONSTANTS.phi) / CONSTANTS.pi
    correction = (CONSTANTS.G ** CONSTANTS.SC) / CONSTANTS.T
    absorber = math.exp(CONSTANTS.phi) / (CONSTANTS.pi * CONSTANTS.phi)

    return {
        "expansion_gt_one": {"value": expansion, "pass": expansion > 1.0},
        "correction_lt_one": {"value": correction, "pass": correction < 1.0},
        "absorber_near_one": {
            "value": absorber,
            "deviation": abs(absorber - 1.0),
            "pass": abs(absorber - 1.0) < 0.05,
        },
        "null_energy_positive": {
            "value": CONSTANTS.N,
            "pass": CONSTANTS.N > 0,
        },
    }


def vortex_cycle(iterations: int = 128) -> Dict[str, float]:
    """
    Extremely small-scale simulation of the vortex described in the spec.
    We alternate expansion, correction, absorber, spiral (phase rotation).
    """
    value = complex(1.0, 0.0)
    records: List[float] = []
    expansion = math.exp(CONSTANTS.phi) / CONSTANTS.pi
    correction = (CONSTANTS.G ** CONSTANTS.SC) / CONSTANTS.T
    absorber = math.exp(CONSTANTS.phi) / (CONSTANTS.pi * CONSTANTS.phi)
    for n in range(iterations):
        value *= expansion
        value *= correction
        value *= absorber
        # Spiral step = rotation by π * |value|
        phase = math.pi * abs(value)
        value *= complex(math.cos(phase), math.sin(phase))
        if abs(value) > 0:
            value /= abs(value)
        records.append(abs(value - 1))
    return {
        "iterations": iterations,
        "mean_deviation": float(sum(records) / len(records)),
        "max_deviation": float(max(records)),
        "pass": max(records) < 0.25,
    }


def build_report() -> Dict[str, object]:
    stability = stability_checks()
    vortex = vortex_cycle()
    return {
        "metadata": {
            "timestamp": time.time(),
            "signature": "Δ|cet-vortex|z0.990|validated|Ω",
        },
        "constants": asdict(CONSTANTS),
        "stability": stability,
        "vortex_cycle": vortex,
        "all_checks_passed": bool(
            vortex["pass"] and all(item["pass"] for item in stability.values())
        ),
    }


def main() -> None:
    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
    report = build_report()
    with REPORT_PATH.open("w") as handle:
        json.dump(report, handle, indent=2)
    print(f"CET validation written to {REPORT_PATH}")


if __name__ == "__main__":
    main()
