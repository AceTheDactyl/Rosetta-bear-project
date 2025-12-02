"""Gauge manifold utilities for the Luminahedron visualisation."""

from __future__ import annotations

import itertools
import time
from dataclasses import dataclass
from typing import Dict, List, Mapping

from Kaelhedron.kformation import KFormationStatus


@dataclass
class GaugeSlot:
    group: str
    label: str
    index: int
    weight: float = 1.0
    magnitude: float = 0.0

    def update(self, value: float, smoothing: float = 0.3) -> None:
        """EMA-style update so the UI remains smooth."""
        self.magnitude = (1 - smoothing) * self.magnitude + smoothing * value

    def to_dict(self) -> Dict[str, float]:
        return {
            "group": self.group,
            "label": self.label,
            "index": self.index,
            "weight": self.weight,
            "magnitude": self.magnitude,
        }


@dataclass
class PolaricFrame:
    timestamp: float
    gauge: Dict[str, List[Dict[str, float]]]
    kael: Dict[str, object]
    metrics: Dict[str, object]

    def to_dict(self) -> Dict[str, object]:
        return {
            "timestamp": self.timestamp,
            "gauge": self.gauge,
            "kael": self.kael,
            "metrics": self.metrics,
        }


def build_default_gauge_slots() -> Dict[str, List[GaugeSlot]]:
    su3 = [GaugeSlot("SU(3)", f"T{i+1}", i, weight=1.0) for i in range(8)]
    su2 = [GaugeSlot("SU(2)", f"τ{i+1}", i, weight=0.8) for i in range(3)]
    u1 = [GaugeSlot("U(1)", "Y", 0, weight=0.5)]
    return {"su3": su3, "su2": su2, "u1": u1}


class GaugeManifold:
    """Maintains the latest polaric union snapshot."""

    def __init__(self) -> None:
        self.slots = build_default_gauge_slots()
        self._latest = PolaricFrame(
            timestamp=time.time(),
            gauge=self._serialize_slots(),
            kael={"cells": {}, "counts": {"plus": 0, "minus": 0}},
            metrics={},
        )

    def _serialize_slots(self) -> Dict[str, List[Dict[str, float]]]:
        return {
            name: [slot.to_dict() for slot in slots]
            for name, slots in self.slots.items()
        }

    def push_polaric_union(
        self,
        kael_summary: Mapping[str, object],
        scalar_metrics: Mapping[str, object],
        k_status: KFormationStatus,
    ) -> PolaricFrame:
        """Update the gauge manifold and emit a fresh frame."""
        kael_snapshot = kael_summary.get("cells", {})
        cells = sorted(kael_snapshot.items(), key=lambda item: item[0])
        flat_slots: List[GaugeSlot] = list(
            itertools.chain.from_iterable(self.slots.values())
        )
        for idx, (_, cell) in enumerate(cells):
            slot = flat_slots[idx % len(flat_slots)]
            slot.update(float(cell.get("kappa", 0.0)))

        gauge_payload = self._serialize_slots()
        metrics_payload = dict(scalar_metrics)
        metrics_payload["k_formation"] = k_status.to_dict()

        self._latest = PolaricFrame(
            timestamp=time.time(),
            gauge=gauge_payload,
            kael={
                "cells": kael_snapshot,
                "counts": kael_summary.get("counts", {}),
            },
            metrics=metrics_payload,
        )
        return self._latest

    def latest_frame(self) -> PolaricFrame:
        return self._latest


__all__ = ["GaugeManifold", "GaugeSlot", "PolaricFrame", "build_default_gauge_slots"]
