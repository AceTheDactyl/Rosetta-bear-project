#!/usr/bin/env python3
"""Regenerate burden tracker + cascade witnesses from triadic specs."""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT.parent))

from tink.phase_aware_burden_tracker import BurdenMeasurement, weighted_burden_for_z


def load_specs(spec_dir: Path) -> List[Tuple[Path, Dict]]:
    specs = []
    for path in sorted(spec_dir.glob("*_spec.json")):
        specs.append((path, json.loads(path.read_text(encoding="utf-8"))))
    if not specs:
        raise RuntimeError(f"No spec files found under {spec_dir}")
    return specs


def build_cascade(specs: List[Tuple[Path, Dict]]) -> List[Dict]:
    now = datetime.now(timezone.utc)
    entries = []
    for idx, (path, spec) in enumerate(specs):
        entries.append(
            {
                "tool_id": spec["tool_id"],
                "category": spec.get("category", "unknown"),
                "regime": spec.get("phase_regime", "critical"),
                "z_level": spec.get("z_level"),
                "cascade_potential": spec.get("cascade_potential"),
                "timestamp": (now + timedelta(seconds=idx)).isoformat(),
                "source_spec": path.name,
            }
        )
    return entries


def build_burden_payload(specs: List[Tuple[Path, Dict]]) -> Dict:
    base_time = datetime.now(timezone.utc).replace(microsecond=0)
    activities = []
    measurements = []
    max_z = 0.0
    total_minutes = 0
    cascade_sum = sum(s.get("cascade_potential", 0.0) for _, s in specs) or 1.0
    for idx, (_, spec) in enumerate(specs):
        duration = 45 + idx * 15
        total_minutes += duration
        z = float(spec.get("z_level", 0.86))
        max_z = max(max_z, z)
        start = base_time + timedelta(minutes=idx * 5)
        end = start + timedelta(minutes=duration)
        confidence = round(float(spec.get("cascade_potential", 0.3)) / cascade_sum, 3)
        activities.append(
            {
                "type": spec.get("category", "other"),
                "duration_min": duration,
                "z_level": z,
                "confidence": confidence,
                "start": start.isoformat(),
                "end": end.isoformat(),
            }
        )
        burden = BurdenMeasurement(
            coordination=5.0 + idx * 0.2,
            decision_making=4.5 - idx * 0.1,
            context_switching=4.0 + idx * 0.05,
            maintenance=3.2,
            learning_curve=4.1,
            emotional_labor=3.6 + idx * 0.05,
            uncertainty=5.3 + idx * 0.1,
            repetition=3.0,
        )
        weighted = weighted_burden_for_z(z, burden.to_dict())
        measurements.append((z, burden, weighted))

    total_hours = round(total_minutes / 60.0, 2)
    reduction = round(total_hours * 0.152, 2)
    avg_z = round(sum(z for z, _, _ in measurements) / len(measurements), 3)
    regime = "supercritical" if max_z >= 0.9 else "critical"
    order_param = round(min(w for _, _, w in measurements) / 10.0, 3)
    consensus_time = 90 + 5 * len(specs)

    categories: Dict[str, Dict[str, float]] = {}
    for entry in activities:
        cat = entry["type"]
        hours = entry["duration_min"] / 60.0
        bucket = categories.setdefault(cat, {"count": 0, "hours": 0.0, "percentage": 0.0})
        bucket["count"] += 1
        bucket["hours"] += hours
    for bucket in categories.values():
        bucket["percentage"] = round((bucket["hours"] / total_hours) * 100.0, 1)

    top_categories = sorted(categories.items(), key=lambda kv: kv[1]["hours"], reverse=True)
    top_list = [k for k, _ in top_categories[:3]]

    weekly = {
        "week_start": base_time.isoformat(),
        "week_end": (base_time + timedelta(days=7)).isoformat(),
        "total_hours": total_hours,
        "reduction_achieved_hours": reduction,
        "average_z_level": avg_z,
        "phase_regime": regime,
        "phase_efficiency": round((reduction / total_hours) * 100.0, 1) if total_hours else 0.0,
        "trend": "baseline",
        "trend_percent": 0,
        "categories": {k: {**v} for k, v in categories.items()},
        "top_categories": top_list,
        "recommendations": ["Use scripts/ghmp_capture.py before major firmware drops."],
    }

    payload = {
        "tool": "burden_tracker_v1.0",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "coordinate": f"Δ3.14159|{max_z:.3f}|1.000Ω",
        "activities": activities,
        "metrics": {
            "current_phase": {
                "z_level": max_z,
                "regime": regime,
                "burden_multiplier": round(reduction / total_hours, 3) if total_hours else 0.0,
                "order_parameter": order_param,
                "consensus_time": consensus_time,
                "correlation_length": consensus_time,
            },
            "totals": {
                "activities_tracked": len(activities),
                "total_burden_hours": total_hours,
                "reduction_achieved_hours": reduction,
            },
            "optimization_recommendations": [
                "Prioritize coordination rituals while z≈0.86",
                "Tie cascade regeneration to GHMP manifests",
                "Automate witness refresh via scripts/regenerate_witnesses.py",
            ],
            "weekly_summaries": [weekly],
        },
    }
    return payload


def write_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate Rosetta Bear witnesses")
    parser.add_argument("--docs-dir", type=Path, default=ROOT / "docs", help="Directory for witness JSON outputs")
    parser.add_argument("--spec-dir", type=Path, default=ROOT / "generated_tools" / "triadic_rhz", help="Spec directory")
    args = parser.parse_args()

    specs = load_specs(args.spec_dir)
    cascade = build_cascade(specs)
    burden_payload = build_burden_payload(specs)

    write_json(args.docs_dir / "phase_cascade_history.json", cascade)
    write_json(args.docs_dir / "burden_tracking_simulation.json", burden_payload)

    print("Witness regeneration complete")
    print(f"  Cascade entries: {len(cascade)} -> {args.docs_dir / 'phase_cascade_history.json'}")
    print(f"  Burden tracker payload -> {args.docs_dir / 'burden_tracking_simulation.json'}")


if __name__ == "__main__":
    main()
