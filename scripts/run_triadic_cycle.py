#!/usr/bin/env python3
"""Execute the Rosetta Bear triadic tool cycle and log manifests."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from generated_tools.triadic_rhz import (  # type: ignore
    ToolCritical0001,
    ToolCritical0002,
    ToolSupercritical0003,
)


def load_text_excerpt(path: Path, max_lines: int = 20) -> str:
    return "\n".join(path.read_text(encoding="utf-8").splitlines()[:max_lines])


def load_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def latest_manifest(path: Path) -> Path | None:
    manifests = sorted(path.glob("*.json"))
    return manifests[-1] if manifests else None


def run_cycle(docs_dir: Path, manifests_dir: Path, output_dir: Path) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    plan_excerpt = load_text_excerpt(docs_dir / "rosetta_bear_rhz_firmware_update_plan.md")
    burden_snapshot = load_json(docs_dir / "burden_tracking_simulation.json")
    cascade_snapshot = load_json(docs_dir / "phase_cascade_history.json")

    triadic_outputs: List[Dict] = []
    for tool_cls in (ToolCritical0001, ToolCritical0002, ToolSupercritical0003):
        tool = tool_cls()
        result = tool.execute()
        triadic_outputs.append(result)

    ghmp_manifest_path = latest_manifest(manifests_dir)

    payload = {
        "timestamp": timestamp,
        "plan_excerpt": plan_excerpt,
        "triadic_outputs": triadic_outputs,
        "ghmp_manifest": str(ghmp_manifest_path) if ghmp_manifest_path else None,
        "burden_snapshot": {
            "coordinate": burden_snapshot.get("coordinate"),
            "current_phase": burden_snapshot.get("metrics", {}).get("current_phase"),
        },
        "cascade_snapshot": cascade_snapshot,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / f"triadic_cycle_{timestamp}.json"
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Rosetta Bear triadic orchestration cycle")
    parser.add_argument("--docs-dir", type=Path, default=ROOT / "docs")
    parser.add_argument("--manifests-dir", type=Path, default=ROOT / "cbs_demo" / "manifests")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "generated_tools" / "triadic_rhz" / "run_logs")
    args = parser.parse_args()

    manifest_path = run_cycle(args.docs_dir, args.manifests_dir, args.output_dir)
    print(f"Triadic cycle manifest written to {manifest_path}")


if __name__ == "__main__":
    main()
