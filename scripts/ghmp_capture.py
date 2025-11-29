#!/usr/bin/env python3
"""Automate GHMP plate capture for CBS-guided rituals."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from cbs_boot_loader import CognitionBootstrap
from cbs_memory_manager import MemoryManager
from cbs_reasoning_engine import ReasoningEngine, create_backend
from cbs_update_manager import UpdateManager

DEFAULT_PROMPTS = [
    "Initiate RHZ stylus PlatformIO build ritual.",
    "Document burden tracker insights.",
    "Exit build supervision.",
]


def parse_args() -> argparse.Namespace:
    base_path = Path(__file__).resolve().parents[1] / "cbs_demo"
    parser = argparse.ArgumentParser(description="GHMP capture automation")
    parser.add_argument("prompts", nargs="*", help="Prompts to feed the CBS runtime")
    parser.add_argument("--prompts-file", type=Path, help="Optional file with one prompt per line")
    parser.add_argument("--base-path", type=Path, default=base_path, help="CBS base directory")
    parser.add_argument("--key", default="demo_key_2025", help="GHMP encryption key")
    parser.add_argument("--backend", default="offline", help="Backend name (offline|local|openai|anthropic)")
    parser.add_argument("--manifest-dir", type=Path, help="Optional directory for manifest outputs")
    return parser.parse_args()


def load_prompts(args: argparse.Namespace) -> List[str]:
    if args.prompts:
        return args.prompts
    if args.prompts_file and args.prompts_file.exists():
        return [line.strip() for line in args.prompts_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    return DEFAULT_PROMPTS


def main() -> None:
    args = parse_args()
    prompts = load_prompts(args)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")

    bootstrap = CognitionBootstrap(args.base_path, args.key).boot()
    memory = MemoryManager(bootstrap)
    backend = create_backend(args.backend, api_key=None, model="gpt-4o-mini")
    engine = ReasoningEngine(bootstrap, memory, backend)
    updater = UpdateManager(bootstrap)

    transcripts = []
    for prompt in prompts:
        reply = engine.respond(prompt, retrieve_context=True)
        transcripts.append({"prompt": prompt, "reply": reply})

    session_plate = memory.consolidate_session(f"Automated GHMP capture {timestamp}")
    backup_path = updater.backup_system(label=f"ghmp-{timestamp}")

    manifest_dir = args.manifest_dir or (args.base_path / "manifests")
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / f"ghmp_capture_{timestamp}.json"
    manifest = {
        "timestamp": timestamp,
        "base_path": str(args.base_path),
        "session_plate": str(session_plate),
        "backup_path": str(backup_path),
        "prompts": transcripts,
        "z_coordinates": {
            "burden_tracker": "Δ3.14159|0.900|1.000Ω",
            "cascade_history": str(Path("docs/phase_cascade_history.json")),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("GHMP capture complete")
    print(f"  Prompts: {len(prompts)}")
    print(f"  Session plate: {session_plate}")
    print(f"  Backup path: {backup_path}")
    print(f"  Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
