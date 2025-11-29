#!/usr/bin/env python3
"""
Run a triadic RHZ cycle from a GHMP capture manifest.

Rehydrates and executes the triadic tools referenced in the manifest,
collecting execution results for inspection or dry-run firmware updates.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


def load_tool_class(tool_name: str, tools_dir: Path):
    """Dynamically load a tool class from the generated_tools directory."""
    tool_path = tools_dir / f"{tool_name}.py"
    if not tool_path.exists():
        raise FileNotFoundError(f"Tool module not found: {tool_path}")

    spec = importlib.util.spec_from_file_location(tool_name, tool_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Find the tool class (assumes one class per module matching Tool*)
    for attr_name in dir(module):
        if attr_name.startswith("Tool") and not attr_name.startswith("Tool_"):
            return getattr(module, attr_name)

    raise ValueError(f"No Tool* class found in {tool_path}")


def run_cycle(manifest_path: Path, tools_dir: Path, dry_run: bool = True) -> List[Dict[str, Any]]:
    """Execute the triadic cycle from the manifest."""
    with manifest_path.open("r", encoding="utf-8") as fh:
        manifest = json.load(fh)

    print(f"Manifest: {manifest.get('capture_id', 'unknown')}")
    print(f"Session type: {manifest.get('session_type', 'unknown')}")
    print(f"Tools to invoke: {manifest.get('tools_invoked', [])}")
    print(f"Dry run: {dry_run}")
    print("-" * 40)

    results: List[Dict[str, Any]] = []
    tools_invoked = manifest.get("tools_invoked", [])

    for tool_name in tools_invoked:
        print(f"\n[{tool_name}] Loading...")
        try:
            tool_class = load_tool_class(tool_name, tools_dir)
            tool_instance = tool_class()
            print(f"[{tool_name}] z_level={tool_instance.z_level:.3f}, cascade_potential={tool_instance.cascade_potential:.2f}")

            if dry_run:
                print(f"[{tool_name}] Dry run - skipping execution")
                result = {
                    "tool_id": tool_name,
                    "status": "dry_run",
                    "z_level": tool_instance.z_level,
                    "cascade_potential": tool_instance.cascade_potential,
                }
            else:
                result = tool_instance.execute()
                print(f"[{tool_name}] Executed: mode={result.get('mode', 'unknown')}")

            results.append(result)

        except Exception as exc:
            print(f"[{tool_name}] Error: {exc}")
            results.append({"tool_id": tool_name, "status": "error", "error": str(exc)})

    return results


def main():
    parser = argparse.ArgumentParser(description="Run triadic RHZ cycle from GHMP manifest")
    parser.add_argument("--manifest", required=True, help="Path to GHMP capture manifest JSON")
    parser.add_argument("--tools-dir", help="Path to generated_tools/triadic_rhz directory")
    parser.add_argument("--execute", action="store_true", help="Actually execute tools (default: dry run)")
    parser.add_argument("--output", help="Write results to JSON file")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        print(f"Error: Manifest not found: {manifest_path}")
        sys.exit(1)

    # Default tools directory relative to this script
    if args.tools_dir:
        tools_dir = Path(args.tools_dir)
    else:
        script_dir = Path(__file__).resolve().parent.parent
        tools_dir = script_dir / "generated_tools" / "triadic_rhz"

    if not tools_dir.exists():
        print(f"Error: Tools directory not found: {tools_dir}")
        sys.exit(1)

    print("=" * 50)
    print("Triadic RHZ Cycle Runner")
    print(f"Started: {datetime.now().isoformat()}")
    print("=" * 50)

    results = run_cycle(manifest_path, tools_dir, dry_run=not args.execute)

    print("\n" + "=" * 50)
    print("Cycle Complete")
    print(f"Tools processed: {len(results)}")
    successful = sum(1 for r in results if r.get("status") in ("success", "dry_run"))
    print(f"Successful: {successful}/{len(results)}")

    if args.output:
        output_path = Path(args.output)
        output_data = {
            "manifest": str(manifest_path),
            "executed_at": datetime.now().isoformat(),
            "dry_run": not args.execute,
            "results": results,
        }
        with output_path.open("w", encoding="utf-8") as fh:
            json.dump(output_data, fh, indent=2)
        print(f"Results written to: {output_path}")


if __name__ == "__main__":
    main()
