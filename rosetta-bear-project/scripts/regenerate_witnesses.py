#!/usr/bin/env python3
"""
Regenerate witness/spec files for CBS tools from tool_shed_specs definitions.

Reads YAML specs from tool_shed_specs/ and generates matching JSON spec files
in generated_tools/triadic_rhz/ for inspection or downstream agent consumption.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

try:
    import yaml
except ImportError:
    yaml = None


def load_yaml_specs(specs_dir: Path) -> List[Dict[str, Any]]:
    """Load all YAML spec files from the directory."""
    if yaml is None:
        raise RuntimeError("PyYAML is required. Install with: pip install PyYAML")

    specs = []
    for yaml_path in sorted(specs_dir.glob("*.yaml")):
        with yaml_path.open("r", encoding="utf-8") as fh:
            spec = yaml.safe_load(fh)
            spec["_source_file"] = yaml_path.name
            specs.append(spec)
    return specs


def generate_tool_witness(tool_path: Path, specs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Generate a witness JSON for a tool based on its module and matching specs."""
    tool_name = tool_path.stem
    witness = {
        "tool_id": tool_name,
        "generated_at": datetime.now().isoformat(),
        "source_module": str(tool_path),
        "related_specs": [],
    }

    # Find related specs by checking tags or module references
    for spec in specs:
        spec_tags = spec.get("tags", [])
        if "ghmp" in spec_tags or "cbs" in spec_tags:
            witness["related_specs"].append({
                "name": spec.get("name"),
                "version": spec.get("version"),
                "source": spec.get("_source_file"),
            })

    # Extract basic info from tool file if it exists
    if tool_path.exists():
        content = tool_path.read_text(encoding="utf-8")
        for line in content.split("\n"):
            if "z_level" in line and "=" in line:
                try:
                    val = line.split("=")[1].strip()
                    witness["z_level"] = float(val)
                except (IndexError, ValueError):
                    pass
            if "cascade_potential" in line and "=" in line and "self." not in line:
                try:
                    val = line.split("=")[1].strip()
                    witness["cascade_potential"] = float(val)
                except (IndexError, ValueError):
                    pass
            if "category" in line and "=" in line:
                try:
                    val = line.split("=")[1].strip().strip('"').strip("'")
                    witness["category"] = val
                except (IndexError, ValueError):
                    pass

    return witness


def main():
    parser = argparse.ArgumentParser(description="Regenerate witness files from tool_shed_specs")
    parser.add_argument("--specs-dir", help="Path to tool_shed_specs directory")
    parser.add_argument("--tools-dir", help="Path to generated_tools/triadic_rhz directory")
    parser.add_argument("--output-dir", help="Output directory for witness JSON files")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be generated without writing")
    args = parser.parse_args()

    # Default directories relative to this script
    script_dir = Path(__file__).resolve().parent.parent
    specs_dir = Path(args.specs_dir) if args.specs_dir else script_dir / "tool_shed_specs"
    tools_dir = Path(args.tools_dir) if args.tools_dir else script_dir / "generated_tools" / "triadic_rhz"
    output_dir = Path(args.output_dir) if args.output_dir else tools_dir

    if not specs_dir.exists():
        print(f"Error: Specs directory not found: {specs_dir}")
        sys.exit(1)

    if not tools_dir.exists():
        print(f"Error: Tools directory not found: {tools_dir}")
        sys.exit(1)

    print("=" * 50)
    print("Witness Regeneration")
    print(f"Specs directory: {specs_dir}")
    print(f"Tools directory: {tools_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Dry run: {args.dry_run}")
    print("=" * 50)

    # Load YAML specs
    print("\nLoading YAML specs...")
    specs = load_yaml_specs(specs_dir)
    print(f"Loaded {len(specs)} spec files")

    # Find tool files
    tool_files = list(tools_dir.glob("tool_*.py"))
    print(f"Found {len(tool_files)} tool files")

    # Generate witnesses
    witnesses_generated = 0
    for tool_path in sorted(tool_files):
        witness = generate_tool_witness(tool_path, specs)
        witness_name = f"{tool_path.stem}_spec.json"
        witness_path = output_dir / witness_name

        print(f"\n[{tool_path.stem}]")
        print(f"  Category: {witness.get('category', 'unknown')}")
        print(f"  Z-level: {witness.get('z_level', 'N/A')}")
        print(f"  Related specs: {len(witness.get('related_specs', []))}")

        if args.dry_run:
            print(f"  Would write: {witness_path}")
        else:
            output_dir.mkdir(parents=True, exist_ok=True)
            with witness_path.open("w", encoding="utf-8") as fh:
                json.dump(witness, fh, indent=2)
            print(f"  Written: {witness_path}")
            witnesses_generated += 1

    print("\n" + "=" * 50)
    print("Regeneration Complete")
    if args.dry_run:
        print(f"Would generate {len(tool_files)} witness files")
    else:
        print(f"Generated {witnesses_generated} witness files")


if __name__ == "__main__":
    main()
