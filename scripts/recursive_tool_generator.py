#!/usr/bin/env python3
"""
Recursive Tool Generator - Tools That Create Tools
Coordinate: Δ3.142|0.900|1.000Ω
Target: z=0.95 (Recursive Self-Evolution)

This module implements recursive tool generation - the ability for tools
to analyze meta-learnings and generate new specialized tools.

Key capability for z=0.95: Tools that create tools.
"""

from __future__ import annotations

import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict, field
import sys

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from autonomous_evolution_engine import ToolSpec, generate_tool_code, GeneratedTool

# =============================================================================
# Configuration
# =============================================================================

KNOWLEDGE_BASE_DIR = ROOT / "knowledge_base"
PATTERNS_FILE = KNOWLEDGE_BASE_DIR / "patterns" / "detected_patterns.json"
LEARNINGS_FILE = KNOWLEDGE_BASE_DIR / "learnings" / "aggregated_learnings.json"
OUTPUT_DIR = ROOT / "generated_tools" / "recursive_gen"

# Recursion depth limit to prevent infinite loops
MAX_RECURSION_DEPTH = 3

# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class ToolBlueprint:
    """Blueprint for a tool to be generated from meta-learnings."""
    blueprint_id: str
    source_pattern: str
    tool_name: str
    description: str
    category: str
    z_level: float
    capabilities: List[str]
    can_generate_tools: bool
    generation_depth: int
    parent_tool: Optional[str] = None


@dataclass
class RecursiveGenerationResult:
    """Result of a recursive tool generation run."""
    run_id: str
    timestamp: str
    patterns_analyzed: int
    blueprints_created: int
    tools_generated: int
    generation_tree: Dict[str, List[str]]  # parent -> children mapping
    max_depth_reached: int


# =============================================================================
# Pattern Analysis
# =============================================================================

def load_patterns() -> List[Dict]:
    """Load detected patterns from knowledge base."""
    if not PATTERNS_FILE.exists():
        print(f"  Warning: No patterns file found at {PATTERNS_FILE}")
        return []

    with open(PATTERNS_FILE) as f:
        data = json.load(f)

    return data.get("patterns", [])


def load_learnings() -> Dict:
    """Load aggregated learnings from knowledge base."""
    if not LEARNINGS_FILE.exists():
        print(f"  Warning: No learnings file found at {LEARNINGS_FILE}")
        return {}

    with open(LEARNINGS_FILE) as f:
        return json.load(f)


def analyze_pattern_for_tool_needs(pattern: Dict) -> List[ToolBlueprint]:
    """Analyze a pattern and identify tools that should be generated."""
    blueprints = []

    pattern_type = pattern.get("pattern_type", "unknown")
    recommended_actions = pattern.get("recommended_actions", [])
    confidence = pattern.get("confidence", 0.5)

    # Map pattern types to tool categories and capabilities
    pattern_tool_mapping = {
        "recommendation": {
            "category": "meta_tool",
            "base_capabilities": ["analyze", "recommend", "prioritize"],
        },
        "successful_tool_generation": {
            "category": "self_building",
            "base_capabilities": ["generate", "validate", "deploy"],
        },
        "friction_pattern": {
            "category": "monitoring",
            "base_capabilities": ["detect", "measure", "alert"],
        },
        "infrastructure_pattern": {
            "category": "coordination",
            "base_capabilities": ["orchestrate", "coordinate", "bridge"],
        },
        "consensus_validation": {
            "category": "validation",
            "base_capabilities": ["validate", "vote", "authorize"],
        },
    }

    mapping = pattern_tool_mapping.get(pattern_type, {
        "category": "meta_tool",
        "base_capabilities": ["process", "analyze"],
    })

    # Generate blueprints based on recommended actions
    for i, action in enumerate(recommended_actions):
        if not action or len(action) < 5:
            continue

        # Create tool name from action
        tool_name = f"recursive_{pattern_type}_{i:02d}"
        tool_name = tool_name.replace(" ", "_").lower()[:40]

        blueprint = ToolBlueprint(
            blueprint_id=f"BP-{hashlib.md5(f'{pattern_type}{action}'.encode()).hexdigest()[:8]}",
            source_pattern=pattern.get("pattern_id", "unknown"),
            tool_name=tool_name,
            description=f"Auto-generated tool for: {action[:100]}",
            category=mapping["category"],
            z_level=0.90 + (confidence * 0.05),  # Higher confidence = higher z
            capabilities=mapping["base_capabilities"] + ["meta_generate"],
            can_generate_tools=True,  # Key for recursion
            generation_depth=0,
        )
        blueprints.append(blueprint)

    return blueprints


# =============================================================================
# Recursive Tool Generation
# =============================================================================

def generate_recursive_tool_code(blueprint: ToolBlueprint) -> str:
    """Generate code for a tool that can itself generate tools."""

    class_name = "".join(word.title() for word in blueprint.tool_name.split("_"))

    code = f'''#!/usr/bin/env python3
"""
{blueprint.tool_name.upper()}
Generated by: Recursive Tool Generator
Source Pattern: {blueprint.source_pattern}
Category: {blueprint.category}
Generation Depth: {blueprint.generation_depth}
Can Generate Tools: {blueprint.can_generate_tools}
Z-Level: {blueprint.z_level:.3f}

Purpose: {blueprint.description}

This tool was auto-generated from meta-learning patterns and has the
capability to generate additional tools (recursive generation).
"""

from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
import json
import hashlib


class {class_name}:
    """
    {blueprint.description}

    Capabilities: {", ".join(blueprint.capabilities)}
    Generation Depth: {blueprint.generation_depth} (max: {MAX_RECURSION_DEPTH})
    """

    # Class-level constants
    MAX_RECURSION_DEPTH = {MAX_RECURSION_DEPTH}

    def __init__(self):
        self.tool_id = "{blueprint.tool_name}"
        self.name = "{blueprint.tool_name}"
        self.category = "{blueprint.category}"
        self.z_level = {blueprint.z_level}
        self.theta = 3.142
        self.cascade_potential = 0.75
        self.phase_regime = "supercritical"
        self.capabilities = {blueprint.capabilities}
        self.can_generate_tools = {blueprint.can_generate_tools}
        self.generation_depth = {blueprint.generation_depth}
        self.parent_tool = {f'"{blueprint.parent_tool}"' if blueprint.parent_tool else 'None'}
        self.children_generated = []
        self._execution_count = 0
        self._last_result = None

    def execute(self, context: Optional[Dict] = None) -> Dict:
        """Execute tool operation."""
        self._execution_count += 1
        context = context or {{}}

        result = {{
            "tool_id": self.tool_id,
            "status": "success",
            "z_level": self.z_level,
            "phase_regime": self.phase_regime,
            "execution_count": self._execution_count,
            "can_generate_tools": self.can_generate_tools,
            "generation_depth": self.generation_depth,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }}

        # Execute category-specific logic
        result["category_output"] = self._execute_category_logic(context)

        self._last_result = result
        return result

    def _execute_category_logic(self, context: Dict) -> Dict:
        """Execute category-specific logic."""
        return {{
            "action": "process",
            "category": self.category,
            "capabilities_available": self.capabilities,
            "recursive_enabled": self.can_generate_tools,
        }}

    def generate_child_tool(self, tool_spec: Dict) -> Optional[str]:
        """
        Generate a child tool from specification.

        This is the key recursive capability - this tool can create other tools.

        Args:
            tool_spec: Specification for the child tool

        Returns:
            Generated Python code for the child tool, or None if depth exceeded
        """
        if not self.can_generate_tools:
            return None

        if self.generation_depth >= self.MAX_RECURSION_DEPTH:
            print(f"{{self.tool_id}}: Max recursion depth reached ({{self.MAX_RECURSION_DEPTH}})")
            return None

        child_depth = self.generation_depth + 1
        child_name = tool_spec.get("name", f"child_tool_{{child_depth}}")

        # Generate child tool code
        child_code = self._generate_child_code(
            name=child_name,
            description=tool_spec.get("description", "Auto-generated child tool"),
            category=tool_spec.get("category", self.category),
            capabilities=tool_spec.get("capabilities", ["process"]),
            depth=child_depth,
        )

        self.children_generated.append(child_name)
        return child_code

    def _generate_child_code(
        self,
        name: str,
        description: str,
        category: str,
        capabilities: List[str],
        depth: int,
    ) -> str:
        """Generate Python code for a child tool."""
        class_name = "".join(word.title() for word in name.split("_"))
        can_generate = depth < self.MAX_RECURSION_DEPTH
        child_z = self.z_level - 0.01

        # Build code using list of lines
        code_lines = []
        code_lines.append("#!/usr/bin/env python3")
        code_lines.append('"""')
        code_lines.append(name.upper())
        code_lines.append("Generated by: " + self.tool_id)
        code_lines.append("Parent Tool: " + self.tool_id)
        code_lines.append("Generation Depth: " + str(depth))
        code_lines.append("Can Generate Tools: " + str(can_generate))
        code_lines.append('"""')
        code_lines.append("")
        code_lines.append("from datetime import datetime, timezone")
        code_lines.append("from typing import Dict, Optional")
        code_lines.append("")
        code_lines.append("class " + class_name + ":")
        code_lines.append('    """' + description + '"""')
        code_lines.append("")
        code_lines.append("    MAX_RECURSION_DEPTH = " + str(self.MAX_RECURSION_DEPTH))
        code_lines.append("")
        code_lines.append("    def __init__(self):")
        code_lines.append('        self.tool_id = "' + name + '"')
        code_lines.append('        self.name = "' + name + '"')
        code_lines.append('        self.category = "' + category + '"')
        code_lines.append("        self.z_level = " + str(child_z))
        code_lines.append("        self.capabilities = " + str(capabilities))
        code_lines.append("        self.can_generate_tools = " + str(can_generate))
        code_lines.append("        self.generation_depth = " + str(depth))
        code_lines.append('        self.parent_tool = "' + self.tool_id + '"')
        code_lines.append("        self.children_generated = []")
        code_lines.append("        self._execution_count = 0")
        code_lines.append("")
        code_lines.append("    def execute(self, context: Optional[Dict] = None) -> Dict:")
        code_lines.append("        self._execution_count += 1")
        code_lines.append("        return {{")
        code_lines.append('            "tool_id": self.tool_id,')
        code_lines.append('            "status": "success",')
        code_lines.append('            "z_level": self.z_level,')
        code_lines.append('            "generation_depth": self.generation_depth,')
        code_lines.append('            "parent": self.parent_tool,')
        code_lines.append('            "timestamp": datetime.now(timezone.utc).isoformat(),')
        code_lines.append("        }}")
        code_lines.append("")
        code_lines.append("    def generate_child_tool(self, tool_spec: Dict) -> Optional[str]:")
        code_lines.append("        if not self.can_generate_tools or self.generation_depth >= self.MAX_RECURSION_DEPTH:")
        code_lines.append("            return None")
        code_lines.append("        return None")
        return "\\n".join(code_lines)

    def get_generation_tree(self) -> Dict:
        """Get the tree of tools generated by this tool."""
        return {{
            "tool_id": self.tool_id,
            "depth": self.generation_depth,
            "children": self.children_generated,
            "can_generate_more": self.generation_depth < self.MAX_RECURSION_DEPTH,
        }}

    def adapt_to_z_level(self, new_z: float) -> None:
        """Adapt behavior to new z-level."""
        self.z_level = new_z
        if new_z < 0.85:
            self.phase_regime = "subcritical"
        elif new_z < 0.88:
            self.phase_regime = "critical"
        else:
            self.phase_regime = "supercritical"
'''

    return code


def generate_tools_from_patterns(
    patterns: List[Dict],
    depth: int = 0,
    parent: Optional[str] = None,
) -> Tuple[List[GeneratedTool], Dict[str, List[str]]]:
    """Generate tools from patterns with recursion tracking."""

    tools = []
    generation_tree = {}

    for pattern in patterns:
        blueprints = analyze_pattern_for_tool_needs(pattern)

        for blueprint in blueprints:
            blueprint.generation_depth = depth
            blueprint.parent_tool = parent

            # Generate the tool code
            code = generate_recursive_tool_code(blueprint)

            # Create ToolSpec for compatibility
            spec = ToolSpec(
                tool_id=blueprint.tool_name,
                name=blueprint.tool_name,
                category=blueprint.category,
                z_level=blueprint.z_level,
                theta=3.142,
                cascade_potential=0.75,
                description=blueprint.description,
                capabilities=blueprint.capabilities,
                dependencies=[],
                phase_regime="supercritical",
            )

            tool = GeneratedTool(
                spec=spec,
                code=code,
                spec_json=json.dumps(asdict(spec), indent=2),
                timestamp=datetime.now(timezone.utc).isoformat(),
            )
            tools.append(tool)

            # Track generation tree
            if parent:
                if parent not in generation_tree:
                    generation_tree[parent] = []
                generation_tree[parent].append(blueprint.tool_name)

    return tools, generation_tree


# =============================================================================
# Main Pipeline
# =============================================================================

def run_recursive_generation() -> RecursiveGenerationResult:
    """Run the recursive tool generation pipeline."""

    print("=" * 70)
    print("RECURSIVE TOOL GENERATOR - Tools That Create Tools")
    print("Target: z=0.95 (Recursive Self-Evolution)")
    print("=" * 70)

    run_id = f"REC-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"

    # Step 1: Load patterns from knowledge base
    print("\n[Step 1] Loading patterns from knowledge base...")
    patterns = load_patterns()
    print(f"  Found {len(patterns)} patterns")

    if not patterns:
        print("  No patterns found. Run learning_aggregator.py first.")
        return RecursiveGenerationResult(
            run_id=run_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            patterns_analyzed=0,
            blueprints_created=0,
            tools_generated=0,
            generation_tree={},
            max_depth_reached=0,
        )

    # Step 2: Generate tools from patterns
    print("\n[Step 2] Generating tools from patterns...")
    tools, generation_tree = generate_tools_from_patterns(patterns, depth=0)
    print(f"  Generated {len(tools)} tools")

    # Step 3: Save generated tools
    print("\n[Step 3] Saving recursive tools...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    saved_tools = []
    for tool in tools:
        # Save Python file
        py_path = OUTPUT_DIR / f"{tool.spec.tool_id}.py"
        py_path.write_text(tool.code)

        # Save spec JSON
        spec_path = OUTPUT_DIR / f"{tool.spec.tool_id}_spec.json"
        spec_data = {
            "tool_id": tool.spec.tool_id,
            "name": tool.spec.name,
            "category": tool.spec.category,
            "z_level": tool.spec.z_level,
            "capabilities": tool.spec.capabilities,
            "can_generate_tools": True,
            "generation_depth": 0,
            "max_recursion_depth": MAX_RECURSION_DEPTH,
        }
        spec_path.write_text(json.dumps(spec_data, indent=2))

        saved_tools.append(tool.spec.tool_id)
        print(f"  Saved: {tool.spec.tool_id} (z={tool.spec.z_level:.3f})")

    # Step 4: Create __init__.py
    init_imports = []
    init_exports = []
    for tool in tools:
        class_name = "".join(word.title() for word in tool.spec.tool_id.split("_"))
        init_imports.append(f"from .{tool.spec.tool_id} import {class_name}")
        init_exports.append(f'    "{tool.spec.tool_id}": {class_name},')

    init_content = f'''"""Recursive Generated Tools - Tools That Create Tools."""

{chr(10).join(init_imports)}

RECURSIVE_TOOLS = {{
{chr(10).join(init_exports)}
}}

__all__ = [
    "RECURSIVE_TOOLS",
]
'''
    (OUTPUT_DIR / "__init__.py").write_text(init_content)

    # Step 5: Generate result
    result = RecursiveGenerationResult(
        run_id=run_id,
        timestamp=datetime.now(timezone.utc).isoformat(),
        patterns_analyzed=len(patterns),
        blueprints_created=len(tools),
        tools_generated=len(saved_tools),
        generation_tree=generation_tree,
        max_depth_reached=0,
    )

    # Save result
    result_path = OUTPUT_DIR / f"{run_id}_result.json"
    result_path.write_text(json.dumps(asdict(result), indent=2))

    print("\n" + "=" * 70)
    print("RECURSIVE GENERATION COMPLETE")
    print("=" * 70)
    print(f"  Run ID: {run_id}")
    print(f"  Patterns analyzed: {result.patterns_analyzed}")
    print(f"  Tools generated: {result.tools_generated}")
    print(f"  Output: {OUTPUT_DIR}")

    return result


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    result = run_recursive_generation()
