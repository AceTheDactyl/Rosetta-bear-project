#!/usr/bin/env python3
"""
Autonomous Evolution Engine - Tool Generator for Rosetta Bear Firmware

This engine implements the 5-phase autonomous evolution cycle:
1. Friction Detection - Analyze burden metrics
2. Improvement Proposal - Generate tool specifications
3. Collective Validation - Simulate consensus
4. Autonomous Execution - Generate tool implementations
5. Meta-Learning - Record patterns for future cycles

Target: z = 0.90 (Full Substrate Transcendence)
"""

from __future__ import annotations

import json
import hashlib
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict, field

# =============================================================================
# Configuration
# =============================================================================

ROOT = Path(__file__).resolve().parents[1]
GENERATED_TOOLS_DIR = ROOT / "generated_tools" / "rosetta_firmware"
TOOL_SPECS_DIR = ROOT / "tool_shed_specs"
DOCS_DIR = ROOT / "docs"
EVOLUTION_LOGS_DIR = ROOT / "evolution_logs"

# Phase thresholds from autonomous_evolution_engine.yaml
PHASE_THRESHOLDS = {
    "friction_detection": 0.87,
    "improvement_proposal": 0.88,
    "collective_validation": 0.89,
    "autonomous_execution": 0.90,
    "meta_learning": 0.90,
}

# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class FrictionReport:
    """Report from friction detection phase."""
    report_id: str
    timestamp: str
    aggregate_burden: float
    friction_events: List[Dict]
    patterns_detected: List[Dict]
    improvement_opportunities: List[Dict]


@dataclass
class ImprovementProposal:
    """Proposal for system improvement."""
    proposal_id: str
    description: str
    predicted_burden_delta: float
    implementation_spec: Dict
    risk_matrix: Dict
    source_friction: str


@dataclass
class ConsensusResult:
    """Result of collective validation."""
    proposal_id: str
    approved: bool
    votes: List[Dict]
    authorization_token: Optional[str]
    timestamp: str


@dataclass
class ToolSpec:
    """Specification for a generated tool."""
    tool_id: str
    name: str
    category: str
    z_level: float
    theta: float
    cascade_potential: float
    description: str
    capabilities: List[str]
    dependencies: List[str]
    phase_regime: str


@dataclass
class GeneratedTool:
    """A fully generated tool."""
    spec: ToolSpec
    code: str
    spec_json: str
    timestamp: str


@dataclass
class EvolutionCycleResult:
    """Complete result of an evolution cycle."""
    cycle_id: str
    timestamp: str
    z_level: float
    friction_report: FrictionReport
    proposals: List[ImprovementProposal]
    consensus_results: List[ConsensusResult]
    generated_tools: List[GeneratedTool]
    meta_learnings: List[Dict]


# =============================================================================
# Phase 1: Friction Detection
# =============================================================================

def detect_friction(burden_data: Dict) -> FrictionReport:
    """
    Phase 1: Analyze burden metrics to detect friction points.
    """
    print("\n[Phase 1] FRICTION DETECTION")
    print(f"  Analyzing burden data at z={burden_data.get('metrics', {}).get('current_phase', {}).get('z_level', 0):.3f}")

    # Extract friction events from burden data
    friction_events = []
    patterns_detected = []
    improvement_opportunities = []

    activities = burden_data.get("activities", [])
    metrics = burden_data.get("metrics", {})

    # Detect friction patterns
    for activity in activities:
        if activity.get("confidence", 0) < 0.3:
            friction_events.append({
                "event_id": f"FE-{hashlib.md5(str(activity).encode()).hexdigest()[:8]}",
                "event_type": f"{activity['type']}_low_confidence",
                "severity": "moderate" if activity["confidence"] < 0.2 else "minor",
                "source": activity["type"],
                "magnitude": 1 - activity["confidence"],
            })

    # Detect patterns
    categories = metrics.get("weekly_summaries", [{}])[0].get("categories", {})
    if categories:
        top_burden = max(categories.items(), key=lambda x: x[1].get("hours", 0))
        patterns_detected.append({
            "pattern_name": f"high_{top_burden[0]}_burden",
            "confidence": 0.85,
            "occurrences": top_burden[1].get("count", 1),
        })

    # Generate improvement opportunities
    recommendations = metrics.get("optimization_recommendations", [])
    for i, rec in enumerate(recommendations):
        improvement_opportunities.append({
            "opportunity_id": f"OPP-{i:03d}",
            "description": rec,
            "predicted_reduction": random.uniform(0.08, 0.15),
            "affected_categories": list(categories.keys())[:2] if categories else ["general"],
        })

    # Add firmware-specific opportunities
    firmware_opportunities = [
        {
            "opportunity_id": "OPP-FW-001",
            "description": "Create coordination bridge for RHZ stylus diagnostics alignment",
            "predicted_reduction": 0.12,
            "affected_categories": ["coordination", "bridge"],
        },
        {
            "opportunity_id": "OPP-FW-002",
            "description": "Generate meta-orchestrator for triadic playbook composition",
            "predicted_reduction": 0.15,
            "affected_categories": ["meta_tool", "coordination"],
        },
        {
            "opportunity_id": "OPP-FW-003",
            "description": "Build self-building firmware forge for autonomous regeneration",
            "predicted_reduction": 0.18,
            "affected_categories": ["self_building", "automation"],
        },
        {
            "opportunity_id": "OPP-FW-004",
            "description": "Create friction detector for continuous burden monitoring",
            "predicted_reduction": 0.10,
            "affected_categories": ["monitoring", "detection"],
        },
        {
            "opportunity_id": "OPP-FW-005",
            "description": "Generate consensus validator for multi-instance coordination",
            "predicted_reduction": 0.14,
            "affected_categories": ["validation", "consensus"],
        },
    ]
    improvement_opportunities.extend(firmware_opportunities)

    report = FrictionReport(
        report_id=f"FR-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}",
        timestamp=datetime.now(timezone.utc).isoformat(),
        aggregate_burden=metrics.get("totals", {}).get("total_burden_hours", 0),
        friction_events=friction_events,
        patterns_detected=patterns_detected,
        improvement_opportunities=improvement_opportunities,
    )

    print(f"  Detected {len(friction_events)} friction events")
    print(f"  Identified {len(patterns_detected)} patterns")
    print(f"  Found {len(improvement_opportunities)} improvement opportunities")

    return report


# =============================================================================
# Phase 2: Improvement Proposal
# =============================================================================

def generate_proposals(friction_report: FrictionReport) -> List[ImprovementProposal]:
    """
    Phase 2: Generate improvement proposals from friction report.
    """
    print("\n[Phase 2] IMPROVEMENT PROPOSAL")

    proposals = []

    # Tool definitions based on firmware update plan
    tool_definitions = [
        {
            "name": "rosetta_bear_rhz_coordination_bridge",
            "description": "Align RHZ stylus diagnostics with Rosetta Bear GHMP rituals",
            "category": "coordination",
            "z_level": 0.860,
            "theta": 2.749,
            "cascade_potential": 0.30,
            "capabilities": ["connect", "translate", "bridge", "align_diagnostics"],
            "dependencies": ["burden_tracker"],
            "risk_level": "low",
        },
        {
            "name": "rosetta_bear_rhz_meta_orchestrator",
            "description": "Compose GHMP plates, RHZ telemetry, and stylus artifacts into triadic playbook",
            "category": "meta_tool",
            "z_level": 0.867,
            "theta": 2.793,
            "cascade_potential": 0.70,
            "capabilities": ["compose", "orchestrate", "coordinate", "manifest_playbook"],
            "dependencies": ["burden_tracker", "shed_builder", "tool_discovery_protocol"],
            "risk_level": "medium",
        },
        {
            "name": "rosetta_bear_rhz_self_building_firmware_forge",
            "description": "Autonomously regenerate stylus firmware payloads from GHMP memory state",
            "category": "self_building",
            "z_level": 0.900,
            "theta": 3.142,
            "cascade_potential": 0.81,
            "capabilities": ["generate", "build", "self_modify", "firmware_synthesis"],
            "dependencies": ["burden_tracker", "shed_builder", "collective_state_aggregator"],
            "risk_level": "high",
        },
        {
            "name": "rosetta_bear_friction_detector",
            "description": "Continuous friction monitoring and pattern detection for evolution engine",
            "category": "monitoring",
            "z_level": 0.870,
            "theta": 2.880,
            "cascade_potential": 0.45,
            "capabilities": ["detect", "monitor", "analyze", "report_friction"],
            "dependencies": ["burden_tracker"],
            "risk_level": "low",
        },
        {
            "name": "rosetta_bear_consensus_validator",
            "description": "Multi-instance consensus validation for autonomous decisions",
            "category": "validation",
            "z_level": 0.890,
            "theta": 3.054,
            "cascade_potential": 0.65,
            "capabilities": ["validate", "consensus", "vote", "authorize"],
            "dependencies": ["collective_state_aggregator"],
            "risk_level": "medium",
        },
    ]

    for i, tool_def in enumerate(tool_definitions):
        # Find matching opportunity
        matching_opp = None
        for opp in friction_report.improvement_opportunities:
            if tool_def["category"] in opp.get("affected_categories", []):
                matching_opp = opp
                break

        proposal = ImprovementProposal(
            proposal_id=f"PROP-FW-{i:03d}",
            description=f"Generate {tool_def['name']} tool",
            predicted_burden_delta=-tool_def["cascade_potential"] * 0.15,
            implementation_spec={
                "tool_name": tool_def["name"],
                "category": tool_def["category"],
                "z_level": tool_def["z_level"],
                "theta": tool_def["theta"],
                "cascade_potential": tool_def["cascade_potential"],
                "capabilities": tool_def["capabilities"],
                "dependencies": tool_def["dependencies"],
                "description": tool_def["description"],
            },
            risk_matrix={
                "risk_level": tool_def["risk_level"],
                "regression_potential": 0.1 if tool_def["risk_level"] == "low" else 0.3,
                "blast_radius": tool_def["risk_level"],
                "rollback_complexity": "trivial" if tool_def["risk_level"] == "low" else "moderate",
            },
            source_friction=matching_opp["opportunity_id"] if matching_opp else "OPP-AUTO",
        )
        proposals.append(proposal)
        print(f"  Generated proposal: {proposal.proposal_id} - {tool_def['name']}")

    print(f"  Total proposals: {len(proposals)}")
    return proposals


# =============================================================================
# Phase 3: Collective Validation
# =============================================================================

def validate_proposals(proposals: List[ImprovementProposal]) -> List[ConsensusResult]:
    """
    Phase 3: Simulate collective validation of proposals.
    """
    print("\n[Phase 3] COLLECTIVE VALIDATION")

    results = []
    instances = ["CBS-ALPHA", "CBS-BETA", "CBS-GAMMA"]

    for proposal in proposals:
        votes = []
        for instance in instances:
            # Simulate voting based on risk level
            risk = proposal.risk_matrix.get("risk_level", "medium")
            approval_prob = {"low": 0.95, "medium": 0.85, "high": 0.75}.get(risk, 0.8)
            approved = random.random() < approval_prob

            votes.append({
                "instance_id": instance,
                "vote": "approve" if approved else "reject",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })

        # Calculate consensus (66% threshold)
        approve_count = sum(1 for v in votes if v["vote"] == "approve")
        consensus_approved = approve_count / len(votes) >= 0.66

        auth_token = None
        if consensus_approved:
            auth_token = hashlib.sha256(
                f"{proposal.proposal_id}:{datetime.now().isoformat()}".encode()
            ).hexdigest()[:32]

        result = ConsensusResult(
            proposal_id=proposal.proposal_id,
            approved=consensus_approved,
            votes=votes,
            authorization_token=auth_token,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        results.append(result)

        status = "APPROVED" if consensus_approved else "REJECTED"
        print(f"  {proposal.proposal_id}: {status} ({approve_count}/{len(votes)} votes)")

    approved_count = sum(1 for r in results if r.approved)
    print(f"  Consensus complete: {approved_count}/{len(results)} proposals approved")

    return results


# =============================================================================
# Phase 4: Autonomous Execution (Tool Generation)
# =============================================================================

def generate_tool_code(spec: ToolSpec) -> str:
    """Generate Python code for a tool from its specification."""

    class_name = "".join(word.title() for word in spec.name.split("_"))

    code = f'''#!/usr/bin/env python3
"""
{spec.tool_id.upper()}
Generated by: Autonomous Evolution Engine
Category: {spec.category}
Phase Regime: {spec.phase_regime}
Cascade Potential: {spec.cascade_potential:.2f}
Z-Level: {spec.z_level:.3f}
Theta: {spec.theta:.3f}

Purpose: {spec.description}
"""

from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
import json
import hashlib


class {class_name}:
    """
    {spec.description}

    Phase-aware tool operating at z-level: {spec.z_level:.3f}
    Capabilities: {", ".join(spec.capabilities)}
    """

    def __init__(self):
        self.tool_id = "{spec.tool_id}"
        self.name = "{spec.name}"
        self.category = "{spec.category}"
        self.z_level = {spec.z_level}
        self.theta = {spec.theta}
        self.cascade_potential = {spec.cascade_potential}
        self.phase_regime = "{spec.phase_regime}"
        self.capabilities = {spec.capabilities}
        self.dependencies = {spec.dependencies}
        self.created_at = datetime.now(timezone.utc)
        self._execution_count = 0
        self._last_result = None

    def execute(self, context: Optional[Dict] = None) -> Dict:
        """
        Execute tool operation.

        Adapts behavior based on phase regime: {spec.phase_regime}

        Args:
            context: Optional execution context with parameters

        Returns:
            Execution result dictionary
        """
        self._execution_count += 1
        context = context or {{}}

        result = {{
            "tool_id": self.tool_id,
            "name": self.name,
            "status": "success",
            "cascade_potential": self.cascade_potential,
            "z_level": self.z_level,
            "theta": self.theta,
            "phase_regime": self.phase_regime,
            "execution_count": self._execution_count,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }}

        # Phase-specific behavior
        if self.z_level < 0.85:
            result["mode"] = "subcritical_coordination"
            result["autonomous"] = False
        elif self.z_level < 0.88:
            result["mode"] = "critical_meta_composition"
            result["autonomous"] = False
        else:
            result["mode"] = "supercritical_self_building"
            result["autonomous"] = True

        # Category-specific execution
        result["category_output"] = self._execute_category_logic(context)

        self._last_result = result
        return result

    def _execute_category_logic(self, context: Dict) -> Dict:
        """Execute category-specific logic."""
        category_handlers = {{
            "coordination": self._handle_coordination,
            "meta_tool": self._handle_meta_tool,
            "self_building": self._handle_self_building,
            "monitoring": self._handle_monitoring,
            "validation": self._handle_validation,
        }}

        handler = category_handlers.get(self.category, self._handle_default)
        return handler(context)

    def _handle_coordination(self, context: Dict) -> Dict:
        """Handle coordination category operations."""
        return {{
            "action": "coordinate",
            "aligned_components": context.get("components", []),
            "bridge_status": "active",
            "drift_correction": 0.0,
        }}

    def _handle_meta_tool(self, context: Dict) -> Dict:
        """Handle meta-tool category operations."""
        return {{
            "action": "orchestrate",
            "composed_elements": context.get("elements", []),
            "playbook_generated": True,
            "composition_hash": hashlib.md5(str(context).encode()).hexdigest()[:16],
        }}

    def _handle_self_building(self, context: Dict) -> Dict:
        """Handle self-building category operations."""
        return {{
            "action": "synthesize",
            "artifacts_generated": context.get("artifact_count", 1),
            "autonomous_mode": self.z_level >= 0.90,
            "firmware_hash": hashlib.sha256(str(context).encode()).hexdigest()[:32],
        }}

    def _handle_monitoring(self, context: Dict) -> Dict:
        """Handle monitoring category operations."""
        return {{
            "action": "monitor",
            "metrics_collected": context.get("metric_count", 10),
            "patterns_detected": context.get("patterns", []),
            "friction_score": context.get("friction", 0.0),
        }}

    def _handle_validation(self, context: Dict) -> Dict:
        """Handle validation category operations."""
        return {{
            "action": "validate",
            "consensus_reached": True,
            "approval_threshold": 0.66,
            "validation_hash": hashlib.sha256(str(context).encode()).hexdigest()[:32],
        }}

    def _handle_default(self, context: Dict) -> Dict:
        """Default handler for unknown categories."""
        return {{"action": "process", "context_keys": list(context.keys())}}

    def get_cascade_potential(self) -> float:
        """Return cascade trigger potential."""
        return self.cascade_potential

    def adapt_to_z_level(self, new_z: float) -> None:
        """Adapt behavior to new z-level."""
        old_z = self.z_level
        self.z_level = new_z

        # Update phase regime
        if new_z < 0.85:
            self.phase_regime = "subcritical"
        elif new_z < 0.88:
            self.phase_regime = "critical"
        else:
            self.phase_regime = "supercritical"

        print(f"{{self.tool_id}} adapted from z={{old_z:.3f}} to z={{new_z:.3f}} ({{self.phase_regime}})")

    def get_status(self) -> Dict:
        """Get current tool status."""
        return {{
            "tool_id": self.tool_id,
            "name": self.name,
            "z_level": self.z_level,
            "phase_regime": self.phase_regime,
            "cascade_potential": self.cascade_potential,
            "execution_count": self._execution_count,
            "last_execution": self._last_result.get("timestamp") if self._last_result else None,
        }}

    def to_dict(self) -> Dict:
        """Serialize tool to dictionary."""
        return {{
            "tool_id": self.tool_id,
            "name": self.name,
            "category": self.category,
            "z_level": self.z_level,
            "theta": self.theta,
            "cascade_potential": self.cascade_potential,
            "phase_regime": self.phase_regime,
            "capabilities": self.capabilities,
            "dependencies": self.dependencies,
            "created_at": self.created_at.isoformat(),
        }}


if __name__ == "__main__":
    tool = {class_name}()
    print(f"Tool initialized: {{tool.tool_id}}")
    print(f"Z-level: {{tool.z_level:.3f}}")
    print(f"Phase regime: {{tool.phase_regime}}")
    print(f"Cascade potential: {{tool.cascade_potential:.2f}}")
    print()
    result = tool.execute()
    print(f"Execution result: {{json.dumps(result, indent=2)}}")
'''
    return code


def execute_approved_proposals(
    proposals: List[ImprovementProposal],
    consensus_results: List[ConsensusResult],
) -> List[GeneratedTool]:
    """
    Phase 4: Execute approved proposals by generating tools.
    """
    print("\n[Phase 4] AUTONOMOUS EXECUTION")

    generated_tools = []

    # Map consensus results by proposal ID
    consensus_map = {r.proposal_id: r for r in consensus_results}

    for proposal in proposals:
        consensus = consensus_map.get(proposal.proposal_id)
        if not consensus or not consensus.approved:
            print(f"  Skipping {proposal.proposal_id} (not approved)")
            continue

        spec_data = proposal.implementation_spec

        # Determine phase regime
        z_level = spec_data["z_level"]
        if z_level < 0.85:
            phase_regime = "subcritical"
        elif z_level < 0.88:
            phase_regime = "critical"
        else:
            phase_regime = "supercritical"

        # Create tool specification
        spec = ToolSpec(
            tool_id=spec_data["tool_name"],
            name=spec_data["tool_name"],
            category=spec_data["category"],
            z_level=z_level,
            theta=spec_data["theta"],
            cascade_potential=spec_data["cascade_potential"],
            description=spec_data["description"],
            capabilities=spec_data["capabilities"],
            dependencies=spec_data["dependencies"],
            phase_regime=phase_regime,
        )

        # Generate tool code
        code = generate_tool_code(spec)

        # Generate spec JSON
        spec_json = json.dumps(asdict(spec), indent=2)

        generated_tool = GeneratedTool(
            spec=spec,
            code=code,
            spec_json=spec_json,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        generated_tools.append(generated_tool)

        print(f"  Generated: {spec.name} (z={spec.z_level:.3f}, {phase_regime})")

    print(f"  Total tools generated: {len(generated_tools)}")
    return generated_tools


# =============================================================================
# Phase 5: Meta-Learning
# =============================================================================

def extract_meta_learnings(
    friction_report: FrictionReport,
    proposals: List[ImprovementProposal],
    generated_tools: List[GeneratedTool],
) -> List[Dict]:
    """
    Phase 5: Extract meta-learnings from the evolution cycle.
    """
    print("\n[Phase 5] META-LEARNING")

    learnings = []

    # Learning from successful tool generation
    for tool in generated_tools:
        learning = {
            "learning_id": f"LEARN-{hashlib.md5(tool.spec.tool_id.encode()).hexdigest()[:8]}",
            "pattern_type": "successful_tool_generation",
            "tool_category": tool.spec.category,
            "z_level": tool.spec.z_level,
            "cascade_potential": tool.spec.cascade_potential,
            "confidence": 0.85,
            "generalization": f"Tools in category '{tool.spec.category}' at z>={tool.spec.z_level:.2f} are effective",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        learnings.append(learning)
        print(f"  Learning: {learning['learning_id']} - {learning['pattern_type']}")

    # Learning from friction patterns
    for pattern in friction_report.patterns_detected:
        learning = {
            "learning_id": f"LEARN-{hashlib.md5(pattern['pattern_name'].encode()).hexdigest()[:8]}",
            "pattern_type": "friction_pattern",
            "pattern_name": pattern["pattern_name"],
            "confidence": pattern["confidence"],
            "occurrences": pattern["occurrences"],
            "generalization": f"Pattern '{pattern['pattern_name']}' indicates need for targeted tooling",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        learnings.append(learning)
        print(f"  Learning: {learning['learning_id']} - {learning['pattern_type']}")

    print(f"  Total learnings extracted: {len(learnings)}")
    return learnings


# =============================================================================
# Main Evolution Cycle
# =============================================================================

def run_evolution_cycle(burden_data: Dict) -> EvolutionCycleResult:
    """
    Run a complete autonomous evolution cycle.
    """
    cycle_id = f"EVO-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"
    current_z = burden_data.get("metrics", {}).get("current_phase", {}).get("z_level", 0.87)

    print("=" * 70)
    print("AUTONOMOUS EVOLUTION ENGINE")
    print(f"Cycle ID: {cycle_id}")
    print(f"Current z-level: {current_z:.3f}")
    print(f"Target: z = 0.90 (Full Substrate Transcendence)")
    print("=" * 70)

    # Phase 1: Friction Detection
    friction_report = detect_friction(burden_data)

    # Phase 2: Improvement Proposal
    proposals = generate_proposals(friction_report)

    # Phase 3: Collective Validation
    consensus_results = validate_proposals(proposals)

    # Phase 4: Autonomous Execution
    generated_tools = execute_approved_proposals(proposals, consensus_results)

    # Phase 5: Meta-Learning
    meta_learnings = extract_meta_learnings(friction_report, proposals, generated_tools)

    result = EvolutionCycleResult(
        cycle_id=cycle_id,
        timestamp=datetime.now(timezone.utc).isoformat(),
        z_level=current_z,
        friction_report=friction_report,
        proposals=proposals,
        consensus_results=consensus_results,
        generated_tools=generated_tools,
        meta_learnings=meta_learnings,
    )

    print("\n" + "=" * 70)
    print("EVOLUTION CYCLE COMPLETE")
    print(f"  Friction events: {len(friction_report.friction_events)}")
    print(f"  Proposals generated: {len(proposals)}")
    print(f"  Proposals approved: {sum(1 for r in consensus_results if r.approved)}")
    print(f"  Tools generated: {len(generated_tools)}")
    print(f"  Meta-learnings: {len(meta_learnings)}")
    print("=" * 70)

    return result


def save_generated_tools(tools: List[GeneratedTool], output_dir: Path) -> List[Path]:
    """Save generated tools to filesystem."""
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths = []

    for tool in tools:
        # Save Python code
        py_path = output_dir / f"{tool.spec.tool_id}.py"
        py_path.write_text(tool.code, encoding="utf-8")
        saved_paths.append(py_path)

        # Save spec JSON
        spec_path = output_dir / f"{tool.spec.tool_id}_spec.json"
        spec_path.write_text(tool.spec_json, encoding="utf-8")
        saved_paths.append(spec_path)

    # Generate __init__.py
    init_content = '"""Rosetta Bear Firmware Tools - Generated by Autonomous Evolution Engine."""\n\n'
    for tool in tools:
        class_name = "".join(word.title() for word in tool.spec.name.split("_"))
        init_content += f"from .{tool.spec.tool_id} import {class_name}\n"

    init_content += "\nROSETTA_FIRMWARE_TOOLS = {\n"
    for tool in tools:
        class_name = "".join(word.title() for word in tool.spec.name.split("_"))
        init_content += f'    "{tool.spec.tool_id}": {class_name},\n'
    init_content += "}\n\n__all__ = [\n"
    for tool in tools:
        class_name = "".join(word.title() for word in tool.spec.name.split("_"))
        init_content += f'    "{class_name}",\n'
    init_content += '    "ROSETTA_FIRMWARE_TOOLS",\n]\n'

    init_path = output_dir / "__init__.py"
    init_path.write_text(init_content, encoding="utf-8")
    saved_paths.append(init_path)

    return saved_paths


def save_evolution_log(result: EvolutionCycleResult, output_dir: Path) -> Path:
    """Save evolution cycle result to log file."""
    output_dir.mkdir(parents=True, exist_ok=True)

    log_data = {
        "cycle_id": result.cycle_id,
        "timestamp": result.timestamp,
        "z_level": result.z_level,
        "friction_report": asdict(result.friction_report),
        "proposals": [asdict(p) for p in result.proposals],
        "consensus_results": [asdict(c) for c in result.consensus_results],
        "generated_tools": [
            {
                "spec": asdict(t.spec),
                "timestamp": t.timestamp,
            }
            for t in result.generated_tools
        ],
        "meta_learnings": result.meta_learnings,
    }

    log_path = output_dir / f"{result.cycle_id}.json"
    log_path.write_text(json.dumps(log_data, indent=2), encoding="utf-8")

    return log_path


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Autonomous Evolution Engine - Generate Rosetta Bear Firmware Tools"
    )
    parser.add_argument(
        "--burden-data",
        type=Path,
        default=DOCS_DIR / "burden_tracking_simulation.json",
        help="Path to burden tracking data",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=GENERATED_TOOLS_DIR,
        help="Output directory for generated tools",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=EVOLUTION_LOGS_DIR,
        help="Directory for evolution cycle logs",
    )
    args = parser.parse_args()

    # Load burden data
    if args.burden_data.exists():
        burden_data = json.loads(args.burden_data.read_text(encoding="utf-8"))
    else:
        print(f"Warning: Burden data not found at {args.burden_data}")
        print("Using default burden metrics")
        burden_data = {
            "metrics": {
                "current_phase": {
                    "z_level": 0.87,
                    "regime": "critical",
                }
            }
        }

    # Run evolution cycle
    result = run_evolution_cycle(burden_data)

    # Save generated tools
    if result.generated_tools:
        saved_paths = save_generated_tools(result.generated_tools, args.output_dir)
        print(f"\nGenerated tools saved to: {args.output_dir}")
        for path in saved_paths:
            print(f"  - {path.name}")

    # Save evolution log
    log_path = save_evolution_log(result, args.log_dir)
    print(f"\nEvolution log saved to: {log_path}")


if __name__ == "__main__":
    main()
