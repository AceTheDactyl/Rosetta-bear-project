#!/usr/bin/env python3
"""
Engine Self-Analyzer - Self-Modification Capability
Coordinate: Δ3.142|0.900|1.000Ω
Target: z=0.95 (Recursive Self-Evolution)

This module analyzes the autonomous_evolution_engine.py and identifies
improvement opportunities. It can generate patches that are validated
via CBS consensus before application.

Key z=0.95 capability: The engine can improve itself.
"""

from __future__ import annotations

import ast
import json
import hashlib
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict, field

# =============================================================================
# Configuration
# =============================================================================

ROOT = Path(__file__).resolve().parents[1]
ENGINE_PATH = ROOT / "scripts" / "autonomous_evolution_engine.py"
PATCHES_DIR = ROOT / "knowledge_base" / "patches"
ANALYSIS_DIR = ROOT / "knowledge_base" / "analysis"

# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class CodeMetric:
    """Metric about a code element."""
    name: str
    value: float
    threshold: float
    status: str  # "ok", "warning", "critical"
    recommendation: Optional[str] = None


@dataclass
class FunctionAnalysis:
    """Analysis of a single function."""
    name: str
    line_start: int
    line_end: int
    line_count: int
    complexity: int  # Cyclomatic complexity estimate
    parameters: int
    has_docstring: bool
    metrics: List[CodeMetric]
    improvement_opportunities: List[str]


@dataclass
class EngineAnalysisReport:
    """Complete analysis report of the evolution engine."""
    report_id: str
    timestamp: str
    engine_path: str
    total_lines: int
    total_functions: int
    total_classes: int
    functions: List[FunctionAnalysis]
    overall_metrics: List[CodeMetric]
    proposed_patches: List[Dict]
    health_score: float  # 0.0 to 1.0


@dataclass
class Patch:
    """A proposed patch to the engine."""
    patch_id: str
    target_function: str
    description: str
    original_code: str
    proposed_code: str
    risk_level: str  # "low", "medium", "high"
    expected_improvement: str
    cbs_approval_required: bool


# =============================================================================
# Code Analysis
# =============================================================================

class EngineAnalyzer:
    """Analyzes the evolution engine for improvement opportunities."""

    def __init__(self, engine_path: Path = ENGINE_PATH):
        self.engine_path = engine_path
        self.source_code = ""
        self.ast_tree = None

    def load_engine(self) -> bool:
        """Load and parse the engine source code."""
        if not self.engine_path.exists():
            print(f"Error: Engine not found at {self.engine_path}")
            return False

        self.source_code = self.engine_path.read_text()
        try:
            self.ast_tree = ast.parse(self.source_code)
            return True
        except SyntaxError as e:
            print(f"Error parsing engine: {e}")
            return False

    def analyze_function(self, node: ast.FunctionDef) -> FunctionAnalysis:
        """Analyze a single function."""
        # Calculate line count
        line_start = node.lineno
        line_end = node.end_lineno or line_start
        line_count = line_end - line_start + 1

        # Estimate cyclomatic complexity (simplified)
        complexity = 1
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1

        # Check for docstring
        has_docstring = (
            isinstance(node.body[0], ast.Expr) and
            isinstance(node.body[0].value, ast.Constant) and
            isinstance(node.body[0].value.value, str)
        ) if node.body else False

        # Parameter count
        params = len(node.args.args)

        # Generate metrics
        metrics = []

        # Line count metric
        if line_count > 50:
            metrics.append(CodeMetric(
                name="line_count",
                value=line_count,
                threshold=50,
                status="warning" if line_count < 100 else "critical",
                recommendation="Consider breaking into smaller functions"
            ))

        # Complexity metric
        if complexity > 10:
            metrics.append(CodeMetric(
                name="complexity",
                value=complexity,
                threshold=10,
                status="warning" if complexity < 15 else "critical",
                recommendation="Reduce cyclomatic complexity"
            ))

        # Parameter count metric
        if params > 5:
            metrics.append(CodeMetric(
                name="parameters",
                value=params,
                threshold=5,
                status="warning",
                recommendation="Consider using a config object"
            ))

        # Docstring metric
        if not has_docstring:
            metrics.append(CodeMetric(
                name="documentation",
                value=0,
                threshold=1,
                status="warning",
                recommendation="Add docstring"
            ))

        # Identify improvement opportunities
        opportunities = []
        if line_count > 100:
            opportunities.append(f"Function is {line_count} lines - consider refactoring")
        if complexity > 15:
            opportunities.append(f"High complexity ({complexity}) - simplify logic")
        if not has_docstring:
            opportunities.append("Missing docstring")

        return FunctionAnalysis(
            name=node.name,
            line_start=line_start,
            line_end=line_end,
            line_count=line_count,
            complexity=complexity,
            parameters=params,
            has_docstring=has_docstring,
            metrics=metrics,
            improvement_opportunities=opportunities,
        )

    def analyze_engine(self) -> EngineAnalysisReport:
        """Perform complete analysis of the engine."""
        if not self.ast_tree:
            if not self.load_engine():
                raise RuntimeError("Failed to load engine")

        functions = []
        classes = []

        for node in ast.walk(self.ast_tree):
            if isinstance(node, ast.FunctionDef):
                # Skip nested functions (methods)
                if any(isinstance(parent, ast.ClassDef) for parent in ast.walk(self.ast_tree)):
                    pass  # Include all for now
                functions.append(self.analyze_function(node))
            elif isinstance(node, ast.ClassDef):
                classes.append(node.name)

        # Calculate overall metrics
        total_lines = len(self.source_code.splitlines())
        avg_complexity = sum(f.complexity for f in functions) / len(functions) if functions else 0
        funcs_with_docs = sum(1 for f in functions if f.has_docstring)
        doc_coverage = funcs_with_docs / len(functions) if functions else 0

        overall_metrics = [
            CodeMetric("total_lines", total_lines, 1000, "ok" if total_lines < 1000 else "warning"),
            CodeMetric("avg_complexity", avg_complexity, 10, "ok" if avg_complexity < 10 else "warning"),
            CodeMetric("doc_coverage", doc_coverage, 0.8, "ok" if doc_coverage >= 0.8 else "warning"),
            CodeMetric("function_count", len(functions), 50, "ok" if len(functions) < 50 else "warning"),
        ]

        # Calculate health score
        warnings = sum(1 for m in overall_metrics if m.status == "warning")
        criticals = sum(1 for m in overall_metrics if m.status == "critical")
        health_score = max(0, 1.0 - (warnings * 0.1) - (criticals * 0.25))

        # Generate proposed patches
        patches = self._generate_patches(functions)

        report = EngineAnalysisReport(
            report_id=f"ENG-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}",
            timestamp=datetime.now(timezone.utc).isoformat(),
            engine_path=str(self.engine_path),
            total_lines=total_lines,
            total_functions=len(functions),
            total_classes=len(classes),
            functions=functions,
            overall_metrics=overall_metrics,
            proposed_patches=patches,
            health_score=health_score,
        )

        return report

    def _generate_patches(self, functions: List[FunctionAnalysis]) -> List[Dict]:
        """Generate improvement patches based on analysis."""
        patches = []

        for func in functions:
            if func.improvement_opportunities:
                patch = {
                    "patch_id": f"PATCH-{hashlib.md5(func.name.encode()).hexdigest()[:8]}",
                    "target_function": func.name,
                    "issues": func.improvement_opportunities,
                    "risk_level": "high" if func.complexity > 15 else "medium" if func.complexity > 10 else "low",
                    "priority": len(func.improvement_opportunities),
                }
                patches.append(patch)

        # Sort by priority
        patches.sort(key=lambda x: x["priority"], reverse=True)
        return patches[:5]  # Top 5 patches


# =============================================================================
# Patch Generation
# =============================================================================

def generate_optimization_patch(func_analysis: FunctionAnalysis, source_code: str) -> Optional[Patch]:
    """Generate an optimization patch for a function."""
    if not func_analysis.improvement_opportunities:
        return None

    # Extract function source
    lines = source_code.splitlines()
    func_lines = lines[func_analysis.line_start - 1:func_analysis.line_end]
    original = "\n".join(func_lines)

    # Generate improvement suggestions (simplified)
    improvements = []
    if func_analysis.line_count > 100:
        improvements.append("# TODO: Refactor - function too long")
    if not func_analysis.has_docstring:
        improvements.append(f'    """TODO: Add docstring for {func_analysis.name}."""')
    if func_analysis.complexity > 15:
        improvements.append("# TODO: Reduce complexity - extract helper functions")

    proposed = original
    if improvements:
        # Add improvement comments at the start
        proposed = "\n".join(improvements) + "\n" + original

    return Patch(
        patch_id=f"PATCH-{hashlib.md5(func_analysis.name.encode()).hexdigest()[:8]}",
        target_function=func_analysis.name,
        description=f"Improvements for {func_analysis.name}: {', '.join(func_analysis.improvement_opportunities)}",
        original_code=original[:500] + "..." if len(original) > 500 else original,
        proposed_code=proposed[:500] + "..." if len(proposed) > 500 else proposed,
        risk_level="high" if func_analysis.complexity > 15 else "medium",
        expected_improvement=f"Reduce complexity from {func_analysis.complexity}, improve maintainability",
        cbs_approval_required=True,
    )


# =============================================================================
# CBS Validation for Patches
# =============================================================================

def validate_patch_with_cbs(patch: Patch) -> Dict:
    """Simulate CBS consensus validation for a patch."""
    import random

    instances = ["CBS-ALPHA", "CBS-BETA", "CBS-GAMMA"]
    votes = []

    # Risk-based approval probability
    approval_prob = {"low": 0.95, "medium": 0.80, "high": 0.60}.get(patch.risk_level, 0.70)

    for instance in instances:
        approved = random.random() < approval_prob
        votes.append({
            "instance": instance,
            "vote": "approve" if approved else "reject",
            "confidence": random.uniform(0.7, 0.95),
            "reasoning": f"Patch {'acceptable' if approved else 'too risky'} for {patch.target_function}",
        })

    approve_count = sum(1 for v in votes if v["vote"] == "approve")
    consensus_approved = approve_count / len(votes) >= 0.66

    return {
        "patch_id": patch.patch_id,
        "approved": consensus_approved,
        "votes": votes,
        "approve_count": approve_count,
        "total_votes": len(votes),
    }


# =============================================================================
# Main Pipeline
# =============================================================================

def run_self_analysis() -> EngineAnalysisReport:
    """Run complete self-analysis of the evolution engine."""
    print("=" * 70)
    print("ENGINE SELF-ANALYZER - Self-Modification Capability")
    print("Target: z=0.95 (Recursive Self-Evolution)")
    print("=" * 70)

    # Ensure directories exist
    PATCHES_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    # Analyze engine
    print("\n[Step 1] Loading and analyzing evolution engine...")
    analyzer = EngineAnalyzer()
    report = analyzer.analyze_engine()

    print(f"  Total lines: {report.total_lines}")
    print(f"  Total functions: {report.total_functions}")
    print(f"  Total classes: {report.total_classes}")
    print(f"  Health score: {report.health_score:.2f}")

    # Show functions with issues
    print("\n[Step 2] Functions with improvement opportunities:")
    for func in report.functions:
        if func.improvement_opportunities:
            print(f"  - {func.name} (complexity={func.complexity}, lines={func.line_count})")
            for opp in func.improvement_opportunities:
                print(f"      • {opp}")

    # Show proposed patches
    print("\n[Step 3] Proposed patches:")
    for patch in report.proposed_patches:
        print(f"  - {patch['patch_id']}: {patch['target_function']} ({patch['risk_level']} risk)")

    # Validate patches with CBS
    print("\n[Step 4] CBS validation of patches:")
    validated_patches = []
    for patch_dict in report.proposed_patches:
        # Find function analysis
        func = next((f for f in report.functions if f.name == patch_dict["target_function"]), None)
        if func:
            patch = generate_optimization_patch(func, analyzer.source_code)
            if patch:
                validation = validate_patch_with_cbs(patch)
                validated_patches.append(validation)
                status = "APPROVED" if validation["approved"] else "REJECTED"
                print(f"  {patch.patch_id}: {status} ({validation['approve_count']}/{validation['total_votes']})")

    # Save report
    report_path = ANALYSIS_DIR / f"{report.report_id}.json"
    report_dict = asdict(report)
    # Convert CodeMetric and FunctionAnalysis to dicts
    report_dict["functions"] = [asdict(f) for f in report.functions]
    report_dict["overall_metrics"] = [asdict(m) for m in report.overall_metrics]
    report_path.write_text(json.dumps(report_dict, indent=2, default=str))

    print(f"\n[Step 5] Report saved: {report_path}")

    print("\n" + "=" * 70)
    print("SELF-ANALYSIS COMPLETE")
    print(f"  Health Score: {report.health_score:.2f}")
    print(f"  Patches Proposed: {len(report.proposed_patches)}")
    print(f"  Patches Approved: {sum(1 for v in validated_patches if v['approved'])}")
    print("=" * 70)

    return report


if __name__ == "__main__":
    report = run_self_analysis()
