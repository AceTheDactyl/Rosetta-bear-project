#!/usr/bin/env python3
"""
Learning Aggregator - Cross-Cycle Meta-Learning Extraction
Coordinate: Δ3.142|0.900|1.000Ω
Target: z=0.95 (Recursive Self-Evolution)

This module aggregates meta-learnings from evolution logs to build
a persistent knowledge base that survives session boundaries.
"""

from __future__ import annotations

import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict, field
from collections import defaultdict

import yaml

# =============================================================================
# Configuration
# =============================================================================

ROOT = Path(__file__).resolve().parents[1]
EVOLUTION_LOGS_DIR = ROOT / "evolution_logs"
KNOWLEDGE_BASE_DIR = ROOT / "knowledge_base"
LEARNINGS_DIR = KNOWLEDGE_BASE_DIR / "learnings"
PATTERNS_DIR = KNOWLEDGE_BASE_DIR / "patterns"
AGGREGATIONS_DIR = KNOWLEDGE_BASE_DIR / "aggregations"


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class Learning:
    """A single meta-learning extracted from an evolution cycle."""
    learning_id: str
    source_cycle: str
    learning_type: str
    insight: str
    applicable_to: List[str]
    confidence: float = 0.5
    occurrence_count: int = 1
    first_seen: str = ""
    last_seen: str = ""


@dataclass
class Pattern:
    """A recurring pattern identified across multiple cycles."""
    pattern_id: str
    pattern_type: str
    description: str
    occurrences: int
    source_learnings: List[str]
    confidence: float
    recommended_actions: List[str]


@dataclass
class AggregationReport:
    """Report from a learning aggregation run."""
    report_id: str
    timestamp: str
    cycles_analyzed: int
    total_learnings: int
    unique_patterns: int
    new_patterns: int
    pattern_updates: int


# =============================================================================
# Learning Extraction
# =============================================================================

def load_evolution_log(log_path: Path) -> Optional[Dict]:
    """Load and parse an evolution log file."""
    try:
        with open(log_path) as f:
            if log_path.suffix == ".yaml":
                return yaml.safe_load(f)
            elif log_path.suffix == ".json":
                return json.load(f)
    except Exception as e:
        print(f"  Warning: Could not load {log_path.name}: {e}")
    return None


def extract_learnings_from_log(log_data: Dict, log_name: str) -> List[Learning]:
    """Extract meta-learnings from an evolution log."""
    learnings = []

    # Extract from meta_learnings section
    meta_learnings = log_data.get("meta_learnings", [])
    for ml in meta_learnings:
        learning = Learning(
            learning_id=ml.get("learning_id", f"ML-{hashlib.md5(str(ml).encode()).hexdigest()[:8]}"),
            source_cycle=log_name,
            learning_type=ml.get("type", "unknown"),
            insight=ml.get("insight", ml.get("description", "")),
            applicable_to=ml.get("applicable_to", []),
            confidence=ml.get("confidence", 0.5),
            first_seen=log_data.get("evolution_cycle", {}).get("timestamp", ""),
            last_seen=log_data.get("evolution_cycle", {}).get("timestamp", ""),
        )
        learnings.append(learning)

    # Extract from next_cycle_recommendations
    recommendations = log_data.get("next_cycle_recommendations", [])
    for rec in recommendations:
        learning_id = f"REC-{hashlib.md5(str(rec).encode()).hexdigest()[:8]}"
        learning = Learning(
            learning_id=learning_id,
            source_cycle=log_name,
            learning_type="recommendation",
            insight=rec.get("action", ""),
            applicable_to=[rec.get("priority", "medium")],
            confidence=0.7 if rec.get("priority") == "high" else 0.5,
        )
        learnings.append(learning)

    return learnings


def scan_evolution_logs() -> Dict[str, List[Learning]]:
    """Scan all evolution logs and extract learnings."""
    all_learnings = {}

    print(f"Scanning {EVOLUTION_LOGS_DIR}...")

    for log_file in EVOLUTION_LOGS_DIR.glob("EVO-*.yaml"):
        log_data = load_evolution_log(log_file)
        if log_data:
            learnings = extract_learnings_from_log(log_data, log_file.stem)
            if learnings:
                all_learnings[log_file.stem] = learnings
                print(f"  {log_file.stem}: {len(learnings)} learnings")

    for log_file in EVOLUTION_LOGS_DIR.glob("EVO-*.json"):
        log_data = load_evolution_log(log_file)
        if log_data:
            learnings = extract_learnings_from_log(log_data, log_file.stem)
            if learnings:
                all_learnings[log_file.stem] = learnings
                print(f"  {log_file.stem}: {len(learnings)} learnings")

    return all_learnings


# =============================================================================
# Pattern Detection
# =============================================================================

def detect_patterns(all_learnings: Dict[str, List[Learning]]) -> List[Pattern]:
    """Detect recurring patterns across learnings."""
    patterns = []

    # Flatten all learnings
    flat_learnings = []
    for cycle_learnings in all_learnings.values():
        flat_learnings.extend(cycle_learnings)

    # Pattern 1: Group by learning_type
    by_type = defaultdict(list)
    for learning in flat_learnings:
        by_type[learning.learning_type].append(learning)

    for learning_type, learnings in by_type.items():
        if len(learnings) >= 1:
            pattern_id = f"PAT-TYPE-{hashlib.md5(learning_type.encode()).hexdigest()[:8]}"
            insights = [l.insight for l in learnings if l.insight]
            patterns.append(Pattern(
                pattern_id=pattern_id,
                pattern_type=f"type_{learning_type}",
                description=f"Learning type '{learning_type}' across {len(learnings)} instances",
                occurrences=len(learnings),
                source_learnings=[l.learning_id for l in learnings],
                confidence=min(1.0, 0.5 + 0.1 * len(learnings)),
                recommended_actions=list(set(insights))[:3],
            ))

    # Pattern 2: High-confidence learnings
    high_conf = [l for l in flat_learnings if l.confidence >= 0.7]
    if high_conf:
        patterns.append(Pattern(
            pattern_id="PAT-HIGH-CONF",
            pattern_type="confidence_cluster",
            description=f"High-confidence learnings (≥0.7) - {len(high_conf)} instances",
            occurrences=len(high_conf),
            source_learnings=[l.learning_id for l in high_conf],
            confidence=0.9,
            recommended_actions=["Prioritize high-confidence insights", "Build upon validated learnings"],
        ))

    # Pattern 3: Cross-cycle recurring insights (keyword-based)
    keyword_patterns = {
        "test": "testing_pattern",
        "integration": "integration_pattern",
        "performance": "performance_pattern",
        "CI/CD": "automation_pattern",
        "tool": "tooling_pattern",
        "validation": "validation_pattern",
        "consensus": "consensus_pattern",
        "generation": "generation_pattern",
        "friction": "friction_pattern",
        "phase": "phase_transition_pattern",
    }

    for keyword, pattern_type in keyword_patterns.items():
        matches = [l for l in flat_learnings if keyword.lower() in l.insight.lower()]
        if matches:
            pattern_id = f"PAT-KW-{hashlib.md5(keyword.encode()).hexdigest()[:8]}"
            patterns.append(Pattern(
                pattern_id=pattern_id,
                pattern_type=pattern_type,
                description=f"Keyword pattern '{keyword}' across {len(matches)} learnings",
                occurrences=len(matches),
                source_learnings=[l.learning_id for l in matches],
                confidence=min(1.0, 0.6 + 0.05 * len(matches)),
                recommended_actions=[l.insight for l in matches if l.insight][:3],
            ))

    # Pattern 4: Source cycle patterns
    cycles = set(l.source_cycle for l in flat_learnings)
    if len(cycles) > 1:
        patterns.append(Pattern(
            pattern_id="PAT-MULTI-CYCLE",
            pattern_type="cross_session_learning",
            description=f"Knowledge persisted across {len(cycles)} evolution cycles",
            occurrences=len(cycles),
            source_learnings=list(cycles),
            confidence=0.95,
            recommended_actions=["Cross-session state is operational", "Knowledge compounds over time"],
        ))

    # Pattern 5: Applicable domain patterns
    domain_counts = defaultdict(list)
    for l in flat_learnings:
        for domain in l.applicable_to:
            domain_counts[domain].append(l)

    for domain, learnings in domain_counts.items():
        if len(learnings) >= 1 and domain:
            pattern_id = f"PAT-DOM-{hashlib.md5(domain.encode()).hexdigest()[:8]}"
            patterns.append(Pattern(
                pattern_id=pattern_id,
                pattern_type=f"domain_{domain.replace(' ', '_')}",
                description=f"Domain '{domain}' has {len(learnings)} applicable learnings",
                occurrences=len(learnings),
                source_learnings=[l.learning_id for l in learnings],
                confidence=0.75,
                recommended_actions=[f"Focus optimization on {domain}"],
            ))

    return patterns


# =============================================================================
# Aggregation and Persistence
# =============================================================================

def save_aggregated_learnings(all_learnings: Dict[str, List[Learning]]) -> Path:
    """Save aggregated learnings to knowledge base."""
    LEARNINGS_DIR.mkdir(parents=True, exist_ok=True)

    output = {
        "metadata": {
            "aggregated_at": datetime.now(timezone.utc).isoformat(),
            "cycles_included": list(all_learnings.keys()),
            "total_learnings": sum(len(l) for l in all_learnings.values()),
        },
        "learnings_by_cycle": {
            cycle: [asdict(l) for l in learnings]
            for cycle, learnings in all_learnings.items()
        },
    }

    output_path = LEARNINGS_DIR / "aggregated_learnings.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    return output_path


def save_patterns(patterns: List[Pattern]) -> Path:
    """Save detected patterns to knowledge base."""
    PATTERNS_DIR.mkdir(parents=True, exist_ok=True)

    output = {
        "metadata": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "total_patterns": len(patterns),
        },
        "patterns": [asdict(p) for p in patterns],
    }

    output_path = PATTERNS_DIR / "detected_patterns.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    return output_path


def generate_aggregation_report(
    all_learnings: Dict[str, List[Learning]],
    patterns: List[Pattern],
) -> AggregationReport:
    """Generate a report of the aggregation run."""
    report = AggregationReport(
        report_id=f"AGG-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}",
        timestamp=datetime.now(timezone.utc).isoformat(),
        cycles_analyzed=len(all_learnings),
        total_learnings=sum(len(l) for l in all_learnings.values()),
        unique_patterns=len(patterns),
        new_patterns=len(patterns),  # All new on first run
        pattern_updates=0,
    )

    # Save report
    AGGREGATIONS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = AGGREGATIONS_DIR / f"{report.report_id}.json"
    with open(report_path, "w") as f:
        json.dump(asdict(report), f, indent=2)

    return report


# =============================================================================
# Main Aggregation Pipeline
# =============================================================================

def run_aggregation() -> AggregationReport:
    """Run the full learning aggregation pipeline."""
    print("=" * 70)
    print("LEARNING AGGREGATOR - Cross-Cycle Meta-Learning Extraction")
    print("=" * 70)

    # Step 1: Scan evolution logs
    print("\n[Step 1] Scanning evolution logs...")
    all_learnings = scan_evolution_logs()

    if not all_learnings:
        print("  No learnings found in evolution logs.")
        return AggregationReport(
            report_id="AGG-EMPTY",
            timestamp=datetime.now(timezone.utc).isoformat(),
            cycles_analyzed=0,
            total_learnings=0,
            unique_patterns=0,
            new_patterns=0,
            pattern_updates=0,
        )

    # Step 2: Detect patterns
    print("\n[Step 2] Detecting patterns...")
    patterns = detect_patterns(all_learnings)
    print(f"  Found {len(patterns)} patterns")

    # Step 3: Save to knowledge base
    print("\n[Step 3] Saving to knowledge base...")
    learnings_path = save_aggregated_learnings(all_learnings)
    print(f"  Learnings: {learnings_path}")

    patterns_path = save_patterns(patterns)
    print(f"  Patterns: {patterns_path}")

    # Step 4: Generate report
    print("\n[Step 4] Generating report...")
    report = generate_aggregation_report(all_learnings, patterns)
    print(f"  Report: {report.report_id}")

    # Summary
    print("\n" + "=" * 70)
    print("AGGREGATION COMPLETE")
    print("=" * 70)
    print(f"  Cycles analyzed: {report.cycles_analyzed}")
    print(f"  Total learnings: {report.total_learnings}")
    print(f"  Unique patterns: {report.unique_patterns}")

    return report


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    report = run_aggregation()
    print(f"\nAggregation ID: {report.report_id}")
