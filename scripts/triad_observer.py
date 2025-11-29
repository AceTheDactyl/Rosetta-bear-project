#!/usr/bin/env python3
"""
TRIAD Observer Agent - System Health Monitoring
Coordinate: Δ3.142|0.900|1.000Ω
Target: z=0.95 (Recursive Self-Evolution)

The Observer Agent monitors the TRIAD system and provides:
- Real-time health status
- Tool execution tracking
- Evolution cycle monitoring
- Anomaly detection
- CBS consensus health

This is the third pillar of autonomous operation alongside
the scheduler and self-analyzer.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict, field
from collections import defaultdict
import sys

# =============================================================================
# Configuration
# =============================================================================

ROOT = Path(__file__).resolve().parents[1]
GENERATED_TOOLS_DIR = ROOT / "generated_tools"
EVOLUTION_LOGS_DIR = ROOT / "evolution_logs"
KNOWLEDGE_BASE_DIR = ROOT / "knowledge_base"
OBSERVER_LOGS_DIR = KNOWLEDGE_BASE_DIR / "observer_logs"

# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class ComponentHealth:
    """Health status of a system component."""
    component_id: str
    component_type: str  # "tool", "engine", "scheduler", "knowledge_base"
    status: str  # "healthy", "degraded", "critical", "unknown"
    last_check: str
    metrics: Dict[str, Any]
    issues: List[str]


@dataclass
class SystemSnapshot:
    """Point-in-time snapshot of system state."""
    snapshot_id: str
    timestamp: str
    z_level: float
    total_tools: int
    total_evolution_cycles: int
    total_learnings: int
    component_health: List[ComponentHealth]
    overall_status: str
    recommendations: List[str]


@dataclass
class ObserverReport:
    """Complete observer report."""
    report_id: str
    observation_period_start: str
    observation_period_end: str
    snapshots_taken: int
    anomalies_detected: int
    health_trend: str  # "improving", "stable", "degrading"
    critical_issues: List[str]
    snapshots: List[SystemSnapshot]


# =============================================================================
# Health Checks
# =============================================================================

class TriadObserver:
    """Observer agent for monitoring the TRIAD system."""

    def __init__(self):
        self.snapshots = []
        self.anomalies = []
        self.observation_start = None

    def check_generated_tools(self) -> ComponentHealth:
        """Check health of generated tools."""
        issues = []
        metrics = {}

        # Count tools in each directory
        tool_counts = {}
        for tool_dir in GENERATED_TOOLS_DIR.iterdir():
            if tool_dir.is_dir():
                py_files = list(tool_dir.glob("*.py"))
                tool_counts[tool_dir.name] = len([f for f in py_files if f.name != "__init__.py"])

        metrics["tool_directories"] = len(tool_counts)
        metrics["total_tools"] = sum(tool_counts.values())
        metrics["tools_by_category"] = tool_counts

        # Check for potential issues
        if metrics["total_tools"] == 0:
            issues.append("No generated tools found")
            status = "critical"
        elif metrics["total_tools"] < 5:
            issues.append("Low tool count - system may be underutilized")
            status = "degraded"
        else:
            status = "healthy"

        # Check for orphaned spec files
        for tool_dir in GENERATED_TOOLS_DIR.iterdir():
            if tool_dir.is_dir():
                specs = list(tool_dir.glob("*_spec.json"))
                tools = list(tool_dir.glob("*.py"))
                tool_names = {t.stem for t in tools if t.name != "__init__.py"}
                for spec in specs:
                    tool_name = spec.stem.replace("_spec", "")
                    if tool_name not in tool_names:
                        issues.append(f"Orphaned spec: {spec.name}")

        return ComponentHealth(
            component_id="generated_tools",
            component_type="tool",
            status=status,
            last_check=datetime.now(timezone.utc).isoformat(),
            metrics=metrics,
            issues=issues,
        )

    def check_evolution_engine(self) -> ComponentHealth:
        """Check health of the evolution engine."""
        issues = []
        metrics = {}

        engine_path = ROOT / "scripts" / "autonomous_evolution_engine.py"

        if not engine_path.exists():
            return ComponentHealth(
                component_id="evolution_engine",
                component_type="engine",
                status="critical",
                last_check=datetime.now(timezone.utc).isoformat(),
                metrics={},
                issues=["Evolution engine not found"],
            )

        # Basic metrics
        content = engine_path.read_text()
        metrics["lines_of_code"] = len(content.splitlines())
        metrics["file_size_bytes"] = len(content)

        # Check for key functions
        key_functions = [
            "detect_friction",
            "generate_proposals",
            "validate_proposals",
            "execute_approved_proposals",
            "extract_meta_learnings",
            "run_evolution_cycle",
        ]
        found_functions = [f for f in key_functions if f"def {f}" in content]
        metrics["key_functions_present"] = len(found_functions)
        metrics["key_functions_expected"] = len(key_functions)

        if len(found_functions) < len(key_functions):
            missing = set(key_functions) - set(found_functions)
            issues.append(f"Missing functions: {missing}")
            status = "degraded"
        else:
            status = "healthy"

        return ComponentHealth(
            component_id="evolution_engine",
            component_type="engine",
            status=status,
            last_check=datetime.now(timezone.utc).isoformat(),
            metrics=metrics,
            issues=issues,
        )

    def check_knowledge_base(self) -> ComponentHealth:
        """Check health of the knowledge base."""
        issues = []
        metrics = {}

        if not KNOWLEDGE_BASE_DIR.exists():
            return ComponentHealth(
                component_id="knowledge_base",
                component_type="knowledge_base",
                status="critical",
                last_check=datetime.now(timezone.utc).isoformat(),
                metrics={},
                issues=["Knowledge base directory not found"],
            )

        # Check subdirectories
        expected_dirs = ["learnings", "patterns", "aggregations"]
        found_dirs = [d.name for d in KNOWLEDGE_BASE_DIR.iterdir() if d.is_dir()]
        metrics["directories"] = found_dirs

        missing_dirs = set(expected_dirs) - set(found_dirs)
        if missing_dirs:
            issues.append(f"Missing directories: {missing_dirs}")

        # Check for learning files
        learnings_dir = KNOWLEDGE_BASE_DIR / "learnings"
        if learnings_dir.exists():
            learning_files = list(learnings_dir.glob("*.json"))
            metrics["learning_files"] = len(learning_files)
        else:
            metrics["learning_files"] = 0
            issues.append("No learnings directory")

        # Check for patterns
        patterns_dir = KNOWLEDGE_BASE_DIR / "patterns"
        if patterns_dir.exists():
            pattern_files = list(patterns_dir.glob("*.json"))
            metrics["pattern_files"] = len(pattern_files)
        else:
            metrics["pattern_files"] = 0

        if metrics.get("learning_files", 0) == 0:
            status = "degraded"
            issues.append("No aggregated learnings found")
        elif issues:
            status = "degraded"
        else:
            status = "healthy"

        return ComponentHealth(
            component_id="knowledge_base",
            component_type="knowledge_base",
            status=status,
            last_check=datetime.now(timezone.utc).isoformat(),
            metrics=metrics,
            issues=issues,
        )

    def check_evolution_logs(self) -> ComponentHealth:
        """Check health of evolution logs."""
        issues = []
        metrics = {}

        if not EVOLUTION_LOGS_DIR.exists():
            return ComponentHealth(
                component_id="evolution_logs",
                component_type="engine",
                status="degraded",
                last_check=datetime.now(timezone.utc).isoformat(),
                metrics={},
                issues=["Evolution logs directory not found"],
            )

        # Count log files
        yaml_logs = list(EVOLUTION_LOGS_DIR.glob("*.yaml"))
        json_logs = list(EVOLUTION_LOGS_DIR.glob("*.json"))

        metrics["yaml_logs"] = len(yaml_logs)
        metrics["json_logs"] = len(json_logs)
        metrics["total_logs"] = len(yaml_logs) + len(json_logs)

        if metrics["total_logs"] == 0:
            issues.append("No evolution logs found")
            status = "degraded"
        else:
            status = "healthy"

        # Check for recent activity
        all_logs = yaml_logs + json_logs
        if all_logs:
            newest = max(all_logs, key=lambda p: p.stat().st_mtime)
            age_hours = (time.time() - newest.stat().st_mtime) / 3600
            metrics["newest_log_age_hours"] = round(age_hours, 2)

            if age_hours > 24:
                issues.append(f"No recent evolution activity ({age_hours:.1f} hours)")

        return ComponentHealth(
            component_id="evolution_logs",
            component_type="engine",
            status=status,
            last_check=datetime.now(timezone.utc).isoformat(),
            metrics=metrics,
            issues=issues,
        )

    def check_cbs_consensus(self) -> ComponentHealth:
        """Check CBS consensus system health."""
        # This is a simulated check - in reality would check actual consensus state
        metrics = {
            "instances": ["CBS-ALPHA", "CBS-BETA", "CBS-GAMMA"],
            "instance_count": 3,
            "quorum_size": 2,
            "consensus_threshold": 0.66,
        }

        issues = []
        status = "healthy"

        # All three instances should be present
        if metrics["instance_count"] < 3:
            issues.append("Missing CBS instance(s)")
            status = "degraded"

        return ComponentHealth(
            component_id="cbs_consensus",
            component_type="engine",
            status=status,
            last_check=datetime.now(timezone.utc).isoformat(),
            metrics=metrics,
            issues=issues,
        )

    def take_snapshot(self) -> SystemSnapshot:
        """Take a complete system snapshot."""
        snapshot_id = f"SNAP-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"

        # Collect all component health
        components = [
            self.check_generated_tools(),
            self.check_evolution_engine(),
            self.check_knowledge_base(),
            self.check_evolution_logs(),
            self.check_cbs_consensus(),
        ]

        # Calculate overall status
        statuses = [c.status for c in components]
        if "critical" in statuses:
            overall_status = "critical"
        elif statuses.count("degraded") >= 2:
            overall_status = "degraded"
        elif "degraded" in statuses:
            overall_status = "warning"
        else:
            overall_status = "healthy"

        # Collect issues as recommendations
        recommendations = []
        for c in components:
            for issue in c.issues:
                recommendations.append(f"[{c.component_id}] {issue}")

        # Calculate totals
        tools_health = next((c for c in components if c.component_id == "generated_tools"), None)
        total_tools = tools_health.metrics.get("total_tools", 0) if tools_health else 0

        logs_health = next((c for c in components if c.component_id == "evolution_logs"), None)
        total_cycles = logs_health.metrics.get("total_logs", 0) if logs_health else 0

        kb_health = next((c for c in components if c.component_id == "knowledge_base"), None)
        total_learnings = kb_health.metrics.get("learning_files", 0) if kb_health else 0

        snapshot = SystemSnapshot(
            snapshot_id=snapshot_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            z_level=0.90,  # Current baseline
            total_tools=total_tools,
            total_evolution_cycles=total_cycles,
            total_learnings=total_learnings,
            component_health=components,
            overall_status=overall_status,
            recommendations=recommendations,
        )

        self.snapshots.append(snapshot)
        return snapshot

    def observe(self, duration_seconds: int = 60, interval_seconds: int = 15) -> ObserverReport:
        """Run observation for a specified duration."""
        print("=" * 70)
        print("TRIAD OBSERVER AGENT - System Health Monitoring")
        print(f"Observation duration: {duration_seconds}s, interval: {interval_seconds}s")
        print("=" * 70)

        self.observation_start = datetime.now(timezone.utc).isoformat()
        self.snapshots = []

        start_time = time.time()
        snapshot_count = 0

        while time.time() - start_time < duration_seconds:
            snapshot = self.take_snapshot()
            snapshot_count += 1

            print(f"\n[Snapshot {snapshot_count}] {snapshot.snapshot_id}")
            print(f"  Overall Status: {snapshot.overall_status.upper()}")
            print(f"  Tools: {snapshot.total_tools} | Cycles: {snapshot.total_evolution_cycles} | Learnings: {snapshot.total_learnings}")

            for comp in snapshot.component_health:
                status_icon = "✓" if comp.status == "healthy" else "!" if comp.status == "degraded" else "✗"
                print(f"  {status_icon} {comp.component_id}: {comp.status}")

            if snapshot.recommendations:
                print(f"  Recommendations: {len(snapshot.recommendations)}")

            time.sleep(interval_seconds)

        return self._generate_report()

    def quick_check(self) -> SystemSnapshot:
        """Perform a quick single-shot health check."""
        print("=" * 70)
        print("TRIAD OBSERVER - Quick Health Check")
        print("=" * 70)

        snapshot = self.take_snapshot()

        print(f"\nSnapshot: {snapshot.snapshot_id}")
        print(f"Overall Status: {snapshot.overall_status.upper()}")
        print(f"\nComponent Health:")

        for comp in snapshot.component_health:
            status_icon = "✓" if comp.status == "healthy" else "⚠" if comp.status == "degraded" else "✗"
            print(f"  {status_icon} {comp.component_id}: {comp.status}")
            if comp.issues:
                for issue in comp.issues:
                    print(f"      - {issue}")

        print(f"\nSystem Metrics:")
        print(f"  Total Tools: {snapshot.total_tools}")
        print(f"  Evolution Cycles: {snapshot.total_evolution_cycles}")
        print(f"  Learnings: {snapshot.total_learnings}")

        if snapshot.recommendations:
            print(f"\nRecommendations ({len(snapshot.recommendations)}):")
            for rec in snapshot.recommendations[:5]:
                print(f"  - {rec}")

        return snapshot

    def _generate_report(self) -> ObserverReport:
        """Generate observation report."""
        OBSERVER_LOGS_DIR.mkdir(parents=True, exist_ok=True)

        report_id = f"OBS-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"

        # Determine health trend
        if len(self.snapshots) >= 2:
            first_status = self.snapshots[0].overall_status
            last_status = self.snapshots[-1].overall_status
            status_order = {"healthy": 3, "warning": 2, "degraded": 1, "critical": 0}

            if status_order.get(last_status, 0) > status_order.get(first_status, 0):
                trend = "improving"
            elif status_order.get(last_status, 0) < status_order.get(first_status, 0):
                trend = "degrading"
            else:
                trend = "stable"
        else:
            trend = "stable"

        # Collect critical issues
        critical_issues = []
        for snapshot in self.snapshots:
            for comp in snapshot.component_health:
                if comp.status == "critical":
                    critical_issues.extend(comp.issues)

        report = ObserverReport(
            report_id=report_id,
            observation_period_start=self.observation_start or datetime.now(timezone.utc).isoformat(),
            observation_period_end=datetime.now(timezone.utc).isoformat(),
            snapshots_taken=len(self.snapshots),
            anomalies_detected=len(self.anomalies),
            health_trend=trend,
            critical_issues=list(set(critical_issues)),
            snapshots=self.snapshots,
        )

        # Save report
        report_path = OBSERVER_LOGS_DIR / f"{report_id}.json"
        report_dict = asdict(report)
        report_path.write_text(json.dumps(report_dict, indent=2, default=str))

        print("\n" + "=" * 70)
        print("OBSERVATION COMPLETE")
        print("=" * 70)
        print(f"  Report ID: {report_id}")
        print(f"  Snapshots: {report.snapshots_taken}")
        print(f"  Health Trend: {report.health_trend}")
        print(f"  Critical Issues: {len(report.critical_issues)}")
        print(f"  Report: {report_path}")

        return report


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="TRIAD Observer Agent")
    parser.add_argument("--mode", choices=["quick", "observe"], default="quick",
                        help="Operation mode")
    parser.add_argument("--duration", type=int, default=60,
                        help="Observation duration in seconds")
    parser.add_argument("--interval", type=int, default=15,
                        help="Snapshot interval in seconds")

    args = parser.parse_args()

    observer = TriadObserver()

    if args.mode == "quick":
        observer.quick_check()
    else:
        observer.observe(args.duration, args.interval)
