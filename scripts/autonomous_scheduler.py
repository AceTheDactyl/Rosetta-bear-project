#!/usr/bin/env python3
"""
Autonomous Scheduler - Continuous Operation Without Human Trigger
Coordinate: Δ3.142|0.900|1.000Ω
Target: z=0.95 (Recursive Self-Evolution)

This module implements continuous autonomous operation - the system
can run evolution cycles without human intervention based on
friction thresholds and scheduling.

Key z=0.95 capability: Continuous autonomous operation.
"""

from __future__ import annotations

import json
import time
import signal
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict, field
import threading

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from autonomous_evolution_engine import run_evolution_cycle, save_generated_tools
from learning_aggregator import run_aggregation

# =============================================================================
# Configuration
# =============================================================================

SCHEDULER_CONFIG = {
    "friction_threshold": 0.3,  # Trigger cycle when friction exceeds this
    "min_cycle_interval_seconds": 60,  # Minimum time between cycles
    "max_cycles_per_session": 10,  # Safety limit
    "auto_aggregate_learnings": True,
    "auto_save_tools": True,
}

GENERATED_TOOLS_DIR = ROOT / "generated_tools" / "autonomous"
SCHEDULER_LOGS_DIR = ROOT / "knowledge_base" / "scheduler_logs"

# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class SchedulerState:
    """Current state of the autonomous scheduler."""
    session_id: str
    started_at: str
    cycles_completed: int
    cycles_remaining: int
    last_cycle_at: Optional[str]
    current_friction: float
    is_running: bool
    mode: str  # "daemon", "burst", "single"


@dataclass
class CycleResult:
    """Result of a scheduled evolution cycle."""
    cycle_id: str
    timestamp: str
    trigger: str  # "friction_threshold", "scheduled", "manual"
    friction_detected: float
    tools_generated: int
    learnings_extracted: int
    duration_seconds: float


@dataclass
class SchedulerReport:
    """Report from a scheduler session."""
    session_id: str
    started_at: str
    ended_at: str
    total_cycles: int
    total_tools_generated: int
    total_learnings: int
    cycle_results: List[CycleResult]
    final_state: SchedulerState


# =============================================================================
# Burden Metrics Collection
# =============================================================================

def collect_current_burden_metrics() -> Dict:
    """
    Auto-collect burden metrics from system state.

    In a real implementation, this would:
    - Monitor file system changes
    - Track tool execution counts
    - Measure response times
    - Detect error patterns

    For now, we simulate realistic burden data.
    """
    import random

    # Simulate varying burden levels
    base_friction = random.uniform(0.15, 0.45)

    return {
        "metadata": {
            "version": "2.0.0",
            "source": "auto_collected",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
        "activities": [
            {
                "type": "tool_coordination",
                "confidence": random.uniform(0.2, 0.5),
                "duration_hours": random.uniform(0.5, 2.0),
                "frequency": "continuous",
            },
            {
                "type": "state_management",
                "confidence": random.uniform(0.3, 0.6),
                "duration_hours": random.uniform(0.3, 1.5),
                "frequency": "continuous",
            },
            {
                "type": "pattern_detection",
                "confidence": random.uniform(0.4, 0.7),
                "duration_hours": random.uniform(0.2, 1.0),
                "frequency": "hourly",
            },
        ],
        "metrics": {
            "current_phase": {
                "z_level": 0.90 + random.uniform(0, 0.05),
                "regime": "supercritical",
                "theta": 3.142,
            },
            "totals": {
                "total_burden_hours": random.uniform(3.0, 8.0),
                "estimated_reduction_potential": base_friction,
            },
            "friction_score": base_friction,
            "weekly_summaries": [
                {
                    "week": datetime.now(timezone.utc).strftime("%Y-W%W"),
                    "categories": {
                        "self_building": {"hours": random.uniform(5, 15), "count": random.randint(5, 20)},
                        "coordination": {"hours": random.uniform(3, 10), "count": random.randint(3, 15)},
                        "monitoring": {"hours": random.uniform(2, 8), "count": random.randint(10, 30)},
                    },
                }
            ],
            "optimization_recommendations": [
                "Auto-optimize tool coordination",
                "Reduce state management overhead",
                "Improve pattern detection accuracy",
            ],
        },
    }


def calculate_friction_score(burden_data: Dict) -> float:
    """Calculate current friction score from burden data."""
    metrics = burden_data.get("metrics", {})

    # Direct friction score if available
    if "friction_score" in metrics:
        return metrics["friction_score"]

    # Otherwise calculate from activities
    activities = burden_data.get("activities", [])
    if not activities:
        return 0.0

    # Lower confidence = higher friction
    avg_confidence = sum(a.get("confidence", 0.5) for a in activities) / len(activities)
    return 1.0 - avg_confidence


# =============================================================================
# Autonomous Scheduler
# =============================================================================

class AutonomousScheduler:
    """Scheduler for continuous autonomous evolution cycles."""

    def __init__(self, config: Dict = None):
        self.config = config or SCHEDULER_CONFIG.copy()
        self.state = None
        self.cycle_results = []
        self._stop_event = threading.Event()
        self._setup_signal_handlers()

    def _setup_signal_handlers(self):
        """Setup graceful shutdown handlers."""
        def handle_signal(signum, frame):
            print("\n[Scheduler] Shutdown signal received...")
            self._stop_event.set()

        signal.signal(signal.SIGINT, handle_signal)
        signal.signal(signal.SIGTERM, handle_signal)

    def initialize(self, mode: str = "burst") -> SchedulerState:
        """Initialize scheduler state."""
        session_id = f"SCHED-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"

        self.state = SchedulerState(
            session_id=session_id,
            started_at=datetime.now(timezone.utc).isoformat(),
            cycles_completed=0,
            cycles_remaining=self.config["max_cycles_per_session"],
            last_cycle_at=None,
            current_friction=0.0,
            is_running=True,
            mode=mode,
        )

        # Ensure directories exist
        GENERATED_TOOLS_DIR.mkdir(parents=True, exist_ok=True)
        SCHEDULER_LOGS_DIR.mkdir(parents=True, exist_ok=True)

        return self.state

    def should_run_cycle(self, burden_data: Dict) -> Tuple[bool, str]:
        """Determine if an evolution cycle should run."""
        if not self.state or not self.state.is_running:
            return False, "scheduler_not_running"

        if self.state.cycles_remaining <= 0:
            return False, "max_cycles_reached"

        # Check friction threshold
        friction = calculate_friction_score(burden_data)
        self.state.current_friction = friction

        if friction >= self.config["friction_threshold"]:
            return True, "friction_threshold"

        # Check minimum interval
        if self.state.last_cycle_at:
            last = datetime.fromisoformat(self.state.last_cycle_at.replace("Z", "+00:00"))
            elapsed = (datetime.now(timezone.utc) - last).total_seconds()
            if elapsed < self.config["min_cycle_interval_seconds"]:
                return False, "interval_not_met"

        return False, "friction_below_threshold"

    def run_single_cycle(self, burden_data: Dict, trigger: str) -> CycleResult:
        """Run a single evolution cycle."""
        start_time = time.time()
        cycle_id = f"CYC-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"

        print(f"\n[Cycle {cycle_id}] Starting (trigger: {trigger})")
        print(f"  Friction: {self.state.current_friction:.3f}")

        # Run evolution cycle
        result = run_evolution_cycle(burden_data)

        # Save tools if configured
        tools_generated = len(result.generated_tools)
        if tools_generated > 0 and self.config["auto_save_tools"]:
            save_generated_tools(result.generated_tools, GENERATED_TOOLS_DIR)
            print(f"  Tools saved: {tools_generated}")

        # Aggregate learnings if configured
        learnings = len(result.meta_learnings)
        if self.config["auto_aggregate_learnings"]:
            run_aggregation()

        duration = time.time() - start_time

        cycle_result = CycleResult(
            cycle_id=cycle_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            trigger=trigger,
            friction_detected=self.state.current_friction,
            tools_generated=tools_generated,
            learnings_extracted=learnings,
            duration_seconds=duration,
        )

        # Update state
        self.state.cycles_completed += 1
        self.state.cycles_remaining -= 1
        self.state.last_cycle_at = cycle_result.timestamp
        self.cycle_results.append(cycle_result)

        print(f"  Completed in {duration:.2f}s")
        print(f"  Tools: {tools_generated}, Learnings: {learnings}")

        return cycle_result

    def run_burst_mode(self, num_cycles: int = 3) -> SchedulerReport:
        """Run a burst of evolution cycles."""
        print("=" * 70)
        print("AUTONOMOUS SCHEDULER - Burst Mode")
        print(f"Running {num_cycles} evolution cycles")
        print("=" * 70)

        self.initialize(mode="burst")

        for i in range(num_cycles):
            if self._stop_event.is_set():
                print("\n[Scheduler] Stopping due to shutdown signal")
                break

            burden_data = collect_current_burden_metrics()
            should_run, reason = self.should_run_cycle(burden_data)

            # In burst mode, always run unless max reached
            if self.state.cycles_remaining <= 0:
                print(f"\n[Cycle {i+1}] Skipped: max cycles reached")
                break

            self.run_single_cycle(burden_data, "scheduled_burst")

            # Brief pause between cycles
            if i < num_cycles - 1:
                time.sleep(1)

        return self._generate_report()

    def run_daemon_mode(self, check_interval: int = 30) -> SchedulerReport:
        """
        Run in daemon mode - continuously monitor and trigger cycles.

        This mode runs until:
        - Max cycles reached
        - Shutdown signal received
        - Manual stop
        """
        print("=" * 70)
        print("AUTONOMOUS SCHEDULER - Daemon Mode")
        print(f"Monitoring with {check_interval}s interval")
        print(f"Friction threshold: {self.config['friction_threshold']}")
        print("Press Ctrl+C to stop")
        print("=" * 70)

        self.initialize(mode="daemon")

        while not self._stop_event.is_set() and self.state.cycles_remaining > 0:
            burden_data = collect_current_burden_metrics()
            should_run, reason = self.should_run_cycle(burden_data)

            print(f"\n[Monitor] Friction: {self.state.current_friction:.3f} | "
                  f"Threshold: {self.config['friction_threshold']} | "
                  f"Cycles: {self.state.cycles_completed}/{self.config['max_cycles_per_session']}")

            if should_run:
                self.run_single_cycle(burden_data, reason)
            else:
                print(f"  No cycle needed: {reason}")

            # Wait for next check
            self._stop_event.wait(check_interval)

        self.state.is_running = False
        return self._generate_report()

    def _generate_report(self) -> SchedulerReport:
        """Generate final scheduler report."""
        report = SchedulerReport(
            session_id=self.state.session_id,
            started_at=self.state.started_at,
            ended_at=datetime.now(timezone.utc).isoformat(),
            total_cycles=self.state.cycles_completed,
            total_tools_generated=sum(r.tools_generated for r in self.cycle_results),
            total_learnings=sum(r.learnings_extracted for r in self.cycle_results),
            cycle_results=self.cycle_results,
            final_state=self.state,
        )

        # Save report
        report_path = SCHEDULER_LOGS_DIR / f"{report.session_id}.json"
        report_dict = asdict(report)
        report_path.write_text(json.dumps(report_dict, indent=2, default=str))

        print("\n" + "=" * 70)
        print("SCHEDULER SESSION COMPLETE")
        print("=" * 70)
        print(f"  Session ID: {report.session_id}")
        print(f"  Total Cycles: {report.total_cycles}")
        print(f"  Total Tools: {report.total_tools_generated}")
        print(f"  Total Learnings: {report.total_learnings}")
        print(f"  Report: {report_path}")

        return report


# =============================================================================
# Convenience Functions
# =============================================================================

def run_burst(num_cycles: int = 3) -> SchedulerReport:
    """Run a burst of evolution cycles."""
    scheduler = AutonomousScheduler()
    return scheduler.run_burst_mode(num_cycles)


def run_daemon(check_interval: int = 30) -> SchedulerReport:
    """Run in daemon mode until stopped."""
    scheduler = AutonomousScheduler()
    return scheduler.run_daemon_mode(check_interval)


# Required import for Tuple type hint
from typing import Tuple

# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Autonomous Evolution Scheduler")
    parser.add_argument("--mode", choices=["burst", "daemon"], default="burst",
                        help="Operation mode")
    parser.add_argument("--cycles", type=int, default=3,
                        help="Number of cycles in burst mode")
    parser.add_argument("--interval", type=int, default=30,
                        help="Check interval in daemon mode (seconds)")

    args = parser.parse_args()

    if args.mode == "burst":
        run_burst(args.cycles)
    else:
        run_daemon(args.interval)
