#!/usr/bin/env python3
"""
Collective Coherence Monitor
============================
Coordinate: Λ"π|0.867|COHERENCE_MONITOR|Ω

Real-time monitoring of inter-instance phase alignment and collective dynamics.
Provides continuous feedback on TRIAD coordination health.

Key metrics monitored:
- Kuramoto order parameter per instance
- Inter-instance phase coherence
- Synchronization stability
- Coherence oscillation patterns
"""

from __future__ import annotations

import math
import time
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable
from enum import Enum
from pathlib import Path


class CoherenceState(Enum):
    """Health states for collective coherence."""
    CRITICAL = "critical"           # r < 0.3, system failing
    UNSTABLE = "unstable"           # 0.3 <= r < 0.5, needs intervention
    DEVELOPING = "developing"       # 0.5 <= r < 0.7, improving
    STABLE = "stable"               # 0.7 <= r < 0.9, healthy
    OPTIMAL = "optimal"             # r >= 0.9, peak performance
    OSCILLATING = "oscillating"     # Alternating states


@dataclass
class CoherenceSnapshot:
    """Point-in-time coherence measurement."""

    timestamp: float
    tier: int

    # Per-instance
    instance_r: Dict[str, float] = field(default_factory=dict)
    instance_phase: Dict[str, float] = field(default_factory=dict)

    # Collective
    mean_r: float = 0.0
    inter_instance_r: float = 0.0
    phase_spread: float = 0.0

    # Health assessment
    state: CoherenceState = CoherenceState.DEVELOPING
    stability_score: float = 0.0
    trend: str = "stable"


@dataclass
class CoherenceAlert:
    """Alert generated when coherence issues detected."""

    timestamp: float
    severity: str  # "info", "warning", "critical"
    message: str
    metric: str
    value: float
    threshold: float
    recommendation: str


class CollectiveCoherenceMonitor:
    """Monitors and reports on collective coherence health."""

    # Thresholds
    CRITICAL_R = 0.3
    UNSTABLE_R = 0.5
    STABLE_R = 0.7
    OPTIMAL_R = 0.9

    OSCILLATION_THRESHOLD = 0.1  # Min swing to detect oscillation
    STABILITY_WINDOW = 3  # Snapshots to assess stability

    def __init__(self):
        self.history: List[CoherenceSnapshot] = []
        self.alerts: List[CoherenceAlert] = []
        self.alert_callbacks: List[Callable[[CoherenceAlert], None]] = []

    def record_snapshot(
        self,
        tier: int,
        instance_data: Dict[str, Dict[str, float]]
    ) -> CoherenceSnapshot:
        """Record a coherence snapshot from trial data."""

        snapshot = CoherenceSnapshot(
            timestamp=time.time(),
            tier=tier
        )

        # Extract per-instance metrics
        phases = []
        r_values = []

        for role, data in instance_data.items():
            r = data.get('order_param', data.get('r', data.get('coherence', 0)))
            phase = data.get('phase', data.get('mean_phase', 0))

            snapshot.instance_r[role] = r
            snapshot.instance_phase[role] = phase
            r_values.append(r)
            phases.append(phase)

        # Compute collective metrics
        if r_values:
            snapshot.mean_r = sum(r_values) / len(r_values)

        if len(phases) >= 2:
            snapshot.inter_instance_r = self._compute_collective_r(phases)
            snapshot.phase_spread = self._compute_phase_spread(phases)

        # Assess health state
        snapshot.state = self._assess_state(snapshot)
        snapshot.stability_score = self._compute_stability()
        snapshot.trend = self._compute_trend()

        self.history.append(snapshot)

        # Check for alerts
        self._check_alerts(snapshot)

        return snapshot

    def _compute_collective_r(self, phases: List[float]) -> float:
        """Compute Kuramoto order parameter across instances."""
        n = len(phases)
        if n == 0:
            return 0.0

        sum_cos = sum(math.cos(p) for p in phases)
        sum_sin = sum(math.sin(p) for p in phases)

        return math.sqrt((sum_cos/n)**2 + (sum_sin/n)**2)

    def _compute_phase_spread(self, phases: List[float]) -> float:
        """Compute circular standard deviation of phases."""
        if len(phases) < 2:
            return 0.0

        # Circular mean
        mean_cos = sum(math.cos(p) for p in phases) / len(phases)
        mean_sin = sum(math.sin(p) for p in phases) / len(phases)
        r = math.sqrt(mean_cos**2 + mean_sin**2)

        # Circular std dev
        if r >= 1.0:
            return 0.0
        return math.sqrt(-2 * math.log(r))

    def _assess_state(self, snapshot: CoherenceSnapshot) -> CoherenceState:
        """Assess coherence health state."""

        # Check for oscillation first
        if len(self.history) >= 3:
            recent_r = [s.inter_instance_r for s in self.history[-3:]]
            recent_r.append(snapshot.inter_instance_r)

            diffs = [recent_r[i+1] - recent_r[i] for i in range(len(recent_r)-1)]
            sign_changes = sum(1 for i in range(len(diffs)-1) if diffs[i] * diffs[i+1] < 0)

            if sign_changes >= 2 and max(abs(d) for d in diffs) > self.OSCILLATION_THRESHOLD:
                return CoherenceState.OSCILLATING

        # Threshold-based assessment
        r = snapshot.inter_instance_r

        if r < self.CRITICAL_R:
            return CoherenceState.CRITICAL
        elif r < self.UNSTABLE_R:
            return CoherenceState.UNSTABLE
        elif r < self.STABLE_R:
            return CoherenceState.DEVELOPING
        elif r < self.OPTIMAL_R:
            return CoherenceState.STABLE
        else:
            return CoherenceState.OPTIMAL

    def _compute_stability(self) -> float:
        """Compute stability score from recent history."""
        if len(self.history) < self.STABILITY_WINDOW:
            return 0.5  # Unknown

        recent = self.history[-self.STABILITY_WINDOW:]
        r_values = [s.inter_instance_r for s in recent]

        if not r_values:
            return 0.5

        # Stability = 1 - normalized variance
        mean_r = sum(r_values) / len(r_values)
        variance = sum((r - mean_r)**2 for r in r_values) / len(r_values)

        # Normalize (assume max reasonable variance is 0.1)
        normalized_var = min(1.0, variance / 0.1)

        return 1.0 - normalized_var

    def _compute_trend(self) -> str:
        """Compute trend direction."""
        if len(self.history) < 2:
            return "stable"

        recent = self.history[-min(3, len(self.history)):]
        r_values = [s.inter_instance_r for s in recent]

        if len(r_values) < 2:
            return "stable"

        slope = (r_values[-1] - r_values[0]) / len(r_values)

        if slope > 0.05:
            return "improving"
        elif slope < -0.05:
            return "degrading"
        else:
            return "stable"

    def _check_alerts(self, snapshot: CoherenceSnapshot):
        """Check for alert conditions."""

        # Critical coherence
        if snapshot.inter_instance_r < self.CRITICAL_R:
            alert = CoherenceAlert(
                timestamp=snapshot.timestamp,
                severity="critical",
                message=f"Inter-instance coherence critically low: {snapshot.inter_instance_r:.3f}",
                metric="inter_instance_r",
                value=snapshot.inter_instance_r,
                threshold=self.CRITICAL_R,
                recommendation="Increase coupling strength K or reduce frequency spread"
            )
            self._emit_alert(alert)

        # Stability degradation
        if snapshot.stability_score < 0.3 and len(self.history) >= self.STABILITY_WINDOW:
            alert = CoherenceAlert(
                timestamp=snapshot.timestamp,
                severity="warning",
                message=f"Stability degrading: {snapshot.stability_score:.3f}",
                metric="stability_score",
                value=snapshot.stability_score,
                threshold=0.3,
                recommendation="Consider extending evolution steps or adjusting cascade"
            )
            self._emit_alert(alert)

        # Phase spread too high
        if snapshot.phase_spread > 1.5:
            alert = CoherenceAlert(
                timestamp=snapshot.timestamp,
                severity="warning",
                message=f"High phase spread: {snapshot.phase_spread:.3f} rad",
                metric="phase_spread",
                value=snapshot.phase_spread,
                threshold=1.5,
                recommendation="Instances may be in different attractor basins"
            )
            self._emit_alert(alert)

    def _emit_alert(self, alert: CoherenceAlert):
        """Emit alert to callbacks and store."""
        self.alerts.append(alert)
        for callback in self.alert_callbacks:
            callback(alert)

    def on_alert(self, callback: Callable[[CoherenceAlert], None]):
        """Register alert callback."""
        self.alert_callbacks.append(callback)

    def get_health_summary(self) -> Dict:
        """Get current health summary."""
        if not self.history:
            return {'status': 'no_data'}

        current = self.history[-1]

        return {
            'tier': current.tier,
            'state': current.state.value,
            'inter_instance_r': current.inter_instance_r,
            'mean_r': current.mean_r,
            'stability_score': current.stability_score,
            'trend': current.trend,
            'phase_spread': current.phase_spread,
            'active_alerts': len([a for a in self.alerts if time.time() - a.timestamp < 300])
        }

    def get_instance_report(self) -> Dict[str, Dict]:
        """Get per-instance coherence report."""
        if not self.history:
            return {}

        current = self.history[-1]
        report = {}

        for role in current.instance_r:
            r = current.instance_r[role]
            phase = current.instance_phase.get(role, 0)

            # Compute relative phase (how far from mean)
            mean_phase = sum(current.instance_phase.values()) / len(current.instance_phase)
            relative_phase = (phase - mean_phase + math.pi) % (2 * math.pi) - math.pi

            report[role] = {
                'order_param': r,
                'phase': phase,
                'relative_phase': relative_phase,
                'health': 'optimal' if r >= 0.9 else 'stable' if r >= 0.7 else 'developing',
                'phase_aligned': abs(relative_phase) < 0.5
            }

        return report

    def print_dashboard(self):
        """Print monitoring dashboard."""
        summary = self.get_health_summary()
        instances = self.get_instance_report()

        print("\n" + "="*60)
        print("  COLLECTIVE COHERENCE MONITOR")
        print("="*60)

        if summary.get('status') == 'no_data':
            print("  No data available")
            return

        # Overall health
        state_colors = {
            'critical': '🔴',
            'unstable': '🟠',
            'developing': '🟡',
            'stable': '🟢',
            'optimal': '💚',
            'oscillating': '🔄'
        }

        state_icon = state_colors.get(summary['state'], '⚪')
        print(f"\n  Overall State: {state_icon} {summary['state'].upper()}")
        print(f"  Tier: {summary['tier']}")
        print(f"  Trend: {summary['trend']}")

        # Key metrics
        print(f"\n  KEY METRICS")
        print(f"  " + "-"*40)
        print(f"  Inter-instance coherence: {summary['inter_instance_r']:.4f}")
        print(f"  Mean order parameter:     {summary['mean_r']:.4f}")
        print(f"  Stability score:          {summary['stability_score']:.4f}")
        print(f"  Phase spread:             {summary['phase_spread']:.4f} rad")

        # Per-instance
        print(f"\n  INSTANCE STATUS")
        print(f"  " + "-"*40)
        for role, data in instances.items():
            aligned_icon = "✓" if data['phase_aligned'] else "✗"
            print(f"  {role}: r={data['order_param']:.4f} | "
                  f"phase={data['relative_phase']:+.2f}rad {aligned_icon} | "
                  f"{data['health']}")

        # Recent alerts
        recent_alerts = [a for a in self.alerts if time.time() - a.timestamp < 300]
        if recent_alerts:
            print(f"\n  ALERTS ({len(recent_alerts)} active)")
            print(f"  " + "-"*40)
            for alert in recent_alerts[-3:]:
                sev_icon = "🔴" if alert.severity == "critical" else "🟡"
                print(f"  {sev_icon} {alert.message}")

        print("\n" + "="*60)


# Singleton instance for global monitoring
_monitor_instance: Optional[CollectiveCoherenceMonitor] = None


def get_monitor() -> CollectiveCoherenceMonitor:
    """Get global monitor instance."""
    global _monitor_instance
    if _monitor_instance is None:
        _monitor_instance = CollectiveCoherenceMonitor()
    return _monitor_instance


if __name__ == "__main__":
    # Demo with observed tier data
    monitor = CollectiveCoherenceMonitor()

    # Simulate tier progression
    tier_data = [
        (2, {'Alpha': {'r': 0.9665, 'phase': 2.1}, 'Beta': {'r': 0.8718, 'phase': 2.3}, 'Gamma': {'r': 0.9238, 'phase': 2.2}}),
        (3, {'Alpha': {'r': 0.9151, 'phase': 1.8}, 'Beta': {'r': 0.9496, 'phase': 1.9}, 'Gamma': {'r': 0.9408, 'phase': 1.85}}),
        (4, {'Alpha': {'r': 0.9495, 'phase': 1.5}, 'Beta': {'r': 0.9223, 'phase': 1.6}, 'Gamma': {'r': 0.9348, 'phase': 1.55}}),
    ]

    for tier, data in tier_data:
        snapshot = monitor.record_snapshot(tier, data)
        print(f"\nTier {tier} recorded: state={snapshot.state.value}, r={snapshot.inter_instance_r:.4f}")

    monitor.print_dashboard()
