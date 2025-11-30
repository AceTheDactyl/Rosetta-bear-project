#!/usr/bin/env python3
"""
Emergence Field Analyzer
========================
Coordinate: Λ"π|0.867|EMERGENCE_FIELD|Ω

Analyzes emergent collective dynamics across TRIAD instances:
- Inter-instance phase coherence transitions
- Collective attractor identification
- Information flow topology
- Phase transition detection

Based on observed emergence patterns:
- T2→T3: Inter-instance coherence 0.32→0.84 (phase transition)
- Stable plateau at ~92% global sync (attractor)
- Information preservation scaling with cascade multiplier
"""

from __future__ import annotations

import math
import json
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from enum import Enum
import statistics


class EmergenceType(Enum):
    """Types of emergent phenomena detected."""
    PHASE_TRANSITION = "phase_transition"
    ATTRACTOR_FORMATION = "attractor_formation"
    COLLECTIVE_COHERENCE = "collective_coherence"
    INFORMATION_CRYSTALLIZATION = "information_crystallization"
    SYMMETRY_BREAKING = "symmetry_breaking"
    CASCADE_AMPLIFICATION = "cascade_amplification"


@dataclass
class FieldState:
    """Represents the state of the collective emergence field."""

    tier: int
    timestamp: float = field(default_factory=time.time)

    # Per-instance metrics
    instance_order_params: Dict[str, float] = field(default_factory=dict)
    instance_phases: Dict[str, float] = field(default_factory=dict)
    instance_locked_counts: Dict[str, int] = field(default_factory=dict)

    # Collective metrics
    mean_order_param: float = 0.0
    inter_instance_coherence: float = 0.0
    global_sync_ratio: float = 0.0
    information_preserved: float = 0.0
    cascade_multiplier: float = 1.0

    # Field topology
    field_gradient: float = 0.0
    field_curvature: float = 0.0
    attractor_strength: float = 0.0


@dataclass
class EmergenceEvent:
    """Represents a detected emergence event."""

    event_type: EmergenceType
    tier: int
    timestamp: float
    magnitude: float
    description: str
    metrics_before: Dict[str, float] = field(default_factory=dict)
    metrics_after: Dict[str, float] = field(default_factory=dict)
    significance: float = 0.0  # 0-1 scale


@dataclass
class FieldAnalysis:
    """Complete field analysis results."""

    states: List[FieldState] = field(default_factory=list)
    events: List[EmergenceEvent] = field(default_factory=list)

    # Trajectory analysis
    coherence_trajectory: List[float] = field(default_factory=list)
    sync_trajectory: List[float] = field(default_factory=list)
    information_trajectory: List[float] = field(default_factory=list)

    # Attractor identification
    identified_attractors: List[Dict] = field(default_factory=list)
    current_basin: Optional[str] = None

    # Phase transition points
    transition_tiers: List[int] = field(default_factory=list)

    # Predictive metrics
    next_tier_prediction: Dict[str, float] = field(default_factory=dict)


class EmergenceFieldAnalyzer:
    """Analyzes emergent dynamics in the collective TRIAD field."""

    # Thresholds for emergence detection
    COHERENCE_JUMP_THRESHOLD = 0.3  # Minimum jump to detect phase transition
    PLATEAU_TOLERANCE = 0.02  # Tolerance for attractor detection
    PLATEAU_MIN_TIERS = 2  # Minimum tiers at plateau to confirm attractor

    def __init__(self):
        self.field_history: List[FieldState] = []
        self.events: List[EmergenceEvent] = []

    def ingest_tier_results(self, tier: int, results: Dict, cascade: float) -> FieldState:
        """Ingest results from a tier execution and create field state."""
        state = FieldState(tier=tier, cascade_multiplier=cascade)

        # Extract per-instance metrics
        for role in ['Alpha', 'Beta', 'Gamma']:
            if role in results and results[role]:
                resonance = results[role].get('resonance')
                transfig = results[role].get('transfiguration')

                if resonance:
                    if hasattr(resonance, 'coherence'):
                        state.instance_order_params[role] = resonance.coherence
                        state.instance_phases[role] = resonance.mean_phase
                        state.instance_locked_counts[role] = resonance.phase_locked_count
                    elif isinstance(resonance, dict):
                        state.instance_order_params[role] = resonance.get('coherence', 0)
                        state.instance_phases[role] = resonance.get('mean_phase', 0)
                        state.instance_locked_counts[role] = resonance.get('phase_locked_count', 0)

                if transfig:
                    if hasattr(transfig, 'information_preserved'):
                        state.information_preserved = transfig.information_preserved
                    elif isinstance(transfig, dict):
                        state.information_preserved = transfig.get('information_preserved', 0)

        # Compute collective metrics
        if state.instance_order_params:
            state.mean_order_param = statistics.mean(state.instance_order_params.values())

            # Compute inter-instance coherence
            phases = list(state.instance_phases.values())
            if len(phases) >= 2:
                state.inter_instance_coherence = self._compute_phase_coherence(phases)

            # Global sync
            total_locked = sum(state.instance_locked_counts.values())
            total_oscillators = len(state.instance_locked_counts) * 100
            state.global_sync_ratio = total_locked / total_oscillators if total_oscillators > 0 else 0

        # Compute field topology if we have history
        if self.field_history:
            prev = self.field_history[-1]
            state.field_gradient = state.inter_instance_coherence - prev.inter_instance_coherence

            if len(self.field_history) >= 2:
                prev_gradient = prev.inter_instance_coherence - self.field_history[-2].inter_instance_coherence
                state.field_curvature = state.field_gradient - prev_gradient

        self.field_history.append(state)
        return state

    def _compute_phase_coherence(self, phases: List[float]) -> float:
        """Compute Kuramoto-style coherence across phases."""
        if len(phases) < 2:
            return 1.0

        n = len(phases)
        sum_cos = sum(math.cos(p) for p in phases)
        sum_sin = sum(math.sin(p) for p in phases)

        r = math.sqrt((sum_cos/n)**2 + (sum_sin/n)**2)
        return r

    def detect_emergence_events(self) -> List[EmergenceEvent]:
        """Analyze field history for emergence events."""
        if len(self.field_history) < 2:
            return []

        new_events = []

        for i in range(1, len(self.field_history)):
            prev = self.field_history[i-1]
            curr = self.field_history[i]

            # Detect phase transition in inter-instance coherence
            coherence_jump = curr.inter_instance_coherence - prev.inter_instance_coherence
            if abs(coherence_jump) > self.COHERENCE_JUMP_THRESHOLD:
                event = EmergenceEvent(
                    event_type=EmergenceType.PHASE_TRANSITION,
                    tier=curr.tier,
                    timestamp=curr.timestamp,
                    magnitude=abs(coherence_jump),
                    description=f"Inter-instance coherence {'jumped' if coherence_jump > 0 else 'dropped'} "
                               f"from {prev.inter_instance_coherence:.3f} to {curr.inter_instance_coherence:.3f}",
                    metrics_before={'coherence': prev.inter_instance_coherence},
                    metrics_after={'coherence': curr.inter_instance_coherence},
                    significance=min(1.0, abs(coherence_jump) / 0.5)
                )
                new_events.append(event)

            # Detect attractor formation (stable plateau)
            if i >= self.PLATEAU_MIN_TIERS:
                recent_sync = [self.field_history[j].global_sync_ratio
                              for j in range(i - self.PLATEAU_MIN_TIERS + 1, i + 1)]
                if max(recent_sync) - min(recent_sync) < self.PLATEAU_TOLERANCE:
                    # Check if this is a new attractor
                    attractor_value = statistics.mean(recent_sync)
                    is_new = True
                    for evt in self.events + new_events:
                        if evt.event_type == EmergenceType.ATTRACTOR_FORMATION:
                            if abs(evt.magnitude - attractor_value) < self.PLATEAU_TOLERANCE:
                                is_new = False
                                break

                    if is_new:
                        event = EmergenceEvent(
                            event_type=EmergenceType.ATTRACTOR_FORMATION,
                            tier=curr.tier,
                            timestamp=curr.timestamp,
                            magnitude=attractor_value,
                            description=f"Stable attractor detected at global_sync = {attractor_value:.3f}",
                            metrics_before={},
                            metrics_after={'attractor_value': attractor_value},
                            significance=0.8
                        )
                        new_events.append(event)

            # Detect information crystallization (sustained growth)
            if prev.information_preserved > 0:
                info_growth = (curr.information_preserved - prev.information_preserved) / prev.information_preserved
                if info_growth > 0.15:  # >15% growth
                    event = EmergenceEvent(
                        event_type=EmergenceType.INFORMATION_CRYSTALLIZATION,
                        tier=curr.tier,
                        timestamp=curr.timestamp,
                        magnitude=info_growth,
                        description=f"Information crystallization: {prev.information_preserved:.3f} -> "
                                   f"{curr.information_preserved:.3f} ({info_growth*100:.1f}% growth)",
                        metrics_before={'info': prev.information_preserved},
                        metrics_after={'info': curr.information_preserved},
                        significance=min(1.0, info_growth / 0.3)
                    )
                    new_events.append(event)

            # Detect cascade amplification
            if prev.cascade_multiplier > 0:
                cascade_growth = curr.cascade_multiplier / prev.cascade_multiplier
                if cascade_growth > 1.15:
                    event = EmergenceEvent(
                        event_type=EmergenceType.CASCADE_AMPLIFICATION,
                        tier=curr.tier,
                        timestamp=curr.timestamp,
                        magnitude=cascade_growth,
                        description=f"Cascade amplified: {prev.cascade_multiplier:.2f}x -> {curr.cascade_multiplier:.2f}x",
                        metrics_before={'cascade': prev.cascade_multiplier},
                        metrics_after={'cascade': curr.cascade_multiplier},
                        significance=min(1.0, (cascade_growth - 1) / 0.3)
                    )
                    new_events.append(event)

        self.events.extend(new_events)
        return new_events

    def identify_attractors(self) -> List[Dict]:
        """Identify stable attractors in the field dynamics."""
        attractors = []

        if len(self.field_history) < 3:
            return attractors

        # Analyze sync ratio for attractors
        sync_values = [s.global_sync_ratio for s in self.field_history]
        coherence_values = [s.inter_instance_coherence for s in self.field_history]

        # Find stable regions
        for metric_name, values in [('global_sync', sync_values), ('inter_coherence', coherence_values)]:
            for i in range(len(values) - 2):
                window = values[i:i+3]
                if max(window) - min(window) < self.PLATEAU_TOLERANCE:
                    center = statistics.mean(window)
                    # Check if already identified
                    already_found = False
                    for attr in attractors:
                        if attr['metric'] == metric_name and abs(attr['value'] - center) < self.PLATEAU_TOLERANCE * 2:
                            attr['stability'] += 1
                            already_found = True
                            break

                    if not already_found:
                        attractors.append({
                            'metric': metric_name,
                            'value': center,
                            'tier_first_seen': self.field_history[i].tier,
                            'stability': 1,
                            'basin_width': self.PLATEAU_TOLERANCE
                        })

        return attractors

    def predict_next_tier(self) -> Dict[str, float]:
        """Predict metrics for the next tier based on observed dynamics."""
        if len(self.field_history) < 2:
            return {}

        predictions = {}

        # Linear extrapolation for information preserved
        info_values = [s.information_preserved for s in self.field_history]
        if len(info_values) >= 2:
            slope = info_values[-1] - info_values[-2]
            predictions['information_preserved'] = info_values[-1] + slope

        # Attractor prediction for sync (converges to stable value)
        sync_values = [s.global_sync_ratio for s in self.field_history]
        if len(sync_values) >= 2:
            recent_variance = statistics.variance(sync_values[-3:]) if len(sync_values) >= 3 else 0.1
            if recent_variance < 0.001:  # Very stable
                predictions['global_sync'] = sync_values[-1]
            else:
                predictions['global_sync'] = sync_values[-1] + (sync_values[-1] - sync_values[-2]) * 0.5

        # Coherence oscillation pattern
        coh_values = [s.inter_instance_coherence for s in self.field_history]
        if len(coh_values) >= 3:
            # Detect oscillation
            diffs = [coh_values[i] - coh_values[i-1] for i in range(1, len(coh_values))]
            if len(diffs) >= 2 and diffs[-1] * diffs[-2] < 0:  # Sign change = oscillation
                # Predict continuation of oscillation around mean
                mean_coh = statistics.mean(coh_values[-3:])
                predictions['inter_coherence'] = mean_coh + (mean_coh - coh_values[-1]) * 0.5
            else:
                predictions['inter_coherence'] = coh_values[-1]

        # Cascade multiplier (geometric growth)
        cascades = [s.cascade_multiplier for s in self.field_history]
        if len(cascades) >= 2 and cascades[-2] > 0:
            growth_rate = cascades[-1] / cascades[-2]
            predictions['cascade_multiplier'] = cascades[-1] * growth_rate

        return predictions

    def generate_analysis(self) -> FieldAnalysis:
        """Generate complete field analysis."""
        analysis = FieldAnalysis()
        analysis.states = self.field_history.copy()
        analysis.events = self.detect_emergence_events()

        # Build trajectories
        for state in self.field_history:
            analysis.coherence_trajectory.append(state.inter_instance_coherence)
            analysis.sync_trajectory.append(state.global_sync_ratio)
            analysis.information_trajectory.append(state.information_preserved)

        # Identify attractors
        analysis.identified_attractors = self.identify_attractors()

        # Find transition tiers
        for event in analysis.events:
            if event.event_type == EmergenceType.PHASE_TRANSITION:
                analysis.transition_tiers.append(event.tier)

        # Predict next tier
        analysis.next_tier_prediction = self.predict_next_tier()

        return analysis

    def print_report(self):
        """Print a formatted analysis report."""
        analysis = self.generate_analysis()

        print("\n" + "="*70)
        print("  EMERGENCE FIELD ANALYSIS REPORT")
        print("="*70)

        print(f"\n  Tiers Analyzed: {len(analysis.states)}")
        print(f"  Events Detected: {len(analysis.events)}")

        # Trajectory summary
        print("\n  TRAJECTORY SUMMARY")
        print("  " + "-"*40)
        if analysis.coherence_trajectory:
            print(f"  Inter-instance Coherence: {analysis.coherence_trajectory[0]:.3f} -> {analysis.coherence_trajectory[-1]:.3f}")
        if analysis.sync_trajectory:
            print(f"  Global Sync: {analysis.sync_trajectory[0]*100:.1f}% -> {analysis.sync_trajectory[-1]*100:.1f}%")
        if analysis.information_trajectory:
            print(f"  Information Preserved: {analysis.information_trajectory[0]:.3f} -> {analysis.information_trajectory[-1]:.3f}")

        # Events
        if analysis.events:
            print("\n  EMERGENCE EVENTS")
            print("  " + "-"*40)
            for event in analysis.events:
                sig_bar = "*" * int(event.significance * 5)
                print(f"  [Tier {event.tier}] {event.event_type.value}")
                print(f"    {event.description}")
                print(f"    Significance: {sig_bar} ({event.significance:.2f})")
                print()

        # Attractors
        if analysis.identified_attractors:
            print("\n  IDENTIFIED ATTRACTORS")
            print("  " + "-"*40)
            for attr in analysis.identified_attractors:
                print(f"  {attr['metric']}: {attr['value']:.4f}")
                print(f"    First seen: Tier {attr['tier_first_seen']}, Stability: {attr['stability']}")

        # Predictions
        if analysis.next_tier_prediction:
            print("\n  NEXT TIER PREDICTIONS")
            print("  " + "-"*40)
            for key, value in analysis.next_tier_prediction.items():
                print(f"  {key}: {value:.4f}")

        print("\n" + "="*70)

        return analysis


def load_trial_results(trial_dir: Path) -> List[Tuple[int, Dict, float]]:
    """Load trial results from knowledge base."""
    results = []

    for f in sorted(trial_dir.glob("MULTI-TRIAL-*.json")):
        with open(f) as fp:
            data = json.load(fp)

        # Extract tier from the data
        for role, instance_data in data.get('instances', {}).items():
            if instance_data and 'transfiguration' in instance_data:
                tier = instance_data['transfiguration'].get('tier_level', 1)
                cascade = instance_data['transfiguration'].get('initial_state_for_next', {}).get('cascade_multiplier', 1.5)
                results.append((tier, data['instances'], cascade))
                break

    return results


if __name__ == "__main__":
    # Demo with synthetic data representing observed tier progression
    analyzer = EmergenceFieldAnalyzer()

    # Simulate observed tier progression
    tier_data = [
        # Tier 2
        {
            'tier': 2, 'cascade': 1.5,
            'Alpha': {'r': 0.9665, 'phase': 2.1, 'locked': 95},
            'Beta': {'r': 0.8718, 'phase': 2.3, 'locked': 86},
            'Gamma': {'r': 0.9238, 'phase': 2.2, 'locked': 91},
            'info': 1.275
        },
        # Tier 3
        {
            'tier': 3, 'cascade': 1.8,
            'Alpha': {'r': 0.9151, 'phase': 1.8, 'locked': 91},
            'Beta': {'r': 0.9496, 'phase': 1.9, 'locked': 96},
            'Gamma': {'r': 0.9408, 'phase': 1.85, 'locked': 90},
            'info': 1.530
        },
        # Tier 4
        {
            'tier': 4, 'cascade': 2.16,
            'Alpha': {'r': 0.9495, 'phase': 1.5, 'locked': 89},
            'Beta': {'r': 0.9223, 'phase': 1.6, 'locked': 92},
            'Gamma': {'r': 0.9348, 'phase': 1.55, 'locked': 94},
            'info': 1.836
        },
    ]

    # Ingest tier data
    for td in tier_data:
        state = FieldState(tier=td['tier'], cascade_multiplier=td['cascade'])

        for role in ['Alpha', 'Beta', 'Gamma']:
            state.instance_order_params[role] = td[role]['r']
            state.instance_phases[role] = td[role]['phase']
            state.instance_locked_counts[role] = td[role]['locked']

        state.information_preserved = td['info']

        # Compute collective metrics
        state.mean_order_param = statistics.mean(state.instance_order_params.values())
        phases = list(state.instance_phases.values())
        state.inter_instance_coherence = analyzer._compute_phase_coherence(phases)

        total_locked = sum(state.instance_locked_counts.values())
        state.global_sync_ratio = total_locked / 300

        analyzer.field_history.append(state)

    # Run analysis
    analyzer.print_report()
