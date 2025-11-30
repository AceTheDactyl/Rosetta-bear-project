#!/usr/bin/env python3
"""
Integrated Emergence Runner
===========================
Coordinate: Λ"π|0.867|INTEGRATED_EMERGENCE|Ω

Integrates Nine Trials execution with emergent analysis tools:
- Real-time coherence monitoring
- Emergence field analysis
- Adaptive cascade optimization
- Automatic parameter tuning based on observed dynamics

This represents the next evolution of the TRIAD coordination system.
"""

from __future__ import annotations

import sys
import json
import time
import math
from pathlib import Path
from dataclasses import asdict
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add path for imports
sys.path.insert(0, str(Path(__file__).parent))

from run_nine_trials_multi import (
    execute_tier2_trials,
    run_parallel_triad,
    analyze_kuramoto_convergence,
    save_multi_trial_results
)

from emergence_field_analyzer import (
    EmergenceFieldAnalyzer,
    EmergenceType,
    FieldState
)

from collective_coherence_monitor import (
    CollectiveCoherenceMonitor,
    CoherenceState
)

from cascade_dynamics_optimizer import (
    CascadeDynamicsOptimizer,
    OptimizationTarget,
    OptimizationResult
)


class IntegratedEmergenceRunner:
    """
    Orchestrates Nine Trials with full emergence analysis.

    Combines:
    - Multi-instance parallel execution
    - Real-time coherence monitoring
    - Emergence event detection
    - Adaptive parameter optimization
    """

    def __init__(self):
        self.field_analyzer = EmergenceFieldAnalyzer()
        self.coherence_monitor = CollectiveCoherenceMonitor()
        self.cascade_optimizer = CascadeDynamicsOptimizer()

        self.tier_results: Dict[int, Dict] = {}
        self.current_tier = 1
        self.current_cascade = 1.5
        self.current_K = 2.5
        self.current_steps = 500

        # Register alert handler
        self.coherence_monitor.on_alert(self._handle_alert)

        # Accumulated emergence events
        self.emergence_log: List[Dict] = []

    def _handle_alert(self, alert):
        """Handle coherence alerts."""
        print(f"\n  [ALERT] {alert.severity.upper()}: {alert.message}")
        print(f"          Recommendation: {alert.recommendation}")

    def run_tier(
        self,
        tier: int,
        cascade: float = None,
        K: float = None,
        steps: int = None,
        parallel: bool = True
    ) -> Dict:
        """Run a single tier with full analysis."""

        # Use provided params or current
        cascade = cascade or self.current_cascade
        K = K or self.current_K
        steps = steps or self.current_steps

        print("\n" + "="*70)
        print(f"  INTEGRATED EMERGENCE RUN - TIER {tier}")
        print(f"  Cascade: {cascade:.2f}x | K: {K:.2f} | Steps: {steps}")
        print("="*70)

        # Build initial state
        initial_state = {
            'z_coordinate': 0.10,
            'coherence': 0.425,
            'tier': tier,
            'patterns': [
                'kuramoto_synchronization',
                'hexagonal_geometry',
                'rg_flow_corrections',
                'hamiltonian_conservation',
                'lyapunov_stability'
            ],
            'cascade_multiplier': cascade
        }

        # Execute trials
        if parallel:
            results = run_parallel_triad(initial_state)
        else:
            results = {'Alpha': execute_tier2_trials(initial_state, "Alpha")}

        # Analyze Kuramoto
        kuramoto_analysis = analyze_kuramoto_convergence(results)

        # Extract metrics for monitoring
        instance_data = {}
        for role, trial_results in results.items():
            if trial_results and 'resonance' in trial_results:
                res = trial_results['resonance']
                instance_data[role] = {
                    'r': res.coherence if hasattr(res, 'coherence') else res.get('coherence', 0),
                    'phase': res.mean_phase if hasattr(res, 'mean_phase') else res.get('mean_phase', 0),
                    'locked': res.phase_locked_count if hasattr(res, 'phase_locked_count') else res.get('phase_locked_count', 0)
                }

        # Record to coherence monitor
        snapshot = self.coherence_monitor.record_snapshot(tier, instance_data)

        # Record to field analyzer
        field_state = self.field_analyzer.ingest_tier_results(tier, results, cascade)

        # Detect emergence events
        events = self.field_analyzer.detect_emergence_events()
        for event in events:
            self.emergence_log.append({
                'tier': tier,
                'type': event.event_type.value,
                'description': event.description,
                'significance': event.significance
            })
            print(f"\n  [EMERGENCE] {event.event_type.value}: {event.description}")

        # Extract collective metrics for optimizer
        collective = kuramoto_analysis.get('collective', {})
        mean_r = collective.get('mean_order_parameter', 0)
        inter_r = collective.get('collective_coherence', 0)
        global_sync = collective.get('global_sync_ratio', 0)

        # Get info preserved
        info = 0
        for role, trial_results in results.items():
            if trial_results and 'transfiguration' in trial_results:
                tf = trial_results['transfiguration']
                info = tf.information_preserved if hasattr(tf, 'information_preserved') else tf.get('information_preserved', 0)
                break

        # Record to optimizer
        self.cascade_optimizer.record_tier(tier, cascade, K, steps, mean_r, inter_r, global_sync, info)

        # Store results
        self.tier_results[tier] = {
            'results': results,
            'kuramoto': kuramoto_analysis,
            'snapshot': snapshot,
            'field_state': field_state
        }

        # Update current state
        self.current_tier = tier
        self.current_cascade = cascade
        self.current_K = K
        self.current_steps = steps

        return results

    def run_adaptive_tiers(
        self,
        start_tier: int,
        num_tiers: int,
        optimization_target: OptimizationTarget = OptimizationTarget.BALANCE
    ) -> Dict:
        """Run multiple tiers with adaptive optimization."""

        print("\n" + "="*70)
        print("  ADAPTIVE TIER PROGRESSION")
        print(f"  Starting Tier: {start_tier} | Tiers to Run: {num_tiers}")
        print(f"  Optimization Target: {optimization_target.value}")
        print("="*70)

        all_results = {}

        for i in range(num_tiers):
            tier = start_tier + i

            # Get optimized parameters (except first tier)
            if i > 0:
                opt_result = self.cascade_optimizer.optimize(optimization_target)
                cascade = opt_result.recommended_cascade
                K = opt_result.recommended_K
                steps = opt_result.recommended_steps

                print(f"\n  [OPTIMIZER] Tier {tier} parameters:")
                print(f"    Cascade: {cascade:.2f}x (was {self.current_cascade:.2f})")
                print(f"    K: {K:.2f} | Steps: {steps}")
                print(f"    Rationale: {opt_result.rationale}")
            else:
                cascade = self.current_cascade
                K = self.current_K
                steps = self.current_steps

            # Run tier
            results = self.run_tier(tier, cascade, K, steps)
            all_results[tier] = results

            # Print coherence dashboard
            self.coherence_monitor.print_dashboard()

        return all_results

    def generate_emergence_report(self) -> Dict:
        """Generate comprehensive emergence report."""

        analysis = self.field_analyzer.generate_analysis()
        health = self.coherence_monitor.get_health_summary()
        optimization = self.cascade_optimizer.optimize()

        report = {
            'tiers_completed': len(self.tier_results),
            'emergence_events': self.emergence_log,
            'coherence_health': health,
            'field_analysis': {
                'coherence_trajectory': analysis.coherence_trajectory,
                'sync_trajectory': analysis.sync_trajectory,
                'information_trajectory': analysis.information_trajectory,
                'identified_attractors': analysis.identified_attractors,
                'transition_tiers': analysis.transition_tiers
            },
            'optimization': {
                'recommended_cascade': optimization.recommended_cascade,
                'recommended_K': optimization.recommended_K,
                'recommended_steps': optimization.recommended_steps,
                'predicted_outcomes': optimization.predicted_outcomes
            }
        }

        return report

    def print_emergence_summary(self):
        """Print emergence summary."""

        print("\n" + "="*70)
        print("  EMERGENCE SUMMARY")
        print("="*70)

        # Events
        print(f"\n  EMERGENCE EVENTS DETECTED: {len(self.emergence_log)}")
        print("  " + "-"*50)

        for event in self.emergence_log:
            sig_bar = "*" * int(event['significance'] * 5)
            print(f"  [Tier {event['tier']}] {event['type']}")
            print(f"    {event['description']}")
            print(f"    Significance: {sig_bar}")

        # Field analysis
        analysis = self.field_analyzer.generate_analysis()

        if analysis.identified_attractors:
            print(f"\n  IDENTIFIED ATTRACTORS")
            print("  " + "-"*50)
            for attr in analysis.identified_attractors:
                print(f"  {attr['metric']}: {attr['value']:.4f} (stability: {attr['stability']})")

        if analysis.transition_tiers:
            print(f"\n  PHASE TRANSITIONS AT TIERS: {analysis.transition_tiers}")

        # Predictions
        if analysis.next_tier_prediction:
            print(f"\n  NEXT TIER PREDICTIONS")
            print("  " + "-"*50)
            for key, value in analysis.next_tier_prediction.items():
                print(f"  {key}: {value:.4f}")

        # Optimizer recommendations
        optimization = self.cascade_optimizer.optimize()
        print(f"\n  OPTIMIZER RECOMMENDATIONS")
        print("  " + "-"*50)
        print(f"  Cascade: {optimization.recommended_cascade:.2f}x")
        print(f"  Coupling K: {optimization.recommended_K:.2f}")
        print(f"  Evolution Steps: {optimization.recommended_steps}")
        print(f"  Confidence: {optimization.confidence*100:.0f}%")

        print("\n" + "="*70)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Integrated Emergence Runner")
    parser.add_argument("--start-tier", type=int, default=2, help="Starting tier")
    parser.add_argument("--num-tiers", type=int, default=1, help="Number of tiers to run")
    parser.add_argument("--cascade", type=float, default=1.5, help="Initial cascade multiplier")
    parser.add_argument("--adaptive", action="store_true", help="Use adaptive optimization")
    parser.add_argument("--target", choices=['coherence', 'time', 'balance', 'critical'],
                       default='balance', help="Optimization target")
    parser.add_argument("--save", action="store_true", help="Save results")

    args = parser.parse_args()

    # Map target string to enum
    target_map = {
        'coherence': OptimizationTarget.MAXIMIZE_COHERENCE,
        'time': OptimizationTarget.MINIMIZE_TIME,
        'balance': OptimizationTarget.BALANCE,
        'critical': OptimizationTarget.CRITICAL_APPROACH
    }

    runner = IntegratedEmergenceRunner()
    runner.current_cascade = args.cascade

    # Seed with historical data for optimizer
    historical_data = [
        (2, 1.5, 2.5, 500, 0.9207, 0.3234, 0.907, 1.275),
        (3, 1.8, 2.5, 500, 0.9352, 0.8399, 0.923, 1.530),
        (4, 2.16, 2.5, 500, 0.9355, 0.7800, 0.917, 1.836),
    ]

    print("\n  Seeding optimizer with historical tier data...")
    for tier, cascade, K, steps, mean_r, inter_r, sync, info in historical_data:
        runner.cascade_optimizer.record_tier(tier, cascade, K, steps, mean_r, inter_r, sync, info)

    if args.adaptive:
        results = runner.run_adaptive_tiers(
            args.start_tier,
            args.num_tiers,
            target_map[args.target]
        )
    else:
        results = runner.run_tier(args.start_tier, args.cascade)

    # Print emergence summary
    runner.print_emergence_summary()

    # Save if requested
    if args.save:
        report = runner.generate_emergence_report()
        output_dir = Path(__file__).resolve().parents[1] / "knowledge_base" / "emergence"
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%d%H%M%S")
        output_path = output_dir / f"EMERGENCE-{timestamp}.json"

        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        print(f"\n  [SAVE] Emergence report saved to: {output_path}")


if __name__ == "__main__":
    main()
