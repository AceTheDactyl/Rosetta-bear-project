#!/usr/bin/env python3
"""
Cascade Dynamics Optimizer
==========================
Coordinate: Λ"π|0.867|CASCADE_OPTIMIZER|Ω

Optimizes cascade parameters based on observed emergence dynamics:
- Adaptive cascade multiplier adjustment
- Coupling strength optimization
- Evolution step tuning
- Critical point approach calibration

Implements feedback control for Nine Trials tier progression.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum


class OptimizationTarget(Enum):
    """Optimization objectives."""
    MAXIMIZE_COHERENCE = "max_coherence"
    MINIMIZE_TIME = "min_time"
    BALANCE = "balance"
    CRITICAL_APPROACH = "critical_approach"


@dataclass
class TierMetrics:
    """Metrics collected from a tier execution."""

    tier: int
    cascade_multiplier: float
    coupling_K: float
    evolution_steps: int

    # Outcomes
    mean_r: float
    inter_instance_r: float
    global_sync: float
    info_preserved: float

    # Derived
    efficiency: float = 0.0  # sync per evolution step
    coherence_gain: float = 0.0  # compared to previous tier


@dataclass
class OptimizationConfig:
    """Configuration for cascade optimization."""

    # Cascade bounds
    min_cascade: float = 1.0
    max_cascade: float = 5.0
    cascade_step: float = 0.1

    # Coupling bounds
    min_K: float = 1.5
    max_K: float = 4.0
    K_step: float = 0.25

    # Evolution bounds
    min_steps: int = 100
    max_steps: int = 2000
    steps_increment: int = 100

    # Target thresholds
    target_coherence: float = 0.85
    target_sync: float = 0.92
    critical_z: float = 0.867

    # Learning rate for adjustments
    learning_rate: float = 0.1


@dataclass
class OptimizationResult:
    """Result of optimization calculation."""

    recommended_cascade: float
    recommended_K: float
    recommended_steps: int

    rationale: str
    confidence: float  # 0-1
    predicted_outcomes: Dict[str, float] = field(default_factory=dict)


class CascadeDynamicsOptimizer:
    """Optimizes cascade parameters using observed dynamics."""

    def __init__(self, config: OptimizationConfig = None):
        self.config = config or OptimizationConfig()
        self.history: List[TierMetrics] = []
        self.current_params = {
            'cascade': 1.5,
            'K': 2.5,
            'steps': 500
        }

    def record_tier(
        self,
        tier: int,
        cascade: float,
        K: float,
        steps: int,
        mean_r: float,
        inter_r: float,
        global_sync: float,
        info: float
    ) -> TierMetrics:
        """Record tier execution metrics."""

        metrics = TierMetrics(
            tier=tier,
            cascade_multiplier=cascade,
            coupling_K=K,
            evolution_steps=steps,
            mean_r=mean_r,
            inter_instance_r=inter_r,
            global_sync=global_sync,
            info_preserved=info
        )

        # Compute efficiency
        if steps > 0:
            metrics.efficiency = global_sync / steps * 1000

        # Compute gain from previous
        if self.history:
            prev = self.history[-1]
            metrics.coherence_gain = inter_r - prev.inter_instance_r

        self.history.append(metrics)
        self.current_params = {'cascade': cascade, 'K': K, 'steps': steps}

        return metrics

    def optimize(self, target: OptimizationTarget = OptimizationTarget.BALANCE) -> OptimizationResult:
        """Compute optimized parameters for next tier."""

        if not self.history:
            # Default recommendations
            return OptimizationResult(
                recommended_cascade=1.5,
                recommended_K=2.5,
                recommended_steps=500,
                rationale="Initial tier - using default parameters",
                confidence=0.5
            )

        current = self.history[-1]

        # Analyze trends
        cascade_effectiveness = self._analyze_cascade_effectiveness()
        K_effectiveness = self._analyze_coupling_effectiveness()
        steps_effectiveness = self._analyze_steps_effectiveness()

        # Compute recommendations based on target
        if target == OptimizationTarget.MAXIMIZE_COHERENCE:
            result = self._optimize_for_coherence(cascade_effectiveness, K_effectiveness, steps_effectiveness)
        elif target == OptimizationTarget.MINIMIZE_TIME:
            result = self._optimize_for_time(cascade_effectiveness, K_effectiveness, steps_effectiveness)
        elif target == OptimizationTarget.CRITICAL_APPROACH:
            result = self._optimize_for_critical(cascade_effectiveness, K_effectiveness, steps_effectiveness)
        else:  # BALANCE
            result = self._optimize_balanced(cascade_effectiveness, K_effectiveness, steps_effectiveness)

        return result

    def _analyze_cascade_effectiveness(self) -> Dict:
        """Analyze how cascade multiplier affects outcomes."""
        if len(self.history) < 2:
            return {'trend': 'unknown', 'optimal_range': (1.5, 2.5)}

        # Correlate cascade with coherence gain
        cascades = [m.cascade_multiplier for m in self.history]
        gains = [m.coherence_gain for m in self.history[1:]] + [0]

        # Simple linear regression
        if len(cascades) >= 2:
            mean_c = sum(cascades) / len(cascades)
            mean_g = sum(gains) / len(gains)

            numerator = sum((c - mean_c) * (g - mean_g) for c, g in zip(cascades, gains))
            denominator = sum((c - mean_c)**2 for c in cascades)

            slope = numerator / denominator if denominator > 0 else 0

            if slope > 0.05:
                trend = 'increasing_helps'
            elif slope < -0.05:
                trend = 'decreasing_helps'
            else:
                trend = 'neutral'
        else:
            trend = 'unknown'
            slope = 0

        # Find best performing cascade
        best_idx = max(range(len(self.history)), key=lambda i: self.history[i].inter_instance_r)
        best_cascade = self.history[best_idx].cascade_multiplier

        return {
            'trend': trend,
            'slope': slope,
            'best_observed': best_cascade,
            'optimal_range': (max(1.0, best_cascade - 0.5), min(5.0, best_cascade + 0.5))
        }

    def _analyze_coupling_effectiveness(self) -> Dict:
        """Analyze coupling strength effectiveness."""
        if len(self.history) < 2:
            return {'trend': 'unknown', 'optimal': 2.5}

        # Find K that maximizes sync
        best_idx = max(range(len(self.history)), key=lambda i: self.history[i].global_sync)
        best_K = self.history[best_idx].coupling_K

        # Check if we're above critical
        current = self.history[-1]
        above_critical = current.inter_instance_r > 0.5

        return {
            'best_observed': best_K,
            'above_critical': above_critical,
            'optimal': best_K if above_critical else best_K + 0.5
        }

    def _analyze_steps_effectiveness(self) -> Dict:
        """Analyze evolution steps effectiveness."""
        if len(self.history) < 2:
            return {'trend': 'unknown', 'optimal': 500}

        # Check efficiency trend
        efficiencies = [m.efficiency for m in self.history]
        recent_eff = efficiencies[-1] if efficiencies else 0

        # Check if more steps helped
        if len(self.history) >= 2:
            prev = self.history[-2]
            current = self.history[-1]

            if current.evolution_steps > prev.evolution_steps:
                if current.global_sync > prev.global_sync + 0.02:
                    trend = 'more_helps'
                else:
                    trend = 'diminishing_returns'
            else:
                trend = 'unknown'
        else:
            trend = 'unknown'

        return {
            'trend': trend,
            'current_efficiency': recent_eff,
            'optimal': 500 if trend == 'diminishing_returns' else 750
        }

    def _optimize_for_coherence(self, cascade_eff, K_eff, steps_eff) -> OptimizationResult:
        """Optimize purely for coherence."""
        current = self.history[-1]

        # More aggressive cascade if it's helping
        if cascade_eff['trend'] == 'increasing_helps':
            new_cascade = min(self.config.max_cascade, current.cascade_multiplier * 1.2)
        else:
            new_cascade = cascade_eff['best_observed']

        # Higher K for stronger sync
        new_K = min(self.config.max_K, K_eff['optimal'] + 0.25)

        # More steps for better convergence
        new_steps = min(self.config.max_steps, current.evolution_steps + 250)

        predicted = {
            'inter_coherence': min(0.95, current.inter_instance_r + 0.05),
            'global_sync': min(0.98, current.global_sync + 0.02)
        }

        return OptimizationResult(
            recommended_cascade=new_cascade,
            recommended_K=new_K,
            recommended_steps=new_steps,
            rationale="Maximizing coherence: increased cascade, K, and evolution time",
            confidence=0.7,
            predicted_outcomes=predicted
        )

    def _optimize_for_time(self, cascade_eff, K_eff, steps_eff) -> OptimizationResult:
        """Optimize for minimal evolution time."""
        current = self.history[-1]

        # High cascade to accelerate
        new_cascade = min(self.config.max_cascade, current.cascade_multiplier * 1.3)

        # High K to speed sync
        new_K = min(self.config.max_K, K_eff['optimal'] + 0.5)

        # Minimal steps
        new_steps = max(self.config.min_steps, current.evolution_steps - 100)

        predicted = {
            'inter_coherence': current.inter_instance_r * 0.95,  # Might be slightly lower
            'global_sync': current.global_sync * 0.98
        }

        return OptimizationResult(
            recommended_cascade=new_cascade,
            recommended_K=new_K,
            recommended_steps=new_steps,
            rationale="Minimizing time: high cascade/K, reduced steps",
            confidence=0.6,
            predicted_outcomes=predicted
        )

    def _optimize_for_critical(self, cascade_eff, K_eff, steps_eff) -> OptimizationResult:
        """Optimize for approaching critical point (z=0.867)."""
        current = self.history[-1]

        # Moderate cascade for controlled approach
        new_cascade = cascade_eff['best_observed']

        # K near critical coupling
        new_K = 2.0  # Exactly at K_c for maximum fluctuations

        # Extended steps for precision
        new_steps = min(self.config.max_steps, current.evolution_steps + 100)

        predicted = {
            'z_coordinate': 0.867,
            'at_critical': True,
            'cascade_multiplier_at_lens': new_cascade * 1.5
        }

        return OptimizationResult(
            recommended_cascade=new_cascade,
            recommended_K=new_K,
            recommended_steps=new_steps,
            rationale="Critical approach: controlled parameters near THE LENS",
            confidence=0.75,
            predicted_outcomes=predicted
        )

    def _optimize_balanced(self, cascade_eff, K_eff, steps_eff) -> OptimizationResult:
        """Balanced optimization."""
        current = self.history[-1]

        # Geometric growth in cascade
        growth_factor = 1.2
        new_cascade = min(self.config.max_cascade, current.cascade_multiplier * growth_factor)

        # Stable K
        new_K = K_eff['optimal']

        # Adaptive steps
        if current.global_sync < 0.90:
            new_steps = min(self.config.max_steps, current.evolution_steps + 100)
        elif current.global_sync > 0.95:
            new_steps = max(self.config.min_steps, current.evolution_steps - 50)
        else:
            new_steps = current.evolution_steps

        # Predict outcomes
        cascade_boost = (new_cascade / current.cascade_multiplier - 1) * 0.1
        predicted = {
            'inter_coherence': min(0.95, current.inter_instance_r + cascade_boost),
            'global_sync': min(0.98, current.global_sync + cascade_boost * 0.5),
            'info_preserved': current.info_preserved * growth_factor
        }

        return OptimizationResult(
            recommended_cascade=new_cascade,
            recommended_K=new_K,
            recommended_steps=new_steps,
            rationale=f"Balanced: cascade {current.cascade_multiplier:.2f}->{new_cascade:.2f}, "
                     f"steps={new_steps}",
            confidence=0.8,
            predicted_outcomes=predicted
        )

    def get_parameter_history(self) -> Dict[str, List]:
        """Get parameter history for analysis."""
        return {
            'tiers': [m.tier for m in self.history],
            'cascades': [m.cascade_multiplier for m in self.history],
            'coupling_K': [m.coupling_K for m in self.history],
            'evolution_steps': [m.evolution_steps for m in self.history],
            'inter_coherence': [m.inter_instance_r for m in self.history],
            'global_sync': [m.global_sync for m in self.history],
            'info_preserved': [m.info_preserved for m in self.history]
        }

    def print_optimization_report(self, target: OptimizationTarget = OptimizationTarget.BALANCE):
        """Print optimization report."""
        result = self.optimize(target)
        history = self.get_parameter_history()

        print("\n" + "="*60)
        print("  CASCADE DYNAMICS OPTIMIZER")
        print("="*60)

        print(f"\n  Optimization Target: {target.value}")
        print(f"  Tiers Analyzed: {len(self.history)}")

        # Current state
        if self.history:
            current = self.history[-1]
            print(f"\n  CURRENT STATE (Tier {current.tier})")
            print("  " + "-"*40)
            print(f"  Cascade: {current.cascade_multiplier:.2f}x")
            print(f"  Coupling K: {current.coupling_K:.2f}")
            print(f"  Evolution steps: {current.evolution_steps}")
            print(f"  Inter-coherence: {current.inter_instance_r:.4f}")
            print(f"  Global sync: {current.global_sync*100:.1f}%")

        # Recommendations
        print(f"\n  RECOMMENDATIONS")
        print("  " + "-"*40)
        print(f"  Cascade: {result.recommended_cascade:.2f}x")
        print(f"  Coupling K: {result.recommended_K:.2f}")
        print(f"  Evolution steps: {result.recommended_steps}")
        print(f"\n  Rationale: {result.rationale}")
        print(f"  Confidence: {result.confidence*100:.0f}%")

        # Predictions
        if result.predicted_outcomes:
            print(f"\n  PREDICTED OUTCOMES")
            print("  " + "-"*40)
            for key, value in result.predicted_outcomes.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.4f}")
                else:
                    print(f"  {key}: {value}")

        print("\n" + "="*60)

        return result


# Global optimizer instance
_optimizer_instance: Optional[CascadeDynamicsOptimizer] = None


def get_optimizer() -> CascadeDynamicsOptimizer:
    """Get global optimizer instance."""
    global _optimizer_instance
    if _optimizer_instance is None:
        _optimizer_instance = CascadeDynamicsOptimizer()
    return _optimizer_instance


if __name__ == "__main__":
    # Demo with observed tier data
    optimizer = CascadeDynamicsOptimizer()

    # Record observed tiers
    tier_data = [
        (2, 1.5, 2.5, 500, 0.9207, 0.3234, 0.907, 1.275),
        (3, 1.8, 2.5, 500, 0.9352, 0.8399, 0.923, 1.530),
        (4, 2.16, 2.5, 500, 0.9355, 0.7800, 0.917, 1.836),
    ]

    for tier, cascade, K, steps, mean_r, inter_r, sync, info in tier_data:
        optimizer.record_tier(tier, cascade, K, steps, mean_r, inter_r, sync, info)

    # Get optimization for next tier
    optimizer.print_optimization_report(OptimizationTarget.BALANCE)
