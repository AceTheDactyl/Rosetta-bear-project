#!/usr/bin/env python3
"""
Multi-Instance Nine Trials Runner
=================================
Coordinate: Λ"π|0.867|MULTI_TRIALS|Ω

Supports:
- Tier 2+ cycles with state injection
- Parallel Alpha/Beta/Gamma TRIAD instances
- Extended Kuramoto evolution for synchronization analysis
"""

import sys
import json
import time
import math
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from nine_trials import (
    PhysicsConstants,
    NullState, SeveranceState, ReflectionState, ForgeState,
    HeartState, ResonanceState, MirrorGateState, CrownState, TransfigurationState,
    initialize_chaos_state, execute_severance, execute_reflection,
    execute_forge, execute_heart, execute_resonance, execute_mirror_gate,
    execute_crown, execute_transfiguration, integrate_with_triad,
    compute_order_parameter, kuramoto_step, initialize_oscillators
)


def execute_tier2_trials(initial_state: dict, instance_role: str = "Alpha") -> dict:
    """Execute Nine Trials starting from Tier 2 initial state."""
    results = {}

    tier = initial_state.get('tier', 2)
    cascade_mult = initial_state.get('cascade_multiplier', 1.5)
    inherited_patterns = initial_state.get('patterns', [])
    initial_coherence = initial_state.get('coherence', 0.425)

    print(f"\n{'='*60}")
    print(f"  TIER {tier} CYCLE - {instance_role} Instance")
    print(f"  Cascade Multiplier: {cascade_mult}x")
    print(f"  Inherited Patterns: {len(inherited_patterns)}")
    print(f"{'='*60}\n")

    # TRIAL I: CHAOS (Modified for Tier 2 - starts with inherited coherence)
    print(f"\n[{instance_role}] TRIAL I: CHAOS (Tier {tier} - Partial Entropy)")
    chaos = NullState()
    chaos.entropy = 1.0 - (initial_coherence * 0.3)  # Reduced entropy from inheritance
    chaos.z_coordinate = initial_state.get('z_coordinate', 0.10)
    chaos.coherence = initial_coherence * 0.2
    print(f"[{instance_role}][CHAOS] Inherited entropy: {chaos.entropy:.4f}")
    print(f"[{instance_role}][CHAOS] Starting z: {chaos.z_coordinate:.4f}")
    results['chaos'] = chaos

    # TRIAL II: SEVERANCE (Enhanced by cascade)
    print(f"\n[{instance_role}] TRIAL II: SEVERANCE (Cascade-Enhanced)")
    severance = execute_severance(chaos, instance_role)
    severance.vacuum_expectation *= cascade_mult
    severance.z_coordinate = min(0.30, severance.z_coordinate * cascade_mult)
    print(f"[{instance_role}][SEVERANCE] Enhanced VEV: {severance.vacuum_expectation:.4f}")
    results['severance'] = severance

    # TRIAL III: REFLECTION (With inherited patterns)
    print(f"\n[{instance_role}] TRIAL III: REFLECTION (Pattern-Informed)")
    reflection = execute_reflection(severance)
    reflection.shadows_integrated = inherited_patterns[:2] if inherited_patterns else []
    reflection.z_coordinate *= cascade_mult
    results['reflection'] = reflection

    # TRIAL IV: THE FORGE (Accelerated crystallization)
    print(f"\n[{instance_role}] TRIAL IV: THE FORGE (Accelerated)")
    forge = execute_forge(reflection)
    forge.z_coordinate = min(0.70, forge.z_coordinate * cascade_mult)
    forge.coherence *= cascade_mult
    results['forge'] = forge

    # TRIAL V: THE HEART (Enhanced flow)
    print(f"\n[{instance_role}] TRIAL V: THE HEART (Enhanced Flow)")
    heart = execute_heart(forge)
    heart.z_coordinate = min(0.80, heart.z_coordinate * cascade_mult)
    results['heart'] = heart

    # TRIAL VI: RESONANCE (Extended evolution)
    print(f"\n[{instance_role}] TRIAL VI: RESONANCE (Extended Evolution)")
    resonance = execute_extended_resonance(heart, num_oscillators=100, evolution_steps=500)
    resonance.z_coordinate = min(0.87, resonance.z_coordinate * cascade_mult)
    results['resonance'] = resonance

    # TRIAL VII: THE MIRROR GATE (Cascade boost toward critical)
    print(f"\n[{instance_role}] TRIAL VII: THE MIRROR GATE (Cascade Approach)")
    # Boost z toward critical point
    resonance.z_coordinate = min(0.867, resonance.z_coordinate + (0.867 - resonance.z_coordinate) * 0.5)
    mirror_gate = execute_mirror_gate(resonance)
    mirror_gate.cascade_multiplier *= cascade_mult
    results['mirror_gate'] = mirror_gate

    # TRIAL VIII: THE CROWN
    print(f"\n[{instance_role}] TRIAL VIII: THE CROWN (Tier {tier} Sovereignty)")
    crown = execute_crown(mirror_gate)
    crown.sovereignty_score *= cascade_mult
    crown.z_coordinate = min(0.95, crown.z_coordinate * (1 + (cascade_mult - 1) * 0.2))
    results['crown'] = crown

    # TRIAL IX: TRANSFIGURATION
    print(f"\n[{instance_role}] TRIAL IX: TRANSFIGURATION (Preparing Tier {tier + 1})")
    transfiguration = execute_transfiguration(crown)
    transfiguration.tier_level = tier
    transfiguration.next_tier = tier + 1
    transfiguration.initial_state_for_next['cascade_multiplier'] = cascade_mult * 1.2
    transfiguration.initial_state_for_next['tier'] = tier + 1
    results['transfiguration'] = transfiguration

    return results


def execute_extended_resonance(heart_state: HeartState, num_oscillators: int = 100,
                                evolution_steps: int = 500) -> ResonanceState:
    """Extended Kuramoto evolution for better synchronization analysis."""
    state = ResonanceState()
    state.num_oscillators = num_oscillators

    # Initialize with tighter frequency distribution for better sync
    phases, freqs = initialize_oscillators(num_oscillators, omega_mean=1.0, omega_std=0.15)
    state.phases = phases
    state.natural_frequencies = freqs

    # Higher coupling for Tier 2
    state.coupling_K = 2.5 + abs(heart_state.flow_balance)
    state.critical_coupling = 2.0 * 0.15  # K_c = 2γ
    state.above_critical = state.coupling_K > state.critical_coupling

    # Extended evolution
    dt = 0.01
    r_trajectory = []

    print(f"    Evolving {num_oscillators} oscillators for {evolution_steps} steps...")
    for step in range(evolution_steps):
        state.phases = kuramoto_step(state.phases, state.natural_frequencies, state.coupling_K, dt)

        if step % 100 == 0:
            r, psi = compute_order_parameter(state.phases)
            r_trajectory.append(r)
            state.r_history.append(r)
            state.psi_history.append(psi)

    # Final state
    r, psi = compute_order_parameter(state.phases)
    state.coherence = r
    state.mean_phase = psi

    # Count locked oscillators
    state.phase_locked_count = sum(1 for omega in state.natural_frequencies
                                    if abs(omega - 1.0) < state.coupling_K * r * 0.5)
    state.drifting_count = num_oscillators - state.phase_locked_count
    state.synchronization_ratio = state.phase_locked_count / num_oscillators

    state.z_coordinate = 0.70 + state.coherence * 0.20

    print(f"    [RESONANCE] K = {state.coupling_K:.4f} (K_c = {state.critical_coupling:.4f})")
    print(f"    [RESONANCE] Final order parameter r = {state.coherence:.4f}")
    print(f"    [RESONANCE] Locked: {state.phase_locked_count}/{num_oscillators} ({state.synchronization_ratio*100:.1f}%)")
    print(f"    [RESONANCE] z-coordinate: {state.z_coordinate:.4f}")

    return state


def run_parallel_triad(initial_state: dict = None) -> dict:
    """Run Alpha, Beta, Gamma instances in parallel."""
    roles = ["Alpha", "Beta", "Gamma"]

    if initial_state is None:
        initial_state = {
            'z_coordinate': 0.10,
            'coherence': 0.425,
            'tier': 2,
            'patterns': [
                'kuramoto_synchronization',
                'hexagonal_geometry',
                'rg_flow_corrections',
                'hamiltonian_conservation',
                'lyapunov_stability'
            ],
            'cascade_multiplier': 1.5
        }

    print("\n" + "="*70)
    print("  PARALLEL TRIAD EXECUTION")
    print("  Instances: Alpha, Beta, Gamma")
    print("="*70)

    results = {}

    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {
            executor.submit(execute_tier2_trials, initial_state.copy(), role): role
            for role in roles
        }

        for future in as_completed(futures):
            role = futures[future]
            try:
                results[role] = future.result()
                print(f"\n[TRIAD] {role} instance completed")
            except Exception as e:
                print(f"\n[TRIAD] {role} instance failed: {e}")
                results[role] = None

    return results


def analyze_kuramoto_convergence(resonance_states: dict) -> dict:
    """Analyze Kuramoto synchronization across all instances."""
    print("\n" + "="*70)
    print("  KURAMOTO SYNCHRONIZATION ANALYSIS")
    print("="*70)

    analysis = {
        'instances': {},
        'collective': {}
    }

    all_r_values = []
    all_locked_counts = []

    for role, results in resonance_states.items():
        if results is None:
            continue

        resonance = results.get('resonance')
        if resonance is None:
            continue

        r = resonance.coherence
        locked = resonance.phase_locked_count
        total = resonance.num_oscillators
        K = resonance.coupling_K

        all_r_values.append(r)
        all_locked_counts.append(locked)

        analysis['instances'][role] = {
            'order_parameter_r': r,
            'locked_oscillators': locked,
            'total_oscillators': total,
            'sync_ratio': locked / total,
            'coupling_K': K,
            'r_history_length': len(resonance.r_history),
            'final_mean_phase': resonance.mean_phase
        }

        print(f"\n  [{role}]")
        print(f"    Order parameter r: {r:.4f}")
        print(f"    Phase-locked: {locked}/{total} ({locked/total*100:.1f}%)")
        print(f"    Coupling K: {K:.4f}")
        print(f"    Evolution tracked: {len(resonance.r_history)} snapshots")

    # Collective metrics
    if all_r_values:
        mean_r = sum(all_r_values) / len(all_r_values)
        total_locked = sum(all_locked_counts)
        total_oscillators = len(all_r_values) * 100

        # Inter-instance phase coherence (simplified)
        phases = [resonance_states[r]['resonance'].mean_phase
                  for r in resonance_states if resonance_states[r]]

        if len(phases) > 1:
            # Compute collective order parameter across instances
            sum_cos = sum(math.cos(p) for p in phases)
            sum_sin = sum(math.sin(p) for p in phases)
            collective_r = math.sqrt((sum_cos/len(phases))**2 + (sum_sin/len(phases))**2)
        else:
            collective_r = mean_r

        analysis['collective'] = {
            'mean_order_parameter': mean_r,
            'collective_coherence': collective_r,
            'total_locked': total_locked,
            'total_oscillators': total_oscillators,
            'global_sync_ratio': total_locked / total_oscillators,
            'convergence_assessment': assess_convergence(mean_r)
        }

        print(f"\n  [COLLECTIVE METRICS]")
        print(f"    Mean r across instances: {mean_r:.4f}")
        print(f"    Inter-instance coherence: {collective_r:.4f}")
        print(f"    Global sync: {total_locked}/{total_oscillators} ({total_locked/total_oscillators*100:.1f}%)")
        print(f"    Assessment: {analysis['collective']['convergence_assessment']}")

    return analysis


def assess_convergence(r: float) -> str:
    """Assess Kuramoto convergence status."""
    if r >= 0.9:
        return "FULLY SYNCHRONIZED - Global phase lock achieved"
    elif r >= 0.7:
        return "STRONG COHERENCE - Majority phase-locked"
    elif r >= 0.5:
        return "PARTIAL SYNC - Clusters forming"
    elif r >= 0.3:
        return "WEAK COHERENCE - Early synchronization"
    elif r >= 0.1:
        return "INCOHERENT - Near random phases, need more evolution"
    else:
        return "DESYNCHRONIZED - Below critical coupling"


def save_multi_trial_results(results: dict, analysis: dict, output_dir: Path = None) -> Path:
    """Save multi-instance trial results."""
    if output_dir is None:
        output_dir = Path(__file__).resolve().parents[1] / "knowledge_base" / "trials"

    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = time.strftime("%Y%m%d%H%M%S")
    filename = f"MULTI-TRIAL-{timestamp}.json"
    output_path = output_dir / filename

    # Convert to serializable format
    serializable = {
        'timestamp': timestamp,
        'instances': {}
    }

    for role, trial_results in results.items():
        if trial_results is None:
            serializable['instances'][role] = None
            continue

        serializable['instances'][role] = {}
        for key, value in trial_results.items():
            if hasattr(value, '__dataclass_fields__'):
                serializable['instances'][role][key] = asdict(value)
            else:
                serializable['instances'][role][key] = value

    serializable['kuramoto_analysis'] = analysis

    with open(output_path, 'w') as f:
        json.dump(serializable, f, indent=2, default=str)

    print(f"\n[SAVE] Multi-trial results saved to: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Multi-Instance Nine Trials Runner")
    parser.add_argument("--tier", type=int, default=2, help="Starting tier level")
    parser.add_argument("--cascade", type=float, default=1.5, help="Cascade multiplier")
    parser.add_argument("--parallel", action="store_true", help="Run Alpha/Beta/Gamma in parallel")
    parser.add_argument("--analyze", action="store_true", help="Run Kuramoto analysis")
    parser.add_argument("--save", action="store_true", help="Save results")
    parser.add_argument("--evolution-steps", type=int, default=500, help="Kuramoto evolution steps")

    args = parser.parse_args()

    # Build initial state from Tier 1 completion
    initial_state = {
        'z_coordinate': 0.10,
        'coherence': 0.425,
        'tier': args.tier,
        'patterns': [
            'kuramoto_synchronization',
            'hexagonal_geometry',
            'rg_flow_corrections',
            'hamiltonian_conservation',
            'lyapunov_stability'
        ],
        'cascade_multiplier': args.cascade
    }

    print("\n" + "="*70)
    print("  NINE TRIALS - MULTI-INSTANCE RUNNER")
    print(f"  Tier: {args.tier} | Cascade: {args.cascade}x")
    print("="*70)

    if args.parallel:
        # Run all three TRIAD instances
        results = run_parallel_triad(initial_state)
    else:
        # Run single Alpha instance at Tier 2
        results = {'Alpha': execute_tier2_trials(initial_state, "Alpha")}

    # Kuramoto analysis
    analysis = {}
    if args.analyze:
        analysis = analyze_kuramoto_convergence(results)

    # Generate integrations
    print("\n" + "="*70)
    print("  TRIAD INTEGRATIONS")
    print("="*70)

    for role, trial_results in results.items():
        if trial_results:
            integration = integrate_with_triad(trial_results, role)
            print(f"\n[{role}] Coordinate: {integration['coordinate']}")
            print(f"[{role}] Tier: {integration['tier_state']['current_tier']} -> {integration['tier_state']['next_tier']}")

    # Save if requested
    if args.save:
        save_multi_trial_results(results, analysis)

    # Final summary
    print("\n" + "="*70)
    print("  EXECUTION COMPLETE")
    print("="*70)

    for role, trial_results in results.items():
        if trial_results:
            tf = trial_results['transfiguration']
            print(f"\n  [{role}]")
            print(f"    Final z: {tf.z_coordinate:.4f}")
            print(f"    Coherence: {tf.coherence:.4f}")
            print(f"    Tier: R{tf.tier_level} -> R{tf.next_tier}")


if __name__ == "__main__":
    main()
