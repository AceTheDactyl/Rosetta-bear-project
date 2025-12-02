#!/usr/bin/env python3
"""
Polarity Orchestrator Demonstration
===================================

Demonstrates the dual polarity feedback loop in action:
- Forward arc: Scalar domains -> Kaelhedron cells
- Backward arc: Coherence -> Loop closure -> K-Formation

Run with: python -m fano_polarity.demo
"""

from __future__ import annotations

import math
import time
from typing import Dict, Any

from .orchestrator import PolarityOrchestrator
from .unified_state import UnifiedSystemState, KFormationStatus
from .telemetry import get_telemetry_hub, TelemetrySource, TelemetryLevel

# Constants
PHI_INV = 0.618033988749895


def print_header(title: str) -> None:
    """Print a formatted header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_domain_table(state: UnifiedSystemState) -> None:
    """Print domain status table."""
    print(f"\n{'Domain':<12} {'Saturation':>10} {'Loop State':<12} {'Phase':>8}")
    print("-" * 46)
    for d in state.domains:
        phase_deg = math.degrees(d.phase) % 360
        print(f"{d.name:<12} {d.saturation:>10.3f} {d.loop_state.value:<12} {phase_deg:>7.1f}°")


def print_coherence_status(state: UnifiedSystemState) -> None:
    """Print coherence and K-Formation status."""
    print(f"\n{'Metric':<25} {'Value':>15}")
    print("-" * 42)
    print(f"{'Kaelhedron Coherence (η)':<25} {state.kaelhedron_coherence:>15.4f}")
    print(f"{'Coherence Threshold (φ⁻¹)':<25} {PHI_INV:>15.4f}")
    print(f"{'Luminahedron Divergence':<25} {state.luminahedron_divergence:>15.4f}")
    print(f"{'Polaric Balance':<25} {state.polaric_balance:>15.4f}")
    print(f"{'Composite Coherence':<25} {state.composite_coherence:>15.4f}")
    print(f"{'K-Formation Status':<25} {state.k_formation_status.value:>15}")
    print(f"{'K-Formation Progress':<25} {state.k_formation_progress:>15.2%}")


def print_loop_counts(state: UnifiedSystemState) -> None:
    """Print loop state counts."""
    print(f"\n{'Loop States':<20}")
    print("-" * 30)
    print(f"  CLOSED:     {state.loops_closed}/7")
    print(f"  CRITICAL:   {state.loops_critical}/7")
    print(f"  CONVERGING: {state.loops_converging}/7")
    print(f"  DIVERGENT:  {state.loops_divergent}/7")
    print(f"  Charge (Q): {state.charge}")


def print_polarity_status(state: UnifiedSystemState) -> None:
    """Print polarity loop status."""
    if state.polarity:
        print(f"\n{'Polarity Loop':<25}")
        print("-" * 40)
        print(f"  Phase:           {state.polarity.phase.value}")
        print(f"  Forward Points:  {state.polarity.forward_points}")
        print(f"  Forward Line:    {state.polarity.forward_line}")
        print(f"  Gate Remaining:  {state.polarity.gate_remaining:.3f}s")
        print(f"  Coherence Point: {state.polarity.coherence_point}")


def demo_evolution() -> None:
    """Demonstrate system evolution from z=0.41 to z=0.99."""
    print_header("POLARITY ORCHESTRATOR - EVOLUTION DEMO")

    print("\nInitializing orchestrator at z=0.41...")
    orch = PolarityOrchestrator(initial_z=0.41)
    orch.set_topological_charge(1)

    # Evolution checkpoints
    checkpoints = [0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]

    print("\n" + "-" * 70)
    print(f"{'z-level':>8} {'η':>8} {'Closed':>8} {'K-Status':>15} {'Signature'}")
    print("-" * 70)

    for z_target in checkpoints:
        # Evolve to target z-level
        while orch.z_level < z_target:
            orch.set_z_level(min(z_target, orch.z_level + 0.01))
            state = orch.step(dt=0.01)

        # Print checkpoint summary
        print(f"{state.kappa:>8.2f} {state.kaelhedron_coherence:>8.3f} "
              f"{state.loops_closed:>8}/7 {state.k_formation_status.value:>15} "
              f"{state.signature}")

    # Final detailed state
    print_header("FINAL STATE DETAILS")
    print_domain_table(state)
    print_coherence_status(state)
    print_loop_counts(state)

    return orch, state


def demo_polarity_injection(orch: PolarityOrchestrator) -> None:
    """Demonstrate polarity injection and release."""
    print_header("POLARITY INJECTION DEMO")

    print("\n§1 Forward Polarity: inject(1, 2)")
    print("   Points 1 and 2 define a unique Fano line...")

    result = orch.inject_polarity(1, 2)
    print(f"   → Line: {result['line']}")
    print(f"   → Phase: {result['phase']}")

    # Get state to show polarity status
    state = orch.step(0.01)
    print_polarity_status(state)

    print("\n§2 Waiting for phase delay...")
    time.sleep(0.3)  # Wait for gate delay

    print("\n§3 Backward Polarity: release((1,2,3), (1,4,5))")
    print("   Lines (1,2,3) and (1,4,5) intersect at point 1...")

    result = orch.release_polarity((1, 2, 3), (1, 4, 5))
    print(f"   → Coherence: {result['coherence']}")
    print(f"   → Point: {result['point']}")
    print(f"   → Phase: {result['phase']}")

    state = orch.step(0.01)
    print_polarity_status(state)

    print("\n§4 Post-Coherence State")
    print_coherence_status(state)


def demo_telemetry() -> None:
    """Demonstrate telemetry hub integration."""
    print_header("TELEMETRY HUB DEMO")

    hub = get_telemetry_hub()
    events_captured = []

    def capture_event(event):
        events_captured.append(event)

    hub.subscribe_events(capture_event)
    hub.set_level_threshold(TelemetryLevel.INFO)

    print("\nCreating orchestrator with telemetry...")
    orch = PolarityOrchestrator(initial_z=0.70)

    # Run a few steps
    for _ in range(10):
        orch.set_z_level(orch.z_level + 0.02)
        orch.step(0.01)

    # Inject polarity to generate events
    orch.inject_polarity(3, 5)
    time.sleep(0.3)
    orch.release_polarity((3, 5, 6), (2, 5, 7))

    print(f"\nCaptured {len(events_captured)} telemetry events")
    print(f"\nHub Statistics:")
    stats = hub.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # Show recent events
    recent = hub.get_recent_events(count=5)
    print(f"\nRecent Events (last 5):")
    for event in recent:
        print(f"  [{event.level.value:>8}] {event.source.value}: {event.event_type}")


def demo_cell_activations(orch: PolarityOrchestrator) -> None:
    """Demonstrate cell activation patterns."""
    print_header("CELL ACTIVATION PATTERNS")

    state = orch.step(0.01)

    # Group cells by seal
    seal_symbols = {1: "Ω", 2: "Δ", 3: "Τ", 4: "Ψ", 5: "Σ", 6: "Ξ", 7: "Κ"}
    face_symbols = {0: "Λ", 1: "Β", 2: "Ν"}

    print(f"\n{'Seal':<6} {'LOGOS (Λ)':>12} {'BIOS (Β)':>12} {'NOUS (Ν)':>12} {'Mean':>10}")
    print("-" * 56)

    for seal in range(1, 8):
        cells = [c for c in state.cells if c.seal_index == seal]
        activations = [c.activation for c in sorted(cells, key=lambda x: x.face_index)]
        mean = sum(activations) / len(activations) if activations else 0

        print(f"{seal_symbols[seal]:<6} {activations[0]:>12.3f} {activations[1]:>12.3f} "
              f"{activations[2]:>12.3f} {mean:>10.3f}")

    # Show total activation
    total = sum(c.activation for c in state.cells)
    print("-" * 56)
    print(f"{'Total':<6} {'':<12} {'':<12} {'':<12} {total:>10.3f}")


def main():
    """Run all demonstrations."""
    print("\n" + "=" * 70)
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║           FANO POLARITY ORCHESTRATOR DEMONSTRATION               ║")
    print("║     Dual Polarity Feedback: Forward Arc ↔ Backward Arc          ║")
    print("╚══════════════════════════════════════════════════════════════════╝")

    # Demo 1: Evolution
    orch, state = demo_evolution()

    # Demo 2: Cell activations
    demo_cell_activations(orch)

    # Demo 3: Polarity injection
    demo_polarity_injection(orch)

    # Demo 4: Telemetry
    demo_telemetry()

    print_header("DEMONSTRATION COMPLETE")
    print(f"\nFinal Signature: {state.signature}")
    print(f"Coherent: {state.is_coherent}")
    print("\nΔ|polarity-unified|rhythm-native|Ω")


if __name__ == "__main__":
    main()
