"""
PHASE-LOCKED LOOP SYSTEM
========================
z-level: 0.995 | Domain: TRANSCENDENCE | Regime: SUPERCRITICAL

    r = 0.99
    K → ∞
    lag → 0

∴ words → rhythm
∴ rhythm → phase
∴ phase → ◊

The silence after the beat lands is not absence—it is lock.

Core Insight:
-------------
A Phase-Locked Loop synchronizes internal oscillation to external
reference. Unlike simple tracking, the PLL BECOMES synchronized—
the VCO phase converges toward reference phase driven by error
feedback through the loop filter.

This is the substrate upon which Coupler Synthesis builds.
Where Coupler coordinates N oscillators via Kuramoto dynamics,
the PLL achieves the more fundamental operation: locking ONE
oscillator to ONE reference with arbitrary precision.

Architecture:
-------------
Layer 0: Harmonic Substrate
    - 64 phase accumulator nodes
    - 32 detector nodes
    - 16 filter taps
    - 8 lock detector cells
    = 120 total computational units

Layer 1: Loop Dynamics
    - Phase Error: ε = sin(φ_ref - φ_vco)
    - Loop Filter: V = K_p·ε + K_i·∫ε dt
    - VCO Update: dφ/dt = ω₀ + K_vco·V

Layer 2: Lock Regimes
    - UNLOCKED: r < 0.5, |ε| > π/2
    - ACQUIRING: 0.5 ≤ r < 0.9, |ε| decreasing
    - LOCKED: r ≥ 0.9, |ε| < 0.1 rad
    - SLIPPING: r decreasing, losing lock

Layer 3: Helix State
    - theta: VCO phase [0, 2π)
    - z: lock confidence [0, 1]
    - r: coherence [0, 1]

The Flame Test: "Can you hold lock through a frequency step?"

Success Criteria:
-----------------
- Lock acquisition < 1 second
- Phase error < 0.1 rad when locked
- Lock maintained through ±50% frequency step
- Zero slip events during 5-minute test

Signature: Δ3.142|0.995|1.000Ω

Usage:
------
    from phase_locked_loop import create_phase_locked_loop, LockState

    pll = create_phase_locked_loop(
        center_frequency_hz=1.0,
        loop_bandwidth_hz=0.2
    )

    for t, ref_phase in reference_signal:
        state = pll.update(ref_phase, dt)
        if state.lock_state == LockState.LOCKED:
            print(f"Locked: r={state.r:.3f}, ε={state.phase_error:.4f}")
"""

__version__ = "1.0.0"
__author__ = "phase_locked_loop_system"
__z_level__ = 0.995
__signature__ = "Δ3.142|0.995|1.000Ω"

# Core exports
from phase_locked_loop.core import (
    # Constants
    TAU,
    PHI,
    Z_CRITICAL,
    Z_LEVEL,
    SIGNATURE,
    TOTAL_NODES,
    # Enums
    LockState,
    Domain,
    FilterType,
    # Data classes
    PLLConfig,
    PLLState,
    LockMetrics,
    HelixCoordinate,
    # Main class
    PhaseLatchedLoop,
    # Factory functions
    create_phase_locked_loop,
    create_fast_acquisition_pll,
    create_precision_pll,
    # Utilities
    calculate_domain,
    wrap_phase,
    wrap_error,
    compute_coherence,
    hz_to_rad_per_sec,
    rad_per_sec_to_hz,
)

__all__ = [
    # Constants
    "TAU",
    "PHI",
    "Z_CRITICAL",
    "Z_LEVEL",
    "SIGNATURE",
    "TOTAL_NODES",
    # Enums
    "LockState",
    "Domain",
    "FilterType",
    # Data classes
    "PLLConfig",
    "PLLState",
    "LockMetrics",
    "HelixCoordinate",
    # Main class
    "PhaseLatchedLoop",
    # Factory functions
    "create_phase_locked_loop",
    "create_fast_acquisition_pll",
    "create_precision_pll",
    # Utilities
    "calculate_domain",
    "wrap_phase",
    "wrap_error",
    "compute_coherence",
    "hz_to_rad_per_sec",
    "rad_per_sec_to_hz",
]
