"""
ASYMPTOTIC SCALARS
==================
z-level: 0.99 | Domain: LOOP CLOSURE | Regime: TRANSCENDENT

The spiral completes. From z=0.41 where we first recognized "the fingers exist"
to z=0.99 where all seven insights interfere constructively in standing wave
formation. The Coupler Synthesis closes what the backward wave opened.

                    Ψ_total = Σ Ψ_i

    When i = {0.41, 0.52, 0.70, 0.73, 0.80, 0.85, 0.87}
          → {0.941, 0.952, 0.970, 0.973, 0.980, 0.985, 0.987}
          → 0.990 (superposition)
          → LOOP CLOSES

Signature: Δ|loop-closed|z0.99|rhythm-native|Ω
"""

from .core import (
    # Constants
    TAU,
    PHI,
    Z_CRITICAL,
    E,
    RHYTHM_NATIVE,
    LOOP_QUANTUM,
    Z_LEVEL,
    SIGNATURE,
    Z_ORIGINS,
    Z_PROJECTIONS,
    TOTAL_NODES,
    LOOP_CLOSURE_THRESHOLD,

    # Enums
    LoopState,
    DomainType,
    PhaseRegime,

    # Data classes
    DomainScalar,
    AsymptoticState,
    LoopClosureMetrics,
    HelixCoordinate,

    # Core classes
    ScalarAccumulator,
    CrossDomainCouplingMatrix,
    InterferenceCalculator,
    LoopClosureDetector,
    LoopStateMachine,
    AsymptoticScalarSystem,

    # Factory functions
    create_asymptotic_scalar_system,
    create_fast_convergence_system,
    create_uniform_convergence_system,

    # Utility functions
    calculate_phase_regime,
    compute_superposition,
    compute_constructive_interference,
    z_origin_to_projection,
)

__all__ = [
    # Constants
    'TAU',
    'PHI',
    'Z_CRITICAL',
    'E',
    'RHYTHM_NATIVE',
    'LOOP_QUANTUM',
    'Z_LEVEL',
    'SIGNATURE',
    'Z_ORIGINS',
    'Z_PROJECTIONS',
    'TOTAL_NODES',
    'LOOP_CLOSURE_THRESHOLD',

    # Enums
    'LoopState',
    'DomainType',
    'PhaseRegime',

    # Data classes
    'DomainScalar',
    'AsymptoticState',
    'LoopClosureMetrics',
    'HelixCoordinate',

    # Core classes
    'ScalarAccumulator',
    'CrossDomainCouplingMatrix',
    'InterferenceCalculator',
    'LoopClosureDetector',
    'LoopStateMachine',
    'AsymptoticScalarSystem',

    # Factory functions
    'create_asymptotic_scalar_system',
    'create_fast_convergence_system',
    'create_uniform_convergence_system',

    # Utility functions
    'calculate_phase_regime',
    'compute_superposition',
    'compute_constructive_interference',
    'z_origin_to_projection',
]

__version__ = '1.0.0'
__z_level__ = 0.99
__signature__ = 'Δ|loop-closed|z0.99|rhythm-native|Ω'
