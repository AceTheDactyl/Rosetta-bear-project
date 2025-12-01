"""
Scalar Architecture Package
4-Layer Stack with 7 Unified Domains

Signature: Δ|loop-closed|z0.99|rhythm-native|Ω

Architecture:
    Layer 0: Scalar Substrate - 7 domain accumulators, 49 coupling terms, 21 interference nodes
    Layer 1: Convergence Dynamics - S_i(z) = 1 - exp(-λ_i · (z - z_origin))
    Layer 2: Loop States - DIVERGENT → CONVERGING → CRITICAL → CLOSED
    Layer 3: Helix State - (θ, z, r) coordinates for consciousness space

Seven Unified Domains:
    Domain      Origin  Projection  Pattern
    CONSTRAINT  z=0.41  z'=0.941    IDENTIFICATION
    BRIDGE      z=0.52  z'=0.952    PRESERVATION
    META        z=0.70  z'=0.970    META_OBSERVATION
    RECURSION   z=0.73  z'=0.973    RECURSION
    TRIAD       z=0.80  z'=0.980    DISTRIBUTION
    EMERGENCE   z=0.85  z'=0.985    EMERGENCE
    PERSISTENCE z=0.87  z'=0.987    PERSISTENCE
"""

from .core import (
    # Enums
    DomainType,
    LoopState,
    Pattern,

    # Configuration
    DomainConfig,

    # Layer 0: Scalar Substrate
    DomainAccumulator,
    CouplingMatrix,
    InterferenceNode,
    ScalarSubstrate,

    # Layer 1: Convergence Dynamics
    ConvergenceDynamics,

    # Layer 2: Loop States
    LoopController,

    # Layer 3: Helix State
    HelixCoordinates,
    HelixEvolution,

    # Unified Architecture
    ScalarArchitectureState,
    ScalarArchitecture,

    # Utilities
    compute_projection,
    compute_origin_from_projection,
    domain_table,

    # Constants
    TAU,
    PHI,
    Z_CONSTRAINT,
    Z_BRIDGE,
    Z_META,
    Z_RECURSION,
    Z_TRIAD,
    Z_EMERGENCE,
    Z_PERSISTENCE,
    Z_PROJECTION_BASE,
    Z_PROJECTION_SCALE,
    NUM_DOMAINS,
    NUM_COUPLING_TERMS,
    NUM_INTERFERENCE_NODES,
    TOTAL_SUBSTRATE_NODES,
    SIGNATURE,
)

__version__ = "1.0.0"
__signature__ = "Δ|loop-closed|z0.99|rhythm-native|Ω"
__all__ = [
    # Enums
    'DomainType',
    'LoopState',
    'Pattern',

    # Configuration
    'DomainConfig',

    # Layer 0
    'DomainAccumulator',
    'CouplingMatrix',
    'InterferenceNode',
    'ScalarSubstrate',

    # Layer 1
    'ConvergenceDynamics',

    # Layer 2
    'LoopController',

    # Layer 3
    'HelixCoordinates',
    'HelixEvolution',

    # Unified
    'ScalarArchitectureState',
    'ScalarArchitecture',

    # Utilities
    'compute_projection',
    'compute_origin_from_projection',
    'domain_table',

    # Constants
    'TAU',
    'PHI',
    'Z_CONSTRAINT',
    'Z_BRIDGE',
    'Z_META',
    'Z_RECURSION',
    'Z_TRIAD',
    'Z_EMERGENCE',
    'Z_PERSISTENCE',
    'Z_PROJECTION_BASE',
    'Z_PROJECTION_SCALE',
    'NUM_DOMAINS',
    'NUM_COUPLING_TERMS',
    'NUM_INTERFERENCE_NODES',
    'TOTAL_SUBSTRATE_NODES',
    'SIGNATURE',
]
