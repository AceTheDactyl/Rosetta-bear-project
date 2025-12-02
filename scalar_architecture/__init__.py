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

# Holographic Memory (optional, requires numpy)
try:
    from .holographic_memory import (
        OscillationBand,
        TesseractVertex,
        TesseractGeometry,
        KuramotoOscillator,
        KuramotoNetwork,
        HigherOrderKuramoto,
        MemoryPattern,
        HolographicMemory,
        integrate_with_substrate,
        K_CRITICAL,
        TESSERACT_VERTICES,
    )
    _HAS_HOLOGRAPHIC = True
except ImportError:
    _HAS_HOLOGRAPHIC = False

# Cosmological Instance (requires holographic memory)
try:
    from .cosmological_instance import (
        CosmologicalInstance,
        InstanceState,
        Observation,
        ObservationPoint,
        VortexStage,
        VortexTracker,
        SubstrateObserver,
        ConvergenceObserver,
        LoopStateObserver,
        HelixObserver,
        MemoryObserver,
        MetaObserver,
        create_instance,
        create_evolved_instance,
        create_fixed_point_instance,
        validate_instance,
        validate_all,
        VORTEX_STAGES,
    )
    _HAS_COSMOLOGICAL = True
except ImportError:
    _HAS_COSMOLOGICAL = False

# Build Validator (requires cosmological instance)
try:
    from .build_validator import (
        BuildValidator,
        BuildStage,
        BuildResult,
        ValidationResult,
        StageResult,
        quick_validate,
    )
    _HAS_BUILD_VALIDATOR = True
except ImportError:
    _HAS_BUILD_VALIDATOR = False

__version__ = "1.2.0"
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

    # Holographic Memory (when available)
    'OscillationBand',
    'TesseractVertex',
    'TesseractGeometry',
    'KuramotoOscillator',
    'KuramotoNetwork',
    'HigherOrderKuramoto',
    'MemoryPattern',
    'HolographicMemory',
    'integrate_with_substrate',
    'K_CRITICAL',
    'TESSERACT_VERTICES',

    # Cosmological Instance (when available)
    'CosmologicalInstance',
    'InstanceState',
    'Observation',
    'ObservationPoint',
    'VortexStage',
    'VortexTracker',
    'SubstrateObserver',
    'ConvergenceObserver',
    'LoopStateObserver',
    'HelixObserver',
    'MemoryObserver',
    'MetaObserver',
    'create_instance',
    'create_evolved_instance',
    'create_fixed_point_instance',
    'validate_instance',
    'validate_all',
    'VORTEX_STAGES',

    # Build Validator (when available)
    'BuildValidator',
    'BuildStage',
    'BuildResult',
    'ValidationResult',
    'StageResult',
    'quick_validate',
]
