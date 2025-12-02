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

# CET Constants (requires numpy)
try:
    from .cet_constants import (
        # Fundamental constants
        PHI as CET_PHI, E, PI,
        PHI_INVERSE, PHI_SQUARED, LN_PHI, E_PHI, TAU as CET_TAU,
        PHI_PI_RATIO, E_PI_RATIO, PHI_E_RATIO,

        # Physical constants
        ALPHA, ALPHA_INVERSE, PROTON_ELECTRON_RATIO,
        PLANCK_LENGTH, PLANCK_TIME, PLANCK_MASS, C, G, H_BAR,

        # Operators
        CETOperator, OperatorState,

        # Physical domains
        PhysicalDomain, DomainAlignment, DOMAIN_SCALES,

        # Alignment functions
        compute_phi_alignment, compute_pi_alignment, compute_e_alignment,
        compute_alpha_alignment, compute_mass_ratio_alignment,

        # Cosmological structure
        CosmologicalEra, CosmologicalTier, TierConfig, TIER_CONFIGS,
        get_era_tiers, get_tier_by_time,

        # Codephrase
        AttractorCodephrase, mythic_codephrase,
        MYTHIC_ERA_NAMES, MYTHIC_TIER_NAMES, MYTHIC_OPERATOR_NAMES,

        # Utilities
        fundamental_constant_table, era_tier_summary,
    )
    _HAS_CET = True
except ImportError:
    _HAS_CET = False

# Vortex Physics (requires numpy and cet_constants)
try:
    from .vortex_physics import (
        # Constants
        STROUHAL_CYLINDER, VON_KARMAN,
        RE_LAMINAR_STEADY, RE_LAMINAR_VORTEX, RE_TURBULENT,

        # Vortex state
        VortexRegime, VortexPolarity, VortexState,

        # Strouhal
        StrouhalRelation,

        # Oscillators
        VanDerPolOscillator, CoupledWakeOscillator,

        # Kármán street
        KarmanStreet,

        # Recursive vortex
        RecursiveVortex,

        # Integration
        map_z_to_vortex_scale, compute_vortex_reynolds, vortex_regime_from_state,

        # Utilities
        strouhal_table, vortex_physics_summary,
    )
    _HAS_VORTEX = True
except ImportError:
    _HAS_VORTEX = False

# Hierarchy Problem (requires cet_constants)
try:
    from .hierarchy_problem import (
        # Constants
        M_PLANCK, M_WEAK, M_GUT, M_PROTON,
        ALPHA_EM, ALPHA_WEAK, ALPHA_STRONG, ALPHA_GRAVITY,
        HIERARCHY_RATIO, EM_GRAVITY_RATIO,
        E8_DIMENSION, E8_RANK, LORENTZ_DIM, SM_GAUGE_DIM, KAELHEDRON_DIM,

        # φ-Hierarchy
        PhiHierarchy, compute_phi_hierarchy_spectrum,

        # E₈ Volume Factor
        E8Sector, E8VolumeFactor, E8_SECTORS, analyze_e8_dilution,

        # Force Activation
        FundamentalForce, ForceActivation, FORCE_ACTIVATIONS,
        compute_force_ratios_from_recursion,

        # Kaelhedron
        KaelhedronSector,

        # Combined
        HierarchyExplanation,

        # Analysis functions
        compute_higgs_vev_from_phi, analyze_fine_structure,
        hierarchy_summary,
    )
    _HAS_HIERARCHY = True
except ImportError:
    _HAS_HIERARCHY = False

# Polaric Duality (requires hierarchy_problem)
try:
    from .polaric_duality import (
        # Constants
        KAELHEDRON_SYMBOL, KAELHEDRON_NAME, KAELHEDRON_DIMENSIONS,
        LUMINAHEDRON_SYMBOL, LUMINAHEDRON_NAME, LUMINAHEDRON_DIMENSIONS,
        POLARIC_SPAN, HIDDEN_DIMENSIONS, POLARIC_RATIO,

        # Enums
        Polarity, PolaricAspect,

        # Classes
        Kaelhedron, Luminahedron, PolaricSystem, PolaricTransform,

        # Correspondences
        POLARIC_CORRESPONDENCES, get_correspondence,

        # Mythic
        MYTHIC_KAELHEDRON, MYTHIC_LUMINAHEDRON, MYTHIC_UNION,

        # Utilities
        polaric_summary, simulate_polaric_dance,
    )
    _HAS_POLARIC = True
except ImportError:
    _HAS_POLARIC = False

# Mythos Mathematics (requires polaric_duality)
try:
    from .mythos_mathematics import (
        # Enums
        MythosCategory,

        # Core class
        MythosEquation,

        # Equation catalogs
        ERA_MATHEMATICS,
        OPERATOR_MATHEMATICS,
        POLARIC_MATHEMATICS,
        VORTEX_MATHEMATICS,
        RECURSION_MATHEMATICS,
        GEOMETRY_MATHEMATICS,
        COMPLETE_CATALOG,

        # Rosetta Stone
        MythosRosettaStone,
        ROSETTA_STONE,

        # Verification
        verify_mythos_equation,
        verify_all as verify_all_mythos,

        # Utilities
        mythos_mathematics_summary,
        lookup_mythos,
        lookup_number,
    )
    _HAS_MYTHOS = True
except ImportError:
    _HAS_MYTHOS = False

__version__ = "1.6.0"
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

    # CET Constants (when available)
    'CET_PHI',
    'E',
    'PI',
    'PHI_INVERSE',
    'PHI_SQUARED',
    'LN_PHI',
    'E_PHI',
    'CET_TAU',
    'PHI_PI_RATIO',
    'E_PI_RATIO',
    'PHI_E_RATIO',
    'ALPHA',
    'ALPHA_INVERSE',
    'PROTON_ELECTRON_RATIO',
    'PLANCK_LENGTH',
    'PLANCK_TIME',
    'PLANCK_MASS',
    'C',
    'G',
    'H_BAR',
    'CETOperator',
    'OperatorState',
    'PhysicalDomain',
    'DomainAlignment',
    'DOMAIN_SCALES',
    'compute_phi_alignment',
    'compute_pi_alignment',
    'compute_e_alignment',
    'compute_alpha_alignment',
    'compute_mass_ratio_alignment',
    'CosmologicalEra',
    'CosmologicalTier',
    'TierConfig',
    'TIER_CONFIGS',
    'get_era_tiers',
    'get_tier_by_time',
    'AttractorCodephrase',
    'mythic_codephrase',
    'MYTHIC_ERA_NAMES',
    'MYTHIC_TIER_NAMES',
    'MYTHIC_OPERATOR_NAMES',
    'fundamental_constant_table',
    'era_tier_summary',

    # Vortex Physics (when available)
    'STROUHAL_CYLINDER',
    'VON_KARMAN',
    'RE_LAMINAR_STEADY',
    'RE_LAMINAR_VORTEX',
    'RE_TURBULENT',
    'VortexRegime',
    'VortexPolarity',
    'VortexState',
    'StrouhalRelation',
    'VanDerPolOscillator',
    'CoupledWakeOscillator',
    'KarmanStreet',
    'RecursiveVortex',
    'map_z_to_vortex_scale',
    'compute_vortex_reynolds',
    'vortex_regime_from_state',
    'strouhal_table',
    'vortex_physics_summary',

    # Hierarchy Problem (when available)
    'M_PLANCK',
    'M_WEAK',
    'M_GUT',
    'M_PROTON',
    'ALPHA_EM',
    'ALPHA_WEAK',
    'ALPHA_STRONG',
    'ALPHA_GRAVITY',
    'HIERARCHY_RATIO',
    'EM_GRAVITY_RATIO',
    'E8_DIMENSION',
    'E8_RANK',
    'LORENTZ_DIM',
    'SM_GAUGE_DIM',
    'KAELHEDRON_DIM',
    'PhiHierarchy',
    'compute_phi_hierarchy_spectrum',
    'E8Sector',
    'E8VolumeFactor',
    'E8_SECTORS',
    'analyze_e8_dilution',
    'FundamentalForce',
    'ForceActivation',
    'FORCE_ACTIVATIONS',
    'compute_force_ratios_from_recursion',
    'KaelhedronSector',
    'HierarchyExplanation',
    'compute_higgs_vev_from_phi',
    'analyze_fine_structure',
    'hierarchy_summary',

    # Polaric Duality (when available)
    'KAELHEDRON_SYMBOL',
    'KAELHEDRON_NAME',
    'KAELHEDRON_DIMENSIONS',
    'LUMINAHEDRON_SYMBOL',
    'LUMINAHEDRON_NAME',
    'LUMINAHEDRON_DIMENSIONS',
    'POLARIC_SPAN',
    'HIDDEN_DIMENSIONS',
    'POLARIC_RATIO',
    'Polarity',
    'PolaricAspect',
    'Kaelhedron',
    'Luminahedron',
    'PolaricSystem',
    'PolaricTransform',
    'POLARIC_CORRESPONDENCES',
    'get_correspondence',
    'MYTHIC_KAELHEDRON',
    'MYTHIC_LUMINAHEDRON',
    'MYTHIC_UNION',
    'polaric_summary',
    'simulate_polaric_dance',

    # Mythos Mathematics (when available)
    'MythosCategory',
    'MythosEquation',
    'ERA_MATHEMATICS',
    'OPERATOR_MATHEMATICS',
    'POLARIC_MATHEMATICS',
    'VORTEX_MATHEMATICS',
    'RECURSION_MATHEMATICS',
    'GEOMETRY_MATHEMATICS',
    'COMPLETE_CATALOG',
    'MythosRosettaStone',
    'ROSETTA_STONE',
    'verify_mythos_equation',
    'verify_all_mythos',
    'mythos_mathematics_summary',
    'lookup_mythos',
    'lookup_number',
]
