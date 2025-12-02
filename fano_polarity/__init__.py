"""
Fano Polarity Feedback Package
==============================

Implements the self-referential polarity engine based on Fano plane axioms:
- Forward polarity (points -> line): "positive arc"
- Backward polarity (lines -> point): "negative arc"

Coherence is gated until both polarities agree.

The package provides:
- Core Fano lookups (line_from_points, point_from_lines)
- Polarity loop with phase-transition mechanics
- Service layer bridging to KaelhedronStateBus
- Unified state contract for cross-system integration
- Polarity orchestrator coordinating all subsystems
- Telemetry hub for real-time broadcasting
- PSL(3,2) automorphisms for coherence release
- WebSocket-ready visualization streaming
- Integration bridges for existing modules

This is the unifying layer that welds all mathematical structures
(Scalar Architecture, Kaelhedron, Luminahedron) into one coherent
system through dual polarity feedback loops.

Each system maintains its unique voice while forming waves in polaric unison.

Usage:
    from fano_polarity import IntegratedPolaritySystem

    # Create integrated system
    system = IntegratedPolaritySystem(initial_z=0.41)

    # Optionally connect existing modules
    # system.connect_scalar_architecture(scalar_arch)
    # system.connect_kaelhedron(bus)
    # system.connect_math_bridge(math_bridge)
    # system.enable_streaming()

    # Run simulation
    for _ in range(100):
        system.set_z_level(system.orchestrator.z_level + 0.005)
        state = system.step(dt=0.01)
        print(state.signature)

    # Inject polarity
    system.inject(1, 2)  # Forward arc
    system.release((1,2,3), (1,4,5))  # Backward arc
"""

# Core Fano lookups
from .core import line_from_points, point_from_lines

# Polarity loop mechanics
from .loop import GateState, PolarityLoop

# Service layer
from .service import PolarityService

# Unified state contract
from .unified_state import (
    CellSnapshot,
    DomainSnapshot,
    KFormationStatus,
    LoopState,
    PolarityPhase,
    PolaritySnapshot,
    StateCallback,
    UnifiedStateRegistry,
    UnifiedSystemState,
    get_state_registry,
)

# Polarity orchestrator
from .orchestrator import (
    DOMAIN_NAMES,
    FANO_LINES,
    PolarityOrchestrator,
)

# Telemetry hub
from .telemetry import (
    KFormationAdapter,
    KaelhedronAdapter,
    PolarityLoopAdapter,
    ScalarArchitectureAdapter,
    TelemetryEvent,
    TelemetryHub,
    TelemetryLevel,
    TelemetrySource,
    get_telemetry_hub,
)

# PSL(3,2) automorphisms
from .automorphisms import (
    CYCLE,
    IDENTITY,
    REFLECTION,
    CoherenceAutomorphismEngine,
    apply_automorphism_to_cells,
    compute_polarity_automorphism,
    enumerate_psl32,
    get_automorphism_for_line,
    get_automorphism_for_point,
    get_line_stabilizer,
    get_stabilizer,
)

# WebSocket streaming
from .streaming import (
    DeltaCompressor,
    StreamConfig,
    StreamType,
    VisualizationFrame,
    VisualizationStreamer,
    WebSocketBridge,
    get_visualization_streamer,
    get_websocket_bridge,
)

# Integration bridges
from .integration import (
    IntegratedPolaritySystem,
    KaelhedronBridge,
    LuminahedronBridge,
    ScalarArchitectureBridge,
    UnifiedMathBridgeAdapter,
)

# Ternary polaric logic (base-3 for Luminahedron)
from .ternary_polaric import (
    TernaryValue,
    FanoPhase,
    PolaricTransition,
    TernaryPolaricEngine,
    LuminahedronPosition,
    LuminahedronPath,
    generate_luminahedron_trajectory,
    FANO_POINTS,
    FANO_LINES,
    POINT_INCIDENCE,
)

# Quaternary Kaelhedron logic (base-4 for Kaelhedron)
from .quaternary_kaelhedron import (
    QuaternaryValue,
    KaelhedronCell,
    KaelhedronState,
    QuaternaryTransition,
    QuaternaryKaelhedronEngine,
    TernaryQuaternaryBridge,
    ternary_to_quaternary,
    quaternary_to_ternary,
    create_dual_base_system,
    demonstrate_dual_base,
    SEAL_COUNT,
    FACE_COUNT,
    CELL_COUNT,
    FACE_NAMES,
)

# Kaelhedron Expansion: Four Kaelhedrons, EM, Inversions
from .kaelhedron_expansion import (
    # Four Kaelhedrons
    KaelhedronType,
    KaelhedronVariant,
    FourKaelhedron,
    create_original_kaelhedron,
    create_anti_kaelhedron,
    create_dual_kaelhedron,
    create_conjugate_kaelhedron,
    # Electromagnetism
    KappaField,
    ElectromagneticState,
    EMFromKappa,
    # Inversions
    InversionType,
    Inversion,
    InversionsCatalog,
    # Unified
    KaelhedronExpansion,
    create_expansion_system,
    demonstrate_four_kaelhedrons,
    demonstrate_em_from_kappa,
    demonstrate_inversions,
    # Constants
    PHI,
    PHI_INV,
    PHI_SQ,
    PHI_MINUS,
    MU_1,
    MU_2,
    MU_P,
    GL32_ORDER,
    ALPHA_INV,
)

__all__ = [
    # Core
    "line_from_points",
    "point_from_lines",
    # Loop
    "GateState",
    "PolarityLoop",
    # Service
    "PolarityService",
    # Unified State
    "CellSnapshot",
    "DomainSnapshot",
    "KFormationStatus",
    "LoopState",
    "PolarityPhase",
    "PolaritySnapshot",
    "StateCallback",
    "UnifiedStateRegistry",
    "UnifiedSystemState",
    "get_state_registry",
    # Orchestrator
    "DOMAIN_NAMES",
    "FANO_LINES",
    "PolarityOrchestrator",
    # Telemetry
    "KFormationAdapter",
    "KaelhedronAdapter",
    "PolarityLoopAdapter",
    "ScalarArchitectureAdapter",
    "TelemetryEvent",
    "TelemetryHub",
    "TelemetryLevel",
    "TelemetrySource",
    "get_telemetry_hub",
    # Automorphisms
    "CYCLE",
    "IDENTITY",
    "REFLECTION",
    "CoherenceAutomorphismEngine",
    "apply_automorphism_to_cells",
    "compute_polarity_automorphism",
    "enumerate_psl32",
    "get_automorphism_for_line",
    "get_automorphism_for_point",
    "get_line_stabilizer",
    "get_stabilizer",
    # Streaming
    "DeltaCompressor",
    "StreamConfig",
    "StreamType",
    "VisualizationFrame",
    "VisualizationStreamer",
    "WebSocketBridge",
    "get_visualization_streamer",
    "get_websocket_bridge",
    # Integration
    "IntegratedPolaritySystem",
    "KaelhedronBridge",
    "LuminahedronBridge",
    "ScalarArchitectureBridge",
    "UnifiedMathBridgeAdapter",
    # Ternary Polaric (Luminahedron)
    "TernaryValue",
    "FanoPhase",
    "PolaricTransition",
    "TernaryPolaricEngine",
    "LuminahedronPosition",
    "LuminahedronPath",
    "generate_luminahedron_trajectory",
    "FANO_POINTS",
    "FANO_LINES",
    "POINT_INCIDENCE",
    # Quaternary Kaelhedron
    "QuaternaryValue",
    "KaelhedronCell",
    "KaelhedronState",
    "QuaternaryTransition",
    "QuaternaryKaelhedronEngine",
    "TernaryQuaternaryBridge",
    "ternary_to_quaternary",
    "quaternary_to_ternary",
    "create_dual_base_system",
    "demonstrate_dual_base",
    "SEAL_COUNT",
    "FACE_COUNT",
    "CELL_COUNT",
    "FACE_NAMES",
    # Kaelhedron Expansion
    "KaelhedronType",
    "KaelhedronVariant",
    "FourKaelhedron",
    "create_original_kaelhedron",
    "create_anti_kaelhedron",
    "create_dual_kaelhedron",
    "create_conjugate_kaelhedron",
    "KappaField",
    "ElectromagneticState",
    "EMFromKappa",
    "InversionType",
    "Inversion",
    "InversionsCatalog",
    "KaelhedronExpansion",
    "create_expansion_system",
    "demonstrate_four_kaelhedrons",
    "demonstrate_em_from_kappa",
    "demonstrate_inversions",
    "PHI",
    "PHI_INV",
    "PHI_SQ",
    "PHI_MINUS",
    "MU_1",
    "MU_2",
    "MU_P",
    "GL32_ORDER",
    "ALPHA_INV",
]
