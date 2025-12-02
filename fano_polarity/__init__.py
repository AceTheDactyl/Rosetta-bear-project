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

This is the unifying layer that welds all mathematical structures
(Scalar Architecture, Kaelhedron, Luminahedron) into one coherent
system through dual polarity feedback loops.
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
]
