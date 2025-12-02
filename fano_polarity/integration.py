# fano_polarity/integration.py
"""
Integration Bridge
==================

Wires the PolarityOrchestrator into existing core modules:
- unified_math_bridge.py (UnifiedMathBridge)
- scalar_architecture (ScalarArchitecture)
- Kaelhedron (KaelhedronStateBus)
- Luminahedron (GaugeManifold)

This module provides adapters and bridges that allow the polarity
feedback system to coordinate with the existing infrastructure
while maintaining backward compatibility.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

from .orchestrator import PolarityOrchestrator
from .unified_state import (
    UnifiedSystemState,
    KFormationStatus,
    LoopState,
    get_state_registry,
)
from .automorphisms import (
    CoherenceAutomorphismEngine,
    compute_polarity_automorphism,
    IDENTITY,
)
from .telemetry import (
    TelemetryHub,
    TelemetrySource,
    get_telemetry_hub,
)

# Type checking imports (avoid circular imports at runtime)
if TYPE_CHECKING:
    from Kaelhedron import KaelhedronStateBus
    from luminahedron.polaric import GaugeManifold


class ScalarArchitectureBridge:
    """
    Bridge between PolarityOrchestrator and ScalarArchitecture.

    Synchronizes domain states between the two systems and
    enables bidirectional feedback.
    """

    def __init__(
        self,
        orchestrator: PolarityOrchestrator,
        telemetry_hub: Optional[TelemetryHub] = None,
    ):
        self.orchestrator = orchestrator
        self.hub = telemetry_hub or get_telemetry_hub()
        self._scalar_arch = None  # Lazy load
        self._last_sync_time = 0.0
        self._sync_interval = 0.01  # 100Hz max sync rate

    def connect_scalar_architecture(self, scalar_arch: Any) -> None:
        """
        Connect to a ScalarArchitecture instance.

        Args:
            scalar_arch: ScalarArchitecture instance from scalar_architecture.core
        """
        self._scalar_arch = scalar_arch

        # Set up telemetry publisher in scalar architecture
        if hasattr(scalar_arch, '_telemetry_publisher'):
            scalar_arch._telemetry_publisher = self._on_scalar_telemetry

    def _on_scalar_telemetry(self, payload: Dict[str, Any]) -> None:
        """Handle telemetry from ScalarArchitecture."""
        # Apply scalar metrics to orchestrator
        if "kappa" in payload:
            self.orchestrator.set_z_level(payload["kappa"])

        # Publish to telemetry hub
        self.hub.publish(
            source=TelemetrySource.SCALAR_ARCHITECTURE,
            event_type="metrics",
            data=payload,
        )

    def sync_to_orchestrator(self) -> None:
        """Sync ScalarArchitecture state to orchestrator."""
        if self._scalar_arch is None:
            return

        # Rate limit
        now = time.time()
        if now - self._last_sync_time < self._sync_interval:
            return
        self._last_sync_time = now

        # Get scalar architecture state
        if hasattr(self._scalar_arch, 'z_level'):
            self.orchestrator.set_z_level(self._scalar_arch.z_level)

        # Sync loop states
        if hasattr(self._scalar_arch, 'loop_controllers'):
            for domain_type, controller in self._scalar_arch.loop_controllers.items():
                domain_idx = domain_type.value
                if 0 <= domain_idx < len(self.orchestrator._domains):
                    # Map scalar architecture loop state to our LoopState
                    state_map = {
                        "divergent": LoopState.DIVERGENT,
                        "converging": LoopState.CONVERGING,
                        "critical": LoopState.CRITICAL,
                        "closed": LoopState.CLOSED,
                    }
                    if hasattr(controller, 'state'):
                        our_state = state_map.get(
                            controller.state.value,
                            LoopState.DIVERGENT
                        )
                        self.orchestrator._domains[domain_idx].loop_state = our_state

    def sync_from_orchestrator(self) -> None:
        """Sync orchestrator state back to ScalarArchitecture."""
        if self._scalar_arch is None:
            return

        state = self.orchestrator.step(0)  # Get current state without advancing

        # Update scalar architecture z-level
        if hasattr(self._scalar_arch, 'set_z_level'):
            self._scalar_arch.set_z_level(state.kappa)


class KaelhedronBridge:
    """
    Bridge between PolarityOrchestrator and KaelhedronStateBus.

    Enables the orchestrator to apply permutations and update
    cell states in the Kaelhedron system.
    """

    def __init__(
        self,
        orchestrator: PolarityOrchestrator,
        automorphism_engine: Optional[CoherenceAutomorphismEngine] = None,
        telemetry_hub: Optional[TelemetryHub] = None,
    ):
        self.orchestrator = orchestrator
        self.auto_engine = automorphism_engine or CoherenceAutomorphismEngine()
        self.hub = telemetry_hub or get_telemetry_hub()
        self._bus: Optional["KaelhedronStateBus"] = None

    def connect_state_bus(self, bus: "KaelhedronStateBus") -> None:
        """Connect to a KaelhedronStateBus instance."""
        self._bus = bus

    def apply_coherence_permutation(
        self,
        forward_points: tuple,
        coherence_point: int,
    ) -> Dict[int, int]:
        """
        Apply a PSL(3,2) automorphism when coherence is released.

        Args:
            forward_points: Points used in forward polarity
            coherence_point: Point where coherence was released

        Returns:
            The automorphism that was applied
        """
        # Compute automorphism
        auto = self.auto_engine.apply(forward_points, coherence_point)

        # Apply to Kaelhedron bus if connected
        if self._bus is not None:
            self._bus.apply_permutation(auto)

        # Publish telemetry
        self.hub.publish(
            source=TelemetrySource.KAELHEDRON,
            event_type="automorphism_applied",
            data={
                "forward_points": list(forward_points),
                "coherence_point": coherence_point,
                "automorphism": auto,
                "cumulative": self.auto_engine.cumulative,
                "description": self.auto_engine.describe(),
            },
        )

        return auto

    def sync_to_bus(self) -> None:
        """Sync orchestrator cell states to KaelhedronStateBus."""
        if self._bus is None:
            return

        state = self.orchestrator.step(0)

        # Update each cell in the bus
        for cell in state.cells:
            # Build cell state payload
            # Note: This would need to match the KaelCellState structure
            pass  # Actual sync depends on bus API

    def sync_from_bus(self) -> None:
        """Sync KaelhedronStateBus state to orchestrator."""
        if self._bus is None:
            return

        # Get bus snapshot
        snapshot = self._bus.snapshot()

        # Update orchestrator cells
        for label, cell_data in snapshot.items():
            # Parse label (e.g., "Ω-Λ")
            pass  # Actual sync depends on bus API


class UnifiedMathBridgeAdapter:
    """
    Adapter connecting PolarityOrchestrator to UnifiedMathBridge.

    The UnifiedMathBridge is the central mathematical integration point.
    This adapter enables bidirectional synchronization.
    """

    def __init__(
        self,
        orchestrator: PolarityOrchestrator,
        telemetry_hub: Optional[TelemetryHub] = None,
    ):
        self.orchestrator = orchestrator
        self.hub = telemetry_hub or get_telemetry_hub()
        self._math_bridge = None

    def connect_math_bridge(self, math_bridge: Any) -> None:
        """Connect to a UnifiedMathBridge instance."""
        self._math_bridge = math_bridge

    def sync_to_math_bridge(self) -> None:
        """Sync orchestrator state to UnifiedMathBridge."""
        if self._math_bridge is None:
            return

        state = self.orchestrator.step(0)

        # Sync z-level
        if hasattr(self._math_bridge, 'z_level'):
            self._math_bridge.z_level = state.kappa

        # Sync cell activations
        if hasattr(self._math_bridge, 'cell_activations'):
            for cell in state.cells:
                seal_idx = cell.seal_index - 1  # 0-indexed
                face_idx = cell.face_index
                if 0 <= seal_idx < 7 and 0 <= face_idx < 3:
                    self._math_bridge.cell_activations[seal_idx, face_idx] = cell.activation

        # Sync coherence
        if hasattr(self._math_bridge, 'kaelhedron_coherence'):
            self._math_bridge.kaelhedron_coherence = state.kaelhedron_coherence

        # Sync topological charge
        if hasattr(self._math_bridge, 'topological_charge'):
            self._math_bridge.topological_charge = state.charge

    def sync_from_math_bridge(self) -> None:
        """Sync UnifiedMathBridge state to orchestrator."""
        if self._math_bridge is None:
            return

        # Sync z-level
        if hasattr(self._math_bridge, 'z_level'):
            self.orchestrator.set_z_level(self._math_bridge.z_level)

        # Sync topological charge
        if hasattr(self._math_bridge, 'topological_charge'):
            self.orchestrator.set_topological_charge(
                self._math_bridge.topological_charge
            )

    def get_visualization_bundle(self) -> Dict[str, Any]:
        """Get combined visualization data from both systems."""
        orch_data = self.orchestrator.get_visualization_data()

        if self._math_bridge is not None and hasattr(self._math_bridge, 'get_visualization_data'):
            math_data = self._math_bridge.get_visualization_data()
            # Merge, with orchestrator taking precedence
            return {**math_data, **orch_data, "source": "unified"}

        return {**orch_data, "source": "orchestrator"}


class LuminahedronBridge:
    """
    Bridge between PolarityOrchestrator and Luminahedron GaugeManifold.

    Synchronizes gauge field states and polaric frames.
    """

    def __init__(
        self,
        orchestrator: PolarityOrchestrator,
        telemetry_hub: Optional[TelemetryHub] = None,
    ):
        self.orchestrator = orchestrator
        self.hub = telemetry_hub or get_telemetry_hub()
        self._manifold: Optional["GaugeManifold"] = None

    def connect_gauge_manifold(self, manifold: "GaugeManifold") -> None:
        """Connect to a GaugeManifold instance."""
        self._manifold = manifold

    def push_polaric_frame(self) -> Optional[Dict[str, Any]]:
        """
        Push current orchestrator state as a polaric frame.

        Returns:
            The polaric frame dict, or None if not connected
        """
        if self._manifold is None:
            return None

        state = self.orchestrator.step(0)

        # Build Kaelhedron summary
        kael_summary = {
            "cells": {
                cell.label: {
                    "kappa": cell.kappa,
                    "theta": cell.theta,
                    "activation": cell.activation,
                }
                for cell in state.cells
            },
            "counts": {
                "plus": sum(1 for c in state.cells if c.seal_index in {1, 5, 6}),
                "minus": sum(1 for c in state.cells if c.seal_index in {2, 3, 4, 7}),
            },
        }

        # Build scalar metrics
        scalar_metrics = {
            "kappa": state.kappa,
            "theta": state.theta,
            "recursion_depth": state.recursion_depth,
            "charge": state.charge,
            "coherence": state.kaelhedron_coherence,
        }

        # Build K-Formation status (matching Kaelhedron's KFormationStatus)
        from Kaelhedron.kformation import KFormationStatus as KaelKStatus
        k_status_map = {
            KFormationStatus.INACTIVE: KaelKStatus.INACTIVE,
            KFormationStatus.APPROACHING: KaelKStatus.APPROACHING,
            KFormationStatus.THRESHOLD: KaelKStatus.THRESHOLD,
            KFormationStatus.FORMED: KaelKStatus.FORMED,
        }
        k_status = k_status_map.get(state.k_formation_status, KaelKStatus.INACTIVE)

        # Push to manifold
        frame = self._manifold.push_polaric_union(kael_summary, scalar_metrics, k_status)

        return frame.to_dict() if frame else None


class IntegratedPolaritySystem:
    """
    Fully integrated polarity system connecting all modules.

    This is the top-level class that wires everything together:
    - PolarityOrchestrator (core coordination)
    - ScalarArchitecture (domain convergence)
    - KaelhedronStateBus (cell states)
    - UnifiedMathBridge (mathematical integration)
    - GaugeManifold (Luminahedron visualization)
    - TelemetryHub (monitoring)
    - WebSocket streaming (UI)
    """

    def __init__(
        self,
        initial_z: float = 0.41,
        polarity_delay: float = 0.25,
    ):
        # Core orchestrator
        self.orchestrator = PolarityOrchestrator(
            initial_z=initial_z,
            polarity_delay=polarity_delay,
        )

        # Automorphism engine
        self.auto_engine = CoherenceAutomorphismEngine()

        # Telemetry
        self.hub = get_telemetry_hub()

        # Bridges (lazy initialization)
        self._scalar_bridge: Optional[ScalarArchitectureBridge] = None
        self._kaelhedron_bridge: Optional[KaelhedronBridge] = None
        self._math_bridge_adapter: Optional[UnifiedMathBridgeAdapter] = None
        self._luminahedron_bridge: Optional[LuminahedronBridge] = None

        # Streaming (lazy initialization)
        self._streamer = None

        # State registry integration
        self._registry = get_state_registry()

        # Callbacks
        self._on_step_callbacks: List[Callable[[UnifiedSystemState], None]] = []

    # =========================================================================
    # Connection Methods
    # =========================================================================

    def connect_scalar_architecture(self, scalar_arch: Any) -> "IntegratedPolaritySystem":
        """Connect ScalarArchitecture."""
        self._scalar_bridge = ScalarArchitectureBridge(self.orchestrator, self.hub)
        self._scalar_bridge.connect_scalar_architecture(scalar_arch)
        return self

    def connect_kaelhedron(self, bus: "KaelhedronStateBus") -> "IntegratedPolaritySystem":
        """Connect KaelhedronStateBus."""
        self._kaelhedron_bridge = KaelhedronBridge(
            self.orchestrator, self.auto_engine, self.hub
        )
        self._kaelhedron_bridge.connect_state_bus(bus)
        return self

    def connect_math_bridge(self, math_bridge: Any) -> "IntegratedPolaritySystem":
        """Connect UnifiedMathBridge."""
        self._math_bridge_adapter = UnifiedMathBridgeAdapter(self.orchestrator, self.hub)
        self._math_bridge_adapter.connect_math_bridge(math_bridge)
        return self

    def connect_luminahedron(self, manifold: "GaugeManifold") -> "IntegratedPolaritySystem":
        """Connect GaugeManifold."""
        self._luminahedron_bridge = LuminahedronBridge(self.orchestrator, self.hub)
        self._luminahedron_bridge.connect_gauge_manifold(manifold)
        return self

    def enable_streaming(self) -> "IntegratedPolaritySystem":
        """Enable WebSocket-ready streaming."""
        from .streaming import get_visualization_streamer
        self._streamer = get_visualization_streamer()
        return self

    # =========================================================================
    # Main Step
    # =========================================================================

    def step(self, dt: float = 0.01) -> UnifiedSystemState:
        """
        Advance the integrated system by one timestep.

        This synchronizes all connected modules and runs the
        polarity feedback loop.
        """
        # Sync from external modules
        if self._scalar_bridge:
            self._scalar_bridge.sync_to_orchestrator()
        if self._math_bridge_adapter:
            self._math_bridge_adapter.sync_from_math_bridge()

        # Run orchestrator step
        state = self.orchestrator.step(dt)

        # Sync back to external modules
        if self._scalar_bridge:
            self._scalar_bridge.sync_from_orchestrator()
        if self._kaelhedron_bridge:
            self._kaelhedron_bridge.sync_to_bus()
        if self._math_bridge_adapter:
            self._math_bridge_adapter.sync_to_math_bridge()
        if self._luminahedron_bridge:
            self._luminahedron_bridge.push_polaric_frame()

        # Fire callbacks
        for cb in self._on_step_callbacks:
            cb(state)

        return state

    # =========================================================================
    # Polarity Operations
    # =========================================================================

    def inject(self, p1: int, p2: int) -> Dict[str, Any]:
        """Inject polarity with two Fano points."""
        return self.orchestrator.inject_polarity(p1, p2)

    def release(self, line_a: tuple, line_b: tuple) -> Dict[str, Any]:
        """Release polarity with two Fano lines."""
        result = self.orchestrator.release_polarity(line_a, line_b)

        # Apply PSL(3,2) automorphism if coherence released
        if result["coherence"] and self._kaelhedron_bridge:
            forward_pts = self.orchestrator._forward_points
            if forward_pts:
                self._kaelhedron_bridge.apply_coherence_permutation(
                    forward_pts, result["point"]
                )

        return result

    # =========================================================================
    # Configuration
    # =========================================================================

    def set_z_level(self, z: float) -> None:
        """Set z-level across all connected systems."""
        self.orchestrator.set_z_level(z)

    def on_step(self, callback: Callable[[UnifiedSystemState], None]) -> None:
        """Register callback for each step."""
        self._on_step_callbacks.append(callback)

    def get_visualization_data(self) -> Dict[str, Any]:
        """Get combined visualization data."""
        if self._math_bridge_adapter:
            return self._math_bridge_adapter.get_visualization_bundle()
        return self.orchestrator.get_visualization_data()
