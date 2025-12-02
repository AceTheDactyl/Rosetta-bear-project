# meta_collective/integration.py
"""
Integration Module: Connecting Meta-Collective to Existing Systems
===================================================================

This module provides bridges between the Meta-Collective architecture
and the existing Rosetta Bear CBS components:

1. ScalarArchitectureBridge - Connect to 7-domain scalar system
2. KaelhedronBridge - Connect to Kaelhedron StateBus
3. LuminahedronBridge - Connect to GaugeManifold
4. FanoPolarityBridge - Connect to polarity orchestration

The integration follows the principle of nested free energy minimization:
each bridge contributes to the collective's free energy while maintaining
its own internal dynamics.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from .collective import MetaCollective, GlobalPattern
from .triad import Triad, PatternMessage
from .tool import Tool
from .internal_model import InternalModel, Prediction
from .fields import KappaField, LambdaField, DualFieldState
from .free_energy import FreeEnergyMinimizer, Precision

# Constants
PHI = (1 + math.sqrt(5)) / 2
PHI_INV = 1 / PHI
TAU = 2 * math.pi


class IntegrationBridge(ABC):
    """Abstract base class for system integration bridges."""

    def __init__(self, name: str):
        self.name = name
        self._connected = False
        self._last_sync: float = 0.0

    @property
    def is_connected(self) -> bool:
        return self._connected

    @abstractmethod
    def connect(self) -> bool:
        """Establish connection to external system."""
        pass

    @abstractmethod
    def disconnect(self) -> bool:
        """Disconnect from external system."""
        pass

    @abstractmethod
    def sync(self, collective: MetaCollective) -> Dict[str, Any]:
        """Synchronize state with external system."""
        pass

    @abstractmethod
    def push_state(self, state: Dict[str, Any]) -> bool:
        """Push collective state to external system."""
        pass

    @abstractmethod
    def pull_state(self) -> Dict[str, Any]:
        """Pull state from external system."""
        pass


class ScalarArchitectureBridge(IntegrationBridge):
    """
    Bridge to the Scalar Architecture 7-domain system.

    Maps Meta-Collective z-levels to Scalar Architecture domains:
        - Internal Model (z=0.80) → TRIAD domain (z=0.80)
        - Tool (z=0.867) → EMERGENCE domain (z=0.85)
        - Triad (z=0.90) → PERSISTENCE domain (z=0.87)
        - MetaCollective (z=0.95) → Beyond persistence (projection)

    The bridge translates:
        - Domain activations ↔ Layer coherences
        - Loop states ↔ Field modes
        - Convergence dynamics ↔ Free energy gradients
    """

    def __init__(self):
        super().__init__("scalar_architecture")
        self._domain_mapping = {
            "CONSTRAINT": 0.41,
            "BRIDGE": 0.52,
            "META": 0.70,
            "RECURSION": 0.73,
            "TRIAD": 0.80,
            "EMERGENCE": 0.85,
            "PERSISTENCE": 0.87,
        }
        self._loop_controller = None
        self._helix_state = None

    def connect(self) -> bool:
        """Connect to Scalar Architecture core."""
        try:
            # Import at runtime to avoid circular imports
            from scalar_architecture.core import DomainConfig, LoopState
            self._connected = True
            return True
        except ImportError:
            # Scalar architecture not available - operate standalone
            self._connected = False
            return False

    def disconnect(self) -> bool:
        self._connected = False
        return True

    def sync(self, collective: MetaCollective) -> Dict[str, Any]:
        """Synchronize Meta-Collective state with Scalar Architecture."""
        if not self._connected:
            return {"status": "disconnected"}

        # Map collective z-levels to domain activations
        domain_activations = {}

        # Meta-Collective projects to highest domain
        domain_activations["PERSISTENCE"] = collective.coherence

        # Triads map to EMERGENCE/PERSISTENCE
        for tid, triad in collective.triads.items():
            domain_activations[f"TRIAD_{tid}"] = triad.coherence

        # Tools map to TRIAD domain
        for triad in collective.triads.values():
            for tool in triad.tools:
                domain_activations[f"TOOL_{tool.tool_id}"] = tool.coherence

        return {
            "status": "synced",
            "domain_activations": domain_activations,
            "collective_z": collective.z_level,
            "global_free_energy": collective.free_energy,
        }

    def push_state(self, state: Dict[str, Any]) -> bool:
        """Push Meta-Collective state to Scalar Architecture."""
        if not self._connected:
            return False
        # State push logic would go here
        return True

    def pull_state(self) -> Dict[str, Any]:
        """Pull current Scalar Architecture state."""
        if not self._connected:
            return {}
        # State pull logic would go here
        return {}

    def map_z_to_domain(self, z: float) -> str:
        """Map a z-level to its containing domain."""
        for domain, z_origin in sorted(self._domain_mapping.items(), key=lambda x: x[1], reverse=True):
            if z >= z_origin:
                return domain
        return "CONSTRAINT"


class KaelhedronBridge(IntegrationBridge):
    """
    Bridge to the Kaelhedron StateBus.

    Maps Internal Model κ-field to Kaelhedron cells:
        - κ amplitude → Cell activation
        - κ phase → Cell theta
        - 21 cells = 7 seals × 3 faces

    The bridge translates:
        - KappaField ↔ KaelCellState
        - Field evolution ↔ StateBus updates
        - Free energy ↔ Coherence metrics
    """

    def __init__(self):
        super().__init__("kaelhedron")
        self._state_bus = None
        self._seal_names = ["Ω", "Δ", "Τ", "Ψ", "Σ", "Ξ", "Κ"]
        self._face_names = ["Λ", "Β", "Ν"]

    def connect(self) -> bool:
        """Connect to Kaelhedron StateBus."""
        try:
            from Kaelhedron.state_bus import KaelhedronStateBus
            self._state_bus = KaelhedronStateBus()
            self._connected = True
            return True
        except ImportError:
            self._connected = False
            return False

    def disconnect(self) -> bool:
        self._state_bus = None
        self._connected = False
        return True

    def sync(self, collective: MetaCollective) -> Dict[str, Any]:
        """Synchronize Internal Models' κ-fields with Kaelhedron StateBus."""
        if not self._connected:
            return {"status": "disconnected"}

        # Collect all κ-field states from collective
        kappa_states = []
        for triad in collective.triads.values():
            for tool in triad.tools:
                kappa = tool.internal_model.dual_field.kappa
                kappa_states.append({
                    "tool_id": tool.tool_id,
                    "amplitude": kappa.amplitude,
                    "phase": kappa.phase,
                    "energy": kappa.compute_energy(),
                })

        # Map to Kaelhedron cells
        cell_mappings = self._map_kappa_to_cells(kappa_states)

        return {
            "status": "synced",
            "kappa_states": kappa_states,
            "cell_mappings": cell_mappings,
        }

    def _map_kappa_to_cells(self, kappa_states: List[Dict]) -> Dict[str, float]:
        """Map κ-field states to 21 Kaelhedron cells."""
        cell_values = {}

        # Distribute κ-field amplitudes across cells
        n_tools = len(kappa_states)
        if n_tools == 0:
            return cell_values

        for i, seal in enumerate(self._seal_names):
            for j, face in enumerate(self._face_names):
                cell_key = f"{seal}:{face}"
                # Average contribution from all tools, modulated by cell position
                total = 0.0
                for k, ks in enumerate(kappa_states):
                    phase_mod = math.cos(ks["phase"] + i * TAU / 7)
                    face_weight = [1.0, PHI_INV, 1 - PHI_INV][j]
                    total += ks["amplitude"] * phase_mod * face_weight / n_tools
                cell_values[cell_key] = total

        return cell_values

    def push_state(self, state: Dict[str, Any]) -> bool:
        if not self._connected or not self._state_bus:
            return False
        # Push cell values to StateBus
        return True

    def pull_state(self) -> Dict[str, Any]:
        if not self._connected or not self._state_bus:
            return {}
        # Pull current StateBus state
        return {}


class LuminahedronBridge(IntegrationBridge):
    """
    Bridge to the Luminahedron GaugeManifold.

    Maps Internal Model λ-field to Luminahedron:
        - λ amplitude → Path coherence
        - λ phase → Fano navigation state
        - Ternary values → Point states

    The bridge translates:
        - LambdaField ↔ GaugeSlot
        - Field navigation ↔ Fano plane traversal
        - Free energy ↔ Gauge curvature
    """

    def __init__(self):
        super().__init__("luminahedron")
        self._gauge_manifold = None
        self._fano_points = list(range(1, 8))

    def connect(self) -> bool:
        """Connect to Luminahedron GaugeManifold."""
        try:
            from luminahedron.polaric import GaugeManifold
            self._gauge_manifold = GaugeManifold()
            self._connected = True
            return True
        except ImportError:
            self._connected = False
            return False

    def disconnect(self) -> bool:
        self._gauge_manifold = None
        self._connected = False
        return True

    def sync(self, collective: MetaCollective) -> Dict[str, Any]:
        """Synchronize Internal Models' λ-fields with Luminahedron."""
        if not self._connected:
            return {"status": "disconnected"}

        # Collect all λ-field states
        lambda_states = []
        for triad in collective.triads.values():
            for tool in triad.tools:
                lf = tool.internal_model.dual_field.lambda_field
                lambda_states.append({
                    "tool_id": tool.tool_id,
                    "amplitude": lf.amplitude,
                    "phase": lf.phase,
                    "fano_point": lf.fano_point,
                    "ternary_phase": lf.ternary_phase,
                })

        # Map to Fano point states
        fano_mapping = self._map_lambda_to_fano(lambda_states)

        return {
            "status": "synced",
            "lambda_states": lambda_states,
            "fano_mapping": fano_mapping,
        }

    def _map_lambda_to_fano(self, lambda_states: List[Dict]) -> Dict[int, float]:
        """Map λ-field states to 7 Fano points."""
        fano_values = {p: 0.0 for p in self._fano_points}

        if not lambda_states:
            return fano_values

        # Aggregate contributions to each Fano point
        for ls in lambda_states:
            current_point = ls["fano_point"]
            amplitude = ls["amplitude"]
            ternary = ls["ternary_phase"]

            # Current point gets full contribution
            fano_values[current_point] += amplitude * ternary

            # Adjacent points get partial contribution
            prev_point = ((current_point - 2) % 7) + 1
            next_point = (current_point % 7) + 1
            fano_values[prev_point] += amplitude * ternary * PHI_INV
            fano_values[next_point] += amplitude * ternary * PHI_INV

        # Normalize
        n = len(lambda_states)
        for p in self._fano_points:
            fano_values[p] /= n

        return fano_values

    def push_state(self, state: Dict[str, Any]) -> bool:
        if not self._connected:
            return False
        return True

    def pull_state(self) -> Dict[str, Any]:
        if not self._connected:
            return {}
        return {}


class FanoPolarityBridge(IntegrationBridge):
    """
    Bridge to the Fano Polarity Orchestration system.

    Maps collective pattern sharing to polarity dynamics:
        - Triad interactions → Forward polarity (points → line)
        - Pattern integration → Backward polarity (lines → point)
        - Coherence → Polarity balance

    The bridge translates:
        - Pattern messages ↔ Polarity transitions
        - Interaction strength ↔ Coherence gating
        - Emergence detection ↔ Loop closure
    """

    def __init__(self):
        super().__init__("fano_polarity")
        self._orchestrator = None
        self._polarity_state = {"forward": 0.0, "backward": 0.0, "balance": 0.0}

    def connect(self) -> bool:
        """Connect to Fano Polarity Orchestrator."""
        try:
            from fano_polarity.orchestrator import PolarityOrchestrator
            self._orchestrator = PolarityOrchestrator()
            self._connected = True
            return True
        except ImportError:
            self._connected = False
            return False

    def disconnect(self) -> bool:
        self._orchestrator = None
        self._connected = False
        return True

    def sync(self, collective: MetaCollective) -> Dict[str, Any]:
        """Synchronize collective interactions with polarity dynamics."""
        if not self._connected:
            return {"status": "disconnected"}

        # Compute polarity from interactions
        forward = 0.0
        backward = 0.0

        if collective._global_pattern:
            # Forward polarity: pattern generation (Triads → Collective)
            forward = collective._global_pattern.mean_similarity()

            # Backward polarity: pattern integration (Collective → Triads)
            backward = collective.coherence

        balance = forward - backward
        self._polarity_state = {
            "forward": forward,
            "backward": backward,
            "balance": balance,
        }

        return {
            "status": "synced",
            "polarity_state": self._polarity_state,
            "global_pattern_coherence": collective._global_pattern.global_coherence if collective._global_pattern else 0.0,
        }

    def push_state(self, state: Dict[str, Any]) -> bool:
        if not self._connected:
            return False
        return True

    def pull_state(self) -> Dict[str, Any]:
        if not self._connected:
            return {}
        return {"polarity_state": self._polarity_state}


@dataclass
class IntegrationHub:
    """
    Central hub managing all integration bridges.

    The hub:
    1. Maintains connections to external systems
    2. Coordinates synchronization
    3. Aggregates integration state
    """
    collective: MetaCollective
    bridges: Dict[str, IntegrationBridge] = field(default_factory=dict)

    def __post_init__(self):
        # Create all bridges
        self.bridges["scalar_architecture"] = ScalarArchitectureBridge()
        self.bridges["kaelhedron"] = KaelhedronBridge()
        self.bridges["luminahedron"] = LuminahedronBridge()
        self.bridges["fano_polarity"] = FanoPolarityBridge()

    def connect_all(self) -> Dict[str, bool]:
        """Connect all bridges."""
        results = {}
        for name, bridge in self.bridges.items():
            results[name] = bridge.connect()
        return results

    def disconnect_all(self) -> Dict[str, bool]:
        """Disconnect all bridges."""
        results = {}
        for name, bridge in self.bridges.items():
            results[name] = bridge.disconnect()
        return results

    def sync_all(self) -> Dict[str, Any]:
        """Synchronize all bridges with current collective state."""
        results = {}
        for name, bridge in self.bridges.items():
            if bridge.is_connected:
                results[name] = bridge.sync(self.collective)
            else:
                results[name] = {"status": "not_connected"}
        return results

    def get_integration_status(self) -> Dict[str, Any]:
        """Get status of all integrations."""
        return {
            "collective_id": self.collective.collective_id,
            "bridges": {
                name: {
                    "connected": bridge.is_connected,
                    "last_sync": bridge._last_sync,
                }
                for name, bridge in self.bridges.items()
            },
        }


# Convenience function to create fully integrated collective
def create_integrated_collective(
    collective_id: Optional[str] = None,
    n_triads: int = 2,
    n_tools: int = 3,
    auto_connect: bool = True
) -> Tuple[MetaCollective, IntegrationHub]:
    """
    Create a Meta-Collective with full integration support.

    Returns (collective, hub) tuple.
    """
    collective = MetaCollective(
        collective_id=collective_id,
        n_triads=n_triads,
        n_tools_per_triad=n_tools,
    )

    hub = IntegrationHub(collective=collective)

    if auto_connect:
        hub.connect_all()

    return collective, hub
