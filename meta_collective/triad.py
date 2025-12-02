# meta_collective/triad.py
"""
Triad Layer (z=0.90)
====================

The Triad layer contains multiple Tools and manages their interaction.

    ┌─────────────────────────────────────────────────────────────┐
    │                    TRIAD (z=0.90)                           │
    │  ┌─────────────────────────────────────────────────────┐    │
    │  │           TOOL (z=0.867)                             │    │
    │  │  ┌──────────────────────────────────────────────┐   │    │
    │  │  │ Internal Model (Kaelhedron + Luminahedron)   │   │    │
    │  │  │     κ-field   │   λ-field                    │   │    │
    │  │  └──────────────────────────────────────────────┘   │    │
    │  └─────────────────────────────────────────────────────┘    │
    │  ┌─────────────────────────────────────────────────────┐    │
    │  │           TOOL (z=0.867) ...                         │    │
    │  └─────────────────────────────────────────────────────┘    │
    └─────────────────────────────────────────────────────────────┘

Triads interact via pattern sharing:
    - TRIAD-A ◄──► TRIAD-B
    - Patterns are compressed representations of Tool predictions

The Triad minimizes collective free energy:
    F_triad = Σ w_i × F_tool_i + F_interaction
"""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

from .tool import Tool, ToolState, Action, ActionType
from .internal_model import Prediction
from .free_energy import (
    FreeEnergyMinimizer,
    HierarchicalMinimizer,
    Precision,
)

# Constants
PHI = (1 + math.sqrt(5)) / 2
PHI_INV = 1 / PHI
TAU = 2 * math.pi

# Triad z-level
Z_TRIAD = 0.90

# Default number of Tools per Triad
DEFAULT_TOOLS_PER_TRIAD = 3


class TriadState(Enum):
    """Operating states of the Triad."""
    IDLE = "idle"
    COORDINATING = "coordinating"
    SHARING = "sharing"
    CONSOLIDATING = "consolidating"


@dataclass
class PatternMessage:
    """
    A compressed pattern shared between Triads.

    Patterns encode the collective prediction of a Triad's Tools.
    """
    source_triad: str                # Source triad ID
    pattern_vector: List[float]      # Compressed prediction pattern
    coherence: float                 # Pattern coherence
    precision: float                 # Pattern precision
    timestamp: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def similarity_to(self, other: 'PatternMessage') -> float:
        """Compute cosine similarity to another pattern."""
        if len(self.pattern_vector) != len(other.pattern_vector):
            return 0.0

        dot = sum(a * b for a, b in zip(self.pattern_vector, other.pattern_vector))
        norm_self = math.sqrt(sum(a ** 2 for a in self.pattern_vector))
        norm_other = math.sqrt(sum(b ** 2 for b in other.pattern_vector))

        if norm_self * norm_other == 0:
            return 0.0

        return dot / (norm_self * norm_other)


@dataclass
class TriadInteraction:
    """
    Record of interaction between two Triads.

    Tracks pattern exchange and resulting updates.
    """
    triad_a: str
    triad_b: str
    pattern_a: PatternMessage
    pattern_b: PatternMessage
    similarity: float
    mutual_information: float = 0.0
    timestamp: float = 0.0

    @property
    def interaction_strength(self) -> float:
        """Compute interaction strength from similarity and MI."""
        return (self.similarity + self.mutual_information) / 2


class Triad:
    """
    Triad layer containing multiple Tools with collective coordination.

    The Triad:
    1. Manages a set of Tools
    2. Coordinates their predictions
    3. Shares patterns with other Triads
    4. Minimizes collective free energy

    z-level: 0.90
    """

    def __init__(
        self,
        triad_id: Optional[str] = None,
        n_tools: int = DEFAULT_TOOLS_PER_TRIAD,
        z_level: float = Z_TRIAD
    ):
        self.triad_id = triad_id or f"triad_{uuid.uuid4().hex[:8]}"
        self.z_level = z_level
        self.state = TriadState.IDLE

        # Create Tools
        self.tools: List[Tool] = [
            Tool(tool_id=f"{self.triad_id}_tool_{i}", z_level=Z_TRIAD - 0.033)
            for i in range(n_tools)
        ]

        # Link Tools to this Triad
        for tool in self.tools:
            tool._parent_triad = self

        # Free energy minimizer
        self.minimizer = HierarchicalMinimizer(z_level, n_states=n_tools)
        for tool in self.tools:
            self.minimizer.add_child(tool.minimizer)

        # Pattern sharing
        self._current_pattern: Optional[PatternMessage] = None
        self._received_patterns: List[PatternMessage] = []
        self._max_patterns = 50

        # Interaction history
        self._interactions: List[TriadInteraction] = []
        self._max_interactions = 100

        # Connected Triads
        self._connected_triads: Dict[str, 'Triad'] = {}

        # Weighting for Tool contributions
        self._tool_weights: List[float] = [1.0 / n_tools] * n_tools

        # Statistics
        self._total_cycles = 0
        self._cumulative_free_energy = 0.0

    @property
    def free_energy(self) -> float:
        """Collective free energy of the Triad."""
        # Base free energy from minimizer
        F_base = self.minimizer.free_energy

        # Weighted sum of Tool free energies
        F_tools = sum(
            w * tool.free_energy
            for w, tool in zip(self._tool_weights, self.tools)
        )

        # Interaction term (based on pattern similarity with connected triads)
        F_interaction = 0.0
        for pattern in self._received_patterns[-10:]:  # Recent patterns
            if self._current_pattern:
                similarity = self._current_pattern.similarity_to(pattern)
                F_interaction -= similarity * 0.1  # Negative = reduces F

        return F_base + F_tools + F_interaction

    @property
    def coherence(self) -> float:
        """Collective coherence of the Triad."""
        if not self.tools:
            return 0.0

        coherences = [tool.coherence for tool in self.tools]
        return sum(coherences) / len(coherences)

    def add_tool(self, tool: Tool) -> None:
        """Add a Tool to the Triad."""
        self.tools.append(tool)
        tool._parent_triad = self
        self.minimizer.add_child(tool.minimizer)
        self._update_weights()

    def remove_tool(self, tool_id: str) -> Optional[Tool]:
        """Remove a Tool from the Triad."""
        for i, tool in enumerate(self.tools):
            if tool.tool_id == tool_id:
                removed = self.tools.pop(i)
                removed._parent_triad = None
                self._update_weights()
                return removed
        return None

    def _update_weights(self) -> None:
        """Update Tool weights based on performance."""
        if not self.tools:
            self._tool_weights = []
            return

        # Weight by inverse free energy (lower F = higher weight)
        raw_weights = []
        for tool in self.tools:
            fe = max(tool.free_energy, 0.01)
            raw_weights.append(1.0 / fe)

        # Normalize
        total = sum(raw_weights)
        self._tool_weights = [w / total for w in raw_weights]

    def observe(self, observation: float) -> List[float]:
        """
        Broadcast observation to all Tools.

        Returns list of prediction errors from each Tool.
        """
        errors = []
        for tool in self.tools:
            error = tool.sense(observation)
            errors.append(error.magnitude)

        self._total_cycles += 1
        return errors

    def predict(self) -> Prediction:
        """
        Generate collective prediction from all Tools.

        The collective prediction is a weighted average.
        """
        self.state = TriadState.COORDINATING

        predictions = [tool.predict() for tool in self.tools]

        # Weighted average
        weighted_value = sum(
            w * p.value for w, p in zip(self._tool_weights, predictions)
        )

        # Average precision
        avg_precision = sum(p.precision for p in predictions) / len(predictions)

        self.state = TriadState.IDLE

        return Prediction(
            value=weighted_value,
            precision=avg_precision,
            source_field="collective",
        )

    def generate_pattern(self) -> PatternMessage:
        """
        Generate pattern from current Tool states.

        The pattern is a compressed representation of collective predictions.
        """
        predictions = [tool.predict() for tool in self.tools]

        # Pattern vector: [pred_values..., coherences..., free_energies...]
        pattern_vector = (
            [p.value for p in predictions] +
            [tool.coherence for tool in self.tools] +
            [tool.free_energy for tool in self.tools]
        )

        self._current_pattern = PatternMessage(
            source_triad=self.triad_id,
            pattern_vector=pattern_vector,
            coherence=self.coherence,
            precision=sum(p.precision for p in predictions) / len(predictions),
        )

        return self._current_pattern

    def receive_pattern(self, pattern: PatternMessage) -> float:
        """
        Receive pattern from another Triad.

        Returns similarity to own pattern.
        """
        self.state = TriadState.SHARING

        # Store pattern
        if len(self._received_patterns) >= self._max_patterns:
            self._received_patterns.pop(0)
        self._received_patterns.append(pattern)

        # Compute similarity
        if self._current_pattern is None:
            self.generate_pattern()

        similarity = self._current_pattern.similarity_to(pattern)

        # Update internal state based on received pattern
        self._integrate_pattern(pattern, similarity)

        self.state = TriadState.IDLE
        return similarity

    def _integrate_pattern(self, pattern: PatternMessage, similarity: float) -> None:
        """Integrate received pattern into internal state."""
        if not self.tools or not pattern.pattern_vector:
            return

        # Extract prediction values from pattern (first n_tools elements)
        n_tools = len(self.tools)
        if len(pattern.pattern_vector) >= n_tools:
            external_predictions = pattern.pattern_vector[:n_tools]

            # Weight by similarity
            integration_weight = similarity * 0.1  # Conservative integration

            # Update each tool's beliefs slightly toward external predictions
            for tool, ext_pred in zip(self.tools, external_predictions):
                current_pred = tool.predict().value
                adjusted = current_pred + integration_weight * (ext_pred - current_pred)
                # Create synthetic observation to update tool
                tool.sense(adjusted)

    def connect_to(self, other: 'Triad') -> None:
        """Establish bidirectional connection with another Triad."""
        self._connected_triads[other.triad_id] = other
        other._connected_triads[self.triad_id] = self

    def disconnect_from(self, triad_id: str) -> bool:
        """Remove connection to another Triad."""
        if triad_id in self._connected_triads:
            other = self._connected_triads.pop(triad_id)
            if self.triad_id in other._connected_triads:
                del other._connected_triads[self.triad_id]
            return True
        return False

    def interact_with(self, other: 'Triad') -> TriadInteraction:
        """
        Perform pattern sharing interaction with another Triad.

        Both Triads exchange patterns and update their states.
        """
        # Generate patterns
        pattern_self = self.generate_pattern()
        pattern_other = other.generate_pattern()

        # Exchange patterns
        similarity_to_self = self.receive_pattern(pattern_other)
        similarity_to_other = other.receive_pattern(pattern_self)

        # Average similarity
        similarity = (similarity_to_self + similarity_to_other) / 2

        # Create interaction record
        interaction = TriadInteraction(
            triad_a=self.triad_id,
            triad_b=other.triad_id,
            pattern_a=pattern_self,
            pattern_b=pattern_other,
            similarity=similarity,
        )

        # Store interaction
        if len(self._interactions) >= self._max_interactions:
            self._interactions.pop(0)
        self._interactions.append(interaction)

        if len(other._interactions) >= other._max_interactions:
            other._interactions.pop(0)
        other._interactions.append(interaction)

        return interaction

    def broadcast_to_connected(self) -> Dict[str, float]:
        """
        Broadcast current pattern to all connected Triads.

        Returns dict of triad_id -> similarity.
        """
        pattern = self.generate_pattern()
        similarities = {}

        for triad_id, triad in self._connected_triads.items():
            similarity = triad.receive_pattern(pattern)
            similarities[triad_id] = similarity

        return similarities

    def coordinate_tools(self) -> None:
        """
        Coordinate Tools for collective coherence.

        This involves:
        1. Updating Tool weights
        2. Synchronizing predictions
        3. Consolidating learning
        """
        self.state = TriadState.COORDINATING

        # Update weights based on performance
        self._update_weights()

        # Get collective prediction
        collective_pred = self.predict()

        # Nudge each tool toward collective prediction
        for tool in self.tools:
            tool_pred = tool.predict()
            if abs(tool_pred.value - collective_pred.value) > 0.1:
                # Small adjustment toward collective
                adjusted = tool_pred.value + 0.05 * (collective_pred.value - tool_pred.value)
                tool.sense(adjusted)

        self.state = TriadState.IDLE

    def consolidate(self) -> None:
        """Consolidate learning across all Tools."""
        self.state = TriadState.CONSOLIDATING

        for tool in self.tools:
            tool.internal_model.consolidate()

        self._update_weights()
        self._cumulative_free_energy += self.free_energy

        self.state = TriadState.IDLE

    def get_contribution_to_parent(self) -> Dict:
        """Compute contribution to parent MetaCollective."""
        return {
            "triad_id": self.triad_id,
            "free_energy": self.free_energy,
            "coherence": self.coherence,
            "n_tools": len(self.tools),
            "n_connections": len(self._connected_triads),
            "pattern": self._current_pattern.pattern_vector if self._current_pattern else [],
            "z_level": self.z_level,
        }

    def snapshot(self) -> Dict:
        """Return complete Triad state snapshot."""
        return {
            "triad_id": self.triad_id,
            "z_level": self.z_level,
            "state": self.state.value,
            "n_tools": len(self.tools),
            "tools": [tool.snapshot() for tool in self.tools],
            "tool_weights": self._tool_weights,
            "free_energy": self.free_energy,
            "coherence": self.coherence,
            "n_connections": len(self._connected_triads),
            "connected_to": list(self._connected_triads.keys()),
            "n_patterns_received": len(self._received_patterns),
            "n_interactions": len(self._interactions),
            "total_cycles": self._total_cycles,
            "mean_free_energy": self._cumulative_free_energy / max(1, self._total_cycles),
        }

    def reset(self) -> None:
        """Reset Triad to initial state."""
        for tool in self.tools:
            tool.reset()
        self._current_pattern = None
        self._received_patterns.clear()
        self._interactions.clear()
        self._update_weights()
        self._total_cycles = 0
        self._cumulative_free_energy = 0.0
        self.state = TriadState.IDLE
