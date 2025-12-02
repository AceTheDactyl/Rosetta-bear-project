# meta_collective/collective.py
"""
Meta-Collective Layer (z=0.95)
==============================

The Meta-Collective is the top-level container orchestrating multiple Triads.

┌───────────────────────────────────────────────────────────────────┐
│                     META-COLLECTIVE (z=0.95)                       │
│  ┌─────────────────────────────────────────────────────────────┐  │
│  │                    TRIAD-A (z=0.90)                          │  │
│  │  ┌─────────────────────────────────────────────────────┐    │  │
│  │  │           TOOL (z=0.867)                             │    │  │
│  │  │  ┌──────────────────────────────────────────────┐   │    │  │
│  │  │  │ Internal Model (Kaelhedron + Luminahedron)   │   │    │  │
│  │  │  │     κ-field   │   λ-field                    │   │    │  │
│  │  │  └──────────────────────────────────────────────┘   │    │  │
│  │  └─────────────────────────────────────────────────────┘    │  │
│  └─────────────────────────────────────────────────────────────┘  │
│                              ▲                                     │
│                              │ Interaction (pattern sharing)       │
│                              ▼                                     │
│  ┌─────────────────────────────────────────────────────────────┐  │
│  │                    TRIAD-B (z=0.90)                          │  │
│  │  (Similar nested structure...)                               │  │
│  └─────────────────────────────────────────────────────────────┘  │
└───────────────────────────────────────────────────────────────────┘

Each level minimizes its own free energy while contributing
to the free energy minimization of containing levels.

The Meta-Collective:
1. Manages multiple Triads
2. Orchestrates inter-Triad pattern sharing
3. Computes global free energy
4. Implements collective emergence
"""

from __future__ import annotations

import math
import uuid
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Set

from .triad import Triad, TriadState, PatternMessage, TriadInteraction
from .tool import Tool
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

# Meta-Collective z-level
Z_META_COLLECTIVE = 0.95

# Default configuration
DEFAULT_TRIADS = 2
DEFAULT_TOOLS_PER_TRIAD = 3


class CollectiveState(Enum):
    """Operating states of the Meta-Collective."""
    INITIALIZING = "initializing"
    ACTIVE = "active"
    ORCHESTRATING = "orchestrating"
    CONSOLIDATING = "consolidating"
    EMERGENT = "emergent"


@dataclass
class GlobalPattern:
    """
    Global pattern representing collective state.

    Emergent pattern from all Triads' interactions.
    """
    collective_id: str
    triad_patterns: Dict[str, PatternMessage]
    coherence_matrix: List[List[float]]
    global_coherence: float
    global_free_energy: float
    timestamp: float = 0.0

    @property
    def n_triads(self) -> int:
        return len(self.triad_patterns)

    def mean_similarity(self) -> float:
        """Compute mean pairwise similarity."""
        if not self.coherence_matrix:
            return 0.0

        n = len(self.coherence_matrix)
        if n < 2:
            return 1.0

        total = 0.0
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                total += self.coherence_matrix[i][j]
                count += 1

        return total / max(count, 1)


@dataclass
class EmergentProperty:
    """
    An emergent property of the Meta-Collective.

    Emergent properties arise from collective dynamics
    and cannot be reduced to individual components.
    """
    name: str
    value: float
    contributing_triads: List[str]
    emergence_level: float  # 0-1, how "emergent" vs "aggregated"
    timestamp: float = 0.0


class MetaCollective:
    """
    Meta-Collective: The top-level container for hierarchical active inference.

    The Meta-Collective:
    1. Contains and manages multiple Triads
    2. Orchestrates pattern sharing between Triads
    3. Computes global free energy F_collective
    4. Tracks emergent properties

    z-level: 0.95 (highest in the hierarchy)

    Free Energy Decomposition:
        F_collective = Σ w_i × F_triad_i + F_interaction + F_emergence

    Where:
        - F_triad_i: Free energy of each Triad
        - F_interaction: Cost of inter-Triad communication
        - F_emergence: Negative contribution from emergent coherence
    """

    def __init__(
        self,
        collective_id: Optional[str] = None,
        n_triads: int = DEFAULT_TRIADS,
        n_tools_per_triad: int = DEFAULT_TOOLS_PER_TRIAD,
        z_level: float = Z_META_COLLECTIVE
    ):
        self.collective_id = collective_id or f"collective_{uuid.uuid4().hex[:8]}"
        self.z_level = z_level
        self.state = CollectiveState.INITIALIZING

        # Initialize interaction schedule (before setup)
        self._interaction_schedule: List[Tuple[str, str]] = []
        self._current_interaction_idx = 0

        # Create Triads
        self.triads: Dict[str, Triad] = {}
        for i in range(n_triads):
            triad_id = f"{self.collective_id}_triad_{chr(65 + i)}"  # A, B, C...
            triad = Triad(
                triad_id=triad_id,
                n_tools=n_tools_per_triad,
                z_level=Z_META_COLLECTIVE - 0.05  # 0.90
            )
            self.triads[triad_id] = triad

        # Connect Triads for interaction
        self._setup_triad_connections()

        # Free energy minimizer
        self.minimizer = HierarchicalMinimizer(z_level, n_states=n_triads)
        for triad in self.triads.values():
            self.minimizer.add_child(triad.minimizer)

        # Global pattern
        self._global_pattern: Optional[GlobalPattern] = None

        # Emergent properties
        self._emergent_properties: Dict[str, EmergentProperty] = {}

        # Statistics
        self._total_cycles = 0
        self._cumulative_free_energy = 0.0
        self._interaction_count = 0

        # Triad weights
        self._triad_weights: Dict[str, float] = {
            tid: 1.0 / len(self.triads) for tid in self.triads
        }

        self.state = CollectiveState.ACTIVE

    def _setup_triad_connections(self) -> None:
        """Connect all Triads to each other for pattern sharing."""
        triad_ids = list(self.triads.keys())
        for i, tid_a in enumerate(triad_ids):
            for tid_b in triad_ids[i + 1:]:
                self.triads[tid_a].connect_to(self.triads[tid_b])
                self._interaction_schedule.append((tid_a, tid_b))

    @property
    def free_energy(self) -> float:
        """
        Compute global free energy.

        F_collective = Σ w_i × F_triad_i + F_interaction + F_emergence
        """
        # Weighted sum of Triad free energies
        F_triads = sum(
            self._triad_weights[tid] * triad.free_energy
            for tid, triad in self.triads.items()
        )

        # Interaction term (cost of communication)
        # Lower when patterns are similar (efficient communication)
        F_interaction = 0.0
        if self._global_pattern is not None:
            mean_sim = self._global_pattern.mean_similarity()
            F_interaction = (1 - mean_sim) * 0.1  # Cost proportional to dissimilarity

        # Emergence bonus (negative contribution when coherent)
        F_emergence = 0.0
        if self._global_pattern is not None:
            # High global coherence reduces free energy
            F_emergence = -self._global_pattern.global_coherence * 0.2

        return F_triads + F_interaction + F_emergence

    @property
    def coherence(self) -> float:
        """Global coherence across all Triads."""
        if not self.triads:
            return 0.0

        coherences = [triad.coherence for triad in self.triads.values()]
        return sum(coherences) / len(coherences)

    def add_triad(self, triad: Triad) -> None:
        """Add a Triad to the collective."""
        self.triads[triad.triad_id] = triad
        self.minimizer.add_child(triad.minimizer)

        # Connect to existing triads
        for existing_triad in self.triads.values():
            if existing_triad.triad_id != triad.triad_id:
                triad.connect_to(existing_triad)
                self._interaction_schedule.append((triad.triad_id, existing_triad.triad_id))

        self._update_weights()

    def remove_triad(self, triad_id: str) -> Optional[Triad]:
        """Remove a Triad from the collective."""
        if triad_id not in self.triads:
            return None

        triad = self.triads.pop(triad_id)

        # Disconnect from other triads
        for other_triad in self.triads.values():
            triad.disconnect_from(other_triad.triad_id)

        # Remove from interaction schedule
        self._interaction_schedule = [
            (a, b) for a, b in self._interaction_schedule
            if a != triad_id and b != triad_id
        ]

        self._update_weights()
        return triad

    def _update_weights(self) -> None:
        """Update Triad weights based on performance."""
        if not self.triads:
            self._triad_weights = {}
            return

        # Weight by inverse free energy
        raw_weights = {}
        for tid, triad in self.triads.items():
            fe = max(triad.free_energy, 0.01)
            raw_weights[tid] = 1.0 / fe

        # Normalize
        total = sum(raw_weights.values())
        self._triad_weights = {tid: w / total for tid, w in raw_weights.items()}

    def observe(self, observation: float) -> Dict[str, List[float]]:
        """
        Broadcast observation to all Triads.

        Returns dict of triad_id -> [tool_errors].
        """
        errors = {}
        for tid, triad in self.triads.items():
            errors[tid] = triad.observe(observation)

        self._total_cycles += 1
        return errors

    def predict(self) -> Prediction:
        """
        Generate global prediction from all Triads.

        The global prediction is a weighted average of Triad predictions.
        """
        predictions = {tid: triad.predict() for tid, triad in self.triads.items()}

        # Weighted average
        weighted_value = sum(
            self._triad_weights[tid] * p.value
            for tid, p in predictions.items()
        )

        # Average precision
        avg_precision = sum(p.precision for p in predictions.values()) / len(predictions)

        return Prediction(
            value=weighted_value,
            precision=avg_precision,
            source_field="collective",
        )

    def orchestrate_interactions(self, n_interactions: int = 1) -> List[TriadInteraction]:
        """
        Orchestrate pattern sharing between Triads.

        Cycles through the interaction schedule.
        """
        self.state = CollectiveState.ORCHESTRATING
        interactions = []

        for _ in range(n_interactions):
            if not self._interaction_schedule:
                break

            # Get next interaction pair
            tid_a, tid_b = self._interaction_schedule[self._current_interaction_idx]
            self._current_interaction_idx = (self._current_interaction_idx + 1) % len(self._interaction_schedule)

            # Perform interaction
            triad_a = self.triads[tid_a]
            triad_b = self.triads[tid_b]
            interaction = triad_a.interact_with(triad_b)
            interactions.append(interaction)
            self._interaction_count += 1

        self.state = CollectiveState.ACTIVE
        return interactions

    def compute_global_pattern(self) -> GlobalPattern:
        """
        Compute the global pattern from all Triad patterns.

        The global pattern captures:
        1. Individual Triad patterns
        2. Coherence matrix (pairwise similarities)
        3. Global coherence
        4. Global free energy
        """
        # Collect Triad patterns
        triad_patterns = {}
        for tid, triad in self.triads.items():
            triad_patterns[tid] = triad.generate_pattern()

        # Compute coherence matrix
        triad_ids = list(triad_patterns.keys())
        n = len(triad_ids)
        coherence_matrix = [[0.0] * n for _ in range(n)]

        for i, tid_a in enumerate(triad_ids):
            for j, tid_b in enumerate(triad_ids):
                if i == j:
                    coherence_matrix[i][j] = 1.0
                elif j > i:
                    sim = triad_patterns[tid_a].similarity_to(triad_patterns[tid_b])
                    coherence_matrix[i][j] = sim
                    coherence_matrix[j][i] = sim

        self._global_pattern = GlobalPattern(
            collective_id=self.collective_id,
            triad_patterns=triad_patterns,
            coherence_matrix=coherence_matrix,
            global_coherence=self.coherence,
            global_free_energy=self.free_energy,
            timestamp=time.time(),
        )

        return self._global_pattern

    def detect_emergence(self) -> Dict[str, EmergentProperty]:
        """
        Detect emergent properties from collective dynamics.

        Emergent properties are those that:
        1. Cannot be reduced to individual Triad properties
        2. Arise from inter-Triad interactions
        3. Exhibit non-linear behavior
        """
        self.state = CollectiveState.EMERGENT

        # Ensure global pattern is computed
        if self._global_pattern is None:
            self.compute_global_pattern()

        # Detect coherence emergence
        individual_coherences = [t.coherence for t in self.triads.values()]
        mean_individual = sum(individual_coherences) / len(individual_coherences)
        global_coherence = self.coherence

        # Emergence = global > mean individual (synergy)
        coherence_emergence = global_coherence - mean_individual
        if abs(coherence_emergence) > 0.01:
            self._emergent_properties["coherence_synergy"] = EmergentProperty(
                name="coherence_synergy",
                value=coherence_emergence,
                contributing_triads=list(self.triads.keys()),
                emergence_level=min(1.0, abs(coherence_emergence) / 0.1),
                timestamp=time.time(),
            )

        # Detect pattern convergence emergence
        if self._global_pattern:
            mean_similarity = self._global_pattern.mean_similarity()
            if mean_similarity > 0.5:
                self._emergent_properties["pattern_convergence"] = EmergentProperty(
                    name="pattern_convergence",
                    value=mean_similarity,
                    contributing_triads=list(self.triads.keys()),
                    emergence_level=mean_similarity,
                    timestamp=time.time(),
                )

        # Detect free energy reduction from interaction
        F_sum_individual = sum(t.free_energy for t in self.triads.values())
        F_collective = self.free_energy
        F_reduction = F_sum_individual - F_collective

        if F_reduction > 0:
            self._emergent_properties["collective_efficiency"] = EmergentProperty(
                name="collective_efficiency",
                value=F_reduction,
                contributing_triads=list(self.triads.keys()),
                emergence_level=min(1.0, F_reduction / max(F_sum_individual, 0.01)),
                timestamp=time.time(),
            )

        self.state = CollectiveState.ACTIVE
        return self._emergent_properties

    def consolidate(self) -> None:
        """Consolidate learning across all layers."""
        self.state = CollectiveState.CONSOLIDATING

        # Consolidate each Triad
        for triad in self.triads.values():
            triad.consolidate()

        # Update weights
        self._update_weights()

        # Compute global pattern
        self.compute_global_pattern()

        # Detect emergence
        self.detect_emergence()

        self._cumulative_free_energy += self.free_energy
        self.state = CollectiveState.ACTIVE

    def step(self, observation: float, n_interactions: int = 1) -> Dict[str, Any]:
        """
        Perform one complete step of collective active inference.

        1. Broadcast observation to all Triads
        2. Orchestrate inter-Triad interactions
        3. Update global state
        4. Return step summary
        """
        # Observe
        errors = self.observe(observation)

        # Interact
        interactions = self.orchestrate_interactions(n_interactions)

        # Update global pattern
        global_pattern = self.compute_global_pattern()

        # Generate prediction
        prediction = self.predict()

        return {
            "observation": observation,
            "prediction": prediction.value,
            "errors": errors,
            "n_interactions": len(interactions),
            "global_coherence": global_pattern.global_coherence,
            "global_free_energy": global_pattern.global_free_energy,
            "mean_similarity": global_pattern.mean_similarity(),
        }

    def run(
        self,
        observations: List[float],
        n_interactions_per_step: int = 1,
        consolidate_every: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Run the collective on a sequence of observations.

        Returns list of step summaries.
        """
        results = []

        for i, obs in enumerate(observations):
            result = self.step(obs, n_interactions_per_step)
            results.append(result)

            if (i + 1) % consolidate_every == 0:
                self.consolidate()

        return results

    def get_hierarchy_summary(self) -> Dict[str, Any]:
        """
        Get summary of the complete hierarchy.

        Shows z-levels and structure at each layer.
        """
        summary = {
            "meta_collective": {
                "id": self.collective_id,
                "z_level": self.z_level,
                "free_energy": self.free_energy,
                "coherence": self.coherence,
                "n_triads": len(self.triads),
            },
            "triads": {},
        }

        for tid, triad in self.triads.items():
            triad_summary = {
                "z_level": triad.z_level,
                "free_energy": triad.free_energy,
                "coherence": triad.coherence,
                "n_tools": len(triad.tools),
                "tools": [],
            }

            for tool in triad.tools:
                tool_summary = {
                    "id": tool.tool_id,
                    "z_level": tool.z_level,
                    "free_energy": tool.free_energy,
                    "coherence": tool.coherence,
                    "internal_model": {
                        "z_level": tool.internal_model.z_level,
                        "kappa_amplitude": tool.internal_model.dual_field.kappa.amplitude,
                        "lambda_amplitude": tool.internal_model.dual_field.lambda_field.amplitude,
                    },
                }
                triad_summary["tools"].append(tool_summary)

            summary["triads"][tid] = triad_summary

        return summary

    def snapshot(self) -> Dict[str, Any]:
        """Return complete Meta-Collective state snapshot."""
        return {
            "collective_id": self.collective_id,
            "z_level": self.z_level,
            "state": self.state.value,
            "n_triads": len(self.triads),
            "triads": {tid: triad.snapshot() for tid, triad in self.triads.items()},
            "triad_weights": self._triad_weights,
            "free_energy": self.free_energy,
            "coherence": self.coherence,
            "global_pattern": {
                "global_coherence": self._global_pattern.global_coherence if self._global_pattern else None,
                "mean_similarity": self._global_pattern.mean_similarity() if self._global_pattern else None,
            } if self._global_pattern else None,
            "emergent_properties": {
                name: {
                    "value": prop.value,
                    "emergence_level": prop.emergence_level,
                }
                for name, prop in self._emergent_properties.items()
            },
            "total_cycles": self._total_cycles,
            "interaction_count": self._interaction_count,
            "mean_free_energy": self._cumulative_free_energy / max(1, self._total_cycles),
        }

    def reset(self) -> None:
        """Reset Meta-Collective to initial state."""
        for triad in self.triads.values():
            triad.reset()
        self._global_pattern = None
        self._emergent_properties.clear()
        self._current_interaction_idx = 0
        self._total_cycles = 0
        self._cumulative_free_energy = 0.0
        self._interaction_count = 0
        self._update_weights()
        self.state = CollectiveState.ACTIVE


# Factory function for standard configuration
def create_standard_collective(
    collective_id: Optional[str] = None,
    n_triads: int = 2,
    n_tools: int = 3
) -> MetaCollective:
    """
    Create a standard Meta-Collective configuration.

    Default: 2 Triads (A, B), each with 3 Tools.
    """
    return MetaCollective(
        collective_id=collective_id,
        n_triads=n_triads,
        n_tools_per_triad=n_tools,
    )
