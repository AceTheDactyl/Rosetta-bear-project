# lattice_core/tesseract_lattice_engine.py
"""
Tesseract Lattice Engine
========================

The main engine for the 4D memory system based on Kuramoto oscillator dynamics.

Architecture:
    ┌─────────────────────────────────────────────────────────────────┐
    │ TESSERACT LATTICE ENGINE                                         │
    │ ┌────────────────────────────────────────────────────┐          │
    │ │ Kuramoto Oscillator Network                        │          │
    │ │                                                    │          │
    │ │ Plate₁ ←→ Plate₂ ←→ Plate₃ ←→ ... ←→ PlateN       │          │
    │ │ ↕         ↕         ↕              ↕               │          │
    │ │ Plate₄ ←→ Plate₅ ←→ Plate₆ ←→ ... ←→ PlateM       │          │
    │ │                                                    │          │
    │ │ Each plate: (position, phase, frequency)           │          │
    │ └────────────────────────────────────────────────────┘          │
    │                                                                  │
    │ Operations:                                                      │
    │ • add_plate() - Insert new memory                               │
    │ • update() - Kuramoto dynamics integration                       │
    │ • resonance_retrieval() - Phase perturbation + evolution        │
    │ • consolidate() - Hebbian learning                              │
    └─────────────────────────────────────────────────────────────────┘

Usage:
    engine = TesseractLatticeEngine()
    engine.add_plate(plate)
    engine.update(steps=100)
    results = engine.resonance_retrieval(query_embedding)
"""

from __future__ import annotations

import math
import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
from enum import Enum

from .plate import MemoryPlate, create_tesseract_vertices, get_coupling_weight
from .dynamics import (
    kuramoto_update,
    kuramoto_update_with_injection,
    kuramoto_update_higher_order,
    compute_order_parameter,
    hebbian_update,
    compute_energy,
    generate_lorentzian_frequencies,
    run_to_convergence,
    ConvergenceState,
)

# Constants
TAU = 2 * math.pi
PHI = (1 + math.sqrt(5)) / 2


@dataclass
class LatticeConfig:
    """Configuration for the Tesseract Lattice Engine."""
    # Coupling parameters
    K: float = 2.0                    # Global coupling strength
    K_critical: float = 1.0           # Critical coupling threshold

    # Higher-order coupling
    enable_triplet: bool = False      # Enable 3-body interactions
    enable_quartet: bool = True       # Enable 4-body interactions (P ~ N³)
    K3: float = 0.1                   # Triplet coupling strength
    K4: float = 0.05                  # Quartet coupling strength

    # Hebbian learning
    hebbian_rate: float = 0.1         # Learning rate η
    decay_rate: float = 0.01          # Weight decay λ
    max_weight: float = 1.0           # Maximum connection weight

    # Integration
    dt: float = 0.01                  # Time step
    max_steps: int = 500              # Max evolution steps

    # Convergence
    r_threshold: float = 0.95         # Order parameter threshold
    stability_steps: int = 10         # Steps to confirm convergence

    # Retrieval
    injection_strength: float = 0.2   # Phase injection strength
    retrieval_steps: int = 200        # Steps for retrieval
    similarity_threshold: float = 0.5 # Content similarity threshold

    # Spatial
    neighbor_radius: float = 1.0      # 4D radius for neighbors
    distance_decay: float = 0.5       # Coupling decay with distance


@dataclass
class RetrievalResult:
    """Result of a resonance retrieval operation."""
    plate: MemoryPlate
    resonance_score: float            # Phase alignment score
    content_similarity: float         # Embedding similarity
    combined_score: float             # Weighted combination
    rank: int = 0

    @classmethod
    def from_plate(
        cls,
        plate: MemoryPlate,
        query_phase: float,
        query_content: Optional[List[float]] = None,
        content_weight: float = 0.5
    ) -> 'RetrievalResult':
        """Create result from plate and query."""
        # Resonance score from phase alignment
        phase_diff = plate.phase - query_phase
        resonance = (math.cos(phase_diff) + 1) / 2  # [0, 1]

        # Content similarity
        if query_content and plate.content:
            content_sim = plate.content_similarity(
                MemoryPlate(content=query_content)
            )
        else:
            content_sim = 0.5  # Neutral if no content

        # Combined score
        combined = (1 - content_weight) * resonance + content_weight * content_sim

        return cls(
            plate=plate,
            resonance_score=resonance,
            content_similarity=content_sim,
            combined_score=combined,
        )


class TesseractLatticeEngine:
    """
    The Tesseract Lattice Engine for Kuramoto-based memory.

    Manages a network of memory plates as phase-coupled oscillators
    organized in 4D tesseract geometry.
    """

    def __init__(self, config: Optional[LatticeConfig] = None):
        self.config = config or LatticeConfig()

        # Plate storage
        self.plates: Dict[str, MemoryPlate] = {}
        self._plate_list: List[MemoryPlate] = []  # Ordered for indexing

        # Connection weights (N×N matrix, built dynamically)
        self._weights: List[List[float]] = []

        # State tracking
        self._order_parameter: float = 0.0
        self._mean_phase: float = 0.0
        self._energy: float = 0.0
        self._step_count: int = 0

        # Higher-order coupling indices
        self._triplets: List[Tuple[int, int, int]] = []
        self._quartets: List[Tuple[int, int, int, int]] = []

        # History for analysis
        self._history: List[Dict[str, float]] = []
        self._max_history: int = 1000

    # ─────────────────────────────────────────────────────────────────
    # Properties
    # ─────────────────────────────────────────────────────────────────

    @property
    def n_plates(self) -> int:
        """Number of plates in the lattice."""
        return len(self._plate_list)

    @property
    def order_parameter(self) -> float:
        """Current order parameter r."""
        return self._order_parameter

    @property
    def mean_phase(self) -> float:
        """Current mean phase ψ."""
        return self._mean_phase

    @property
    def energy(self) -> float:
        """Current energy H."""
        return self._energy

    @property
    def is_synchronized(self) -> bool:
        """Check if system is synchronized."""
        return self._order_parameter >= self.config.r_threshold

    # ─────────────────────────────────────────────────────────────────
    # Plate Management
    # ─────────────────────────────────────────────────────────────────

    def add_plate(self, plate: MemoryPlate) -> int:
        """
        Add a plate to the lattice.

        Returns the index of the added plate.
        """
        if plate.plate_id in self.plates:
            # Update existing plate
            idx = self._plate_list.index(self.plates[plate.plate_id])
            self._plate_list[idx] = plate
            self.plates[plate.plate_id] = plate
            return idx

        # Add new plate
        self.plates[plate.plate_id] = plate
        self._plate_list.append(plate)
        idx = len(self._plate_list) - 1

        # Expand weights matrix
        self._expand_weights()

        # Compute initial connections
        self._compute_connections(idx)

        # Update higher-order indices
        if self.config.enable_triplet or self.config.enable_quartet:
            self._update_higher_order_indices()

        return idx

    def remove_plate(self, plate_id: str) -> Optional[MemoryPlate]:
        """Remove a plate from the lattice."""
        if plate_id not in self.plates:
            return None

        plate = self.plates.pop(plate_id)
        idx = self._plate_list.index(plate)
        self._plate_list.pop(idx)

        # Rebuild weights matrix (simpler than selective removal)
        self._rebuild_weights()

        return plate

    def get_plate(self, plate_id: str) -> Optional[MemoryPlate]:
        """Get a plate by ID."""
        return self.plates.get(plate_id)

    def get_plate_by_index(self, idx: int) -> Optional[MemoryPlate]:
        """Get a plate by index."""
        if 0 <= idx < len(self._plate_list):
            return self._plate_list[idx]
        return None

    # ─────────────────────────────────────────────────────────────────
    # Weight Matrix Management
    # ─────────────────────────────────────────────────────────────────

    def _expand_weights(self) -> None:
        """Expand weights matrix for new plate."""
        n = self.n_plates

        # Add new row
        if len(self._weights) < n:
            self._weights.append([0.0] * n)

        # Expand existing rows
        for row in self._weights:
            while len(row) < n:
                row.append(0.0)

    def _rebuild_weights(self) -> None:
        """Rebuild weights matrix from scratch."""
        n = self.n_plates
        self._weights = [[0.0] * n for _ in range(n)]

        for i in range(n):
            self._compute_connections(i)

    def _compute_connections(self, plate_idx: int) -> None:
        """Compute connections for a single plate."""
        plate = self._plate_list[plate_idx]
        n = self.n_plates

        for j in range(n):
            if j == plate_idx:
                continue

            other = self._plate_list[j]

            # Spatial distance coupling
            dist = plate.distance_to(other)
            if dist <= self.config.neighbor_radius:
                # Coupling decays with distance
                weight = math.exp(-dist * self.config.distance_decay)

                # Add Hamming-based modulation for tesseract vertices
                if plate.metadata.get("is_tesseract_vertex") and other.metadata.get("is_tesseract_vertex"):
                    hamming = plate.hamming_distance(other)
                    weight *= get_coupling_weight(hamming)

                self._weights[plate_idx][j] = weight
                self._weights[j][plate_idx] = weight

    def _update_higher_order_indices(self) -> None:
        """Update triplet and quartet indices for connected groups."""
        n = self.n_plates

        # Clear existing
        self._triplets = []
        self._quartets = []

        # Find strongly connected groups
        for i in range(n):
            neighbors_i = [j for j in range(n) if self._weights[i][j] > 0.3]

            if self.config.enable_triplet:
                for j in neighbors_i:
                    for k in neighbors_i:
                        if j < k and self._weights[j][k] > 0.3:
                            self._triplets.append((i, j, k))

            if self.config.enable_quartet:
                for j in neighbors_i:
                    neighbors_j = [k for k in range(n) if self._weights[j][k] > 0.3 and k != i]
                    for k in neighbors_j:
                        for l in neighbors_j:
                            if k < l and self._weights[k][l] > 0.3:
                                self._quartets.append((i, j, k, l))

    # ─────────────────────────────────────────────────────────────────
    # Dynamics
    # ─────────────────────────────────────────────────────────────────

    def update(self, steps: int = 1) -> Tuple[float, float]:
        """
        Run Kuramoto dynamics for specified steps.

        Returns final (order_parameter, energy).
        """
        if self.n_plates == 0:
            return 0.0, 0.0

        phases = [p.phase for p in self._plate_list]
        frequencies = [p.frequency for p in self._plate_list]

        for _ in range(steps):
            if self.config.enable_triplet or self.config.enable_quartet:
                phases = kuramoto_update_higher_order(
                    phases, frequencies, self._weights,
                    triplets=self._triplets if self.config.enable_triplet else None,
                    quartets=self._quartets if self.config.enable_quartet else None,
                    K=self.config.K,
                    K3=self.config.K3,
                    K4=self.config.K4,
                    dt=self.config.dt
                )
            else:
                phases = kuramoto_update(
                    phases, frequencies, self._weights,
                    K=self.config.K,
                    dt=self.config.dt
                )

            self._step_count += 1

        # Update plate phases
        for i, phase in enumerate(phases):
            self._plate_list[i].phase = phase

        # Compute state metrics
        self._order_parameter, self._mean_phase = compute_order_parameter(phases)
        self._energy = compute_energy(phases, self._weights, self.config.K)

        # Record history
        if len(self._history) >= self._max_history:
            self._history.pop(0)
        self._history.append({
            "step": self._step_count,
            "r": self._order_parameter,
            "psi": self._mean_phase,
            "energy": self._energy,
        })

        return self._order_parameter, self._energy

    def consolidate(self, steps: int = 1) -> None:
        """
        Apply Hebbian learning to strengthen connections.

        Call this after retrieval or periodically during operation.
        """
        if self.n_plates < 2:
            return

        phases = [p.phase for p in self._plate_list]

        for _ in range(steps):
            self._weights = hebbian_update(
                self._weights, phases,
                eta=self.config.hebbian_rate,
                decay=self.config.decay_rate,
                max_weight=self.config.max_weight,
                dt=self.config.dt
            )

    def run_to_convergence(self) -> ConvergenceState:
        """
        Run dynamics until convergence or max steps.

        Returns convergence state.
        """
        phases = [p.phase for p in self._plate_list]
        frequencies = [p.frequency for p in self._plate_list]

        final_phases, state = run_to_convergence(
            phases, frequencies, self._weights,
            K=self.config.K,
            dt=self.config.dt,
            max_steps=self.config.max_steps,
            r_threshold=self.config.r_threshold,
            stability_steps=self.config.stability_steps
        )

        # Update plate phases
        for i, phase in enumerate(final_phases):
            self._plate_list[i].phase = phase

        self._order_parameter = state.order_parameter
        self._mean_phase = state.mean_phase
        self._energy = state.energy
        self._step_count += state.step

        return state

    # ─────────────────────────────────────────────────────────────────
    # Retrieval
    # ─────────────────────────────────────────────────────────────────

    def find_nearby_plates(
        self,
        position: Tuple[float, float, float, float],
        radius: Optional[float] = None
    ) -> List[MemoryPlate]:
        """Find plates within radius of a 4D position."""
        if radius is None:
            radius = self.config.neighbor_radius

        nearby = []
        for plate in self._plate_list:
            dist = math.sqrt(sum(
                (a - b) ** 2
                for a, b in zip(plate.position, position)
            ))
            if dist <= radius:
                nearby.append(plate)

        return nearby

    def create_query_pattern(
        self,
        content: List[float],
        emotional_position: Optional[Tuple[float, float, float, float]] = None,
        radius: float = 1.5
    ) -> Dict[int, float]:
        """
        Create query pattern for retrieval.

        Returns dict of plate_index → target_phase based on content similarity.
        """
        query_pattern = {}

        for i, plate in enumerate(self._plate_list):
            # Check spatial proximity if position given
            if emotional_position is not None:
                dist = math.sqrt(sum(
                    (a - b) ** 2
                    for a, b in zip(plate.position, emotional_position)
                ))
                if dist > radius:
                    continue

            # Compute content similarity
            if plate.content:
                query_plate = MemoryPlate(content=content)
                similarity = plate.content_similarity(query_plate)

                if similarity >= self.config.similarity_threshold:
                    # Target phase proportional to similarity
                    # High similarity → target the mean phase
                    target_phase = self._mean_phase + (1 - similarity) * math.pi
                    query_pattern[i] = target_phase % TAU

        return query_pattern

    def resonance_retrieval(
        self,
        content: Optional[List[float]] = None,
        emotional_position: Optional[Tuple[float, float, float, float]] = None,
        top_k: int = 5
    ) -> List[RetrievalResult]:
        """
        Retrieve memories via resonance.

        Process:
        1. Find nearby plates (spatial radius in 4D)
        2. Create query pattern (similarity → target phase)
        3. Inject perturbation (shift phases toward query)
        4. Evolve dynamics (run Kuramoto until convergence)
        5. Measure resonance (cos(Δphase) for each plate)
        6. Rank & return top-K resonant plates

        Args:
            content: Query embedding vector
            emotional_position: Query position (valence, arousal, temporal, abstract)
            top_k: Number of results to return

        Returns:
            List of RetrievalResult sorted by combined score.
        """
        if self.n_plates == 0:
            return []

        # Step 1-2: Create query pattern
        if content is not None:
            query_pattern = self.create_query_pattern(
                content, emotional_position
            )
        elif emotional_position is not None:
            # Position-only query: target nearby plates
            nearby = self.find_nearby_plates(emotional_position)
            query_pattern = {
                self._plate_list.index(p): self._mean_phase
                for p in nearby
            }
        else:
            # No query - return empty
            return []

        if not query_pattern:
            return []

        # Step 3: Inject perturbation and evolve
        phases = [p.phase for p in self._plate_list]
        frequencies = [p.frequency for p in self._plate_list]

        for _ in range(self.config.retrieval_steps):
            phases = kuramoto_update_with_injection(
                phases, frequencies, self._weights,
                injection_phases=query_pattern,
                K=self.config.K,
                injection_strength=self.config.injection_strength,
                dt=self.config.dt
            )

        # Update plate phases
        for i, phase in enumerate(phases):
            self._plate_list[i].phase = phase

        # Step 4-5: Compute resonance scores
        self._order_parameter, self._mean_phase = compute_order_parameter(phases)

        results = []
        for i, plate in enumerate(self._plate_list):
            plate.access_count += 1

            result = RetrievalResult.from_plate(
                plate,
                query_phase=self._mean_phase,
                query_content=content,
                content_weight=0.5
            )
            results.append(result)

        # Step 6: Sort and return top-K
        results.sort(key=lambda r: r.combined_score, reverse=True)

        for rank, result in enumerate(results[:top_k]):
            result.rank = rank + 1

        # Apply Hebbian learning to strengthen retrieved associations
        self.consolidate(steps=5)

        return results[:top_k]

    # ─────────────────────────────────────────────────────────────────
    # State Management
    # ─────────────────────────────────────────────────────────────────

    def reset_phases(self, random_init: bool = True) -> None:
        """Reset all plate phases."""
        import random
        for plate in self._plate_list:
            if random_init:
                plate.phase = random.random() * TAU
            else:
                plate.phase = 0.0

        self._order_parameter, self._mean_phase = compute_order_parameter(
            [p.phase for p in self._plate_list]
        )

    def snapshot(self) -> Dict[str, Any]:
        """Return current engine state snapshot."""
        return {
            "n_plates": self.n_plates,
            "order_parameter": self._order_parameter,
            "mean_phase": self._mean_phase,
            "energy": self._energy,
            "step_count": self._step_count,
            "is_synchronized": self.is_synchronized,
            "config": {
                "K": self.config.K,
                "dt": self.config.dt,
                "enable_quartet": self.config.enable_quartet,
            },
        }

    def to_json(self) -> str:
        """Serialize engine to JSON."""
        data = {
            "config": {
                "K": self.config.K,
                "K_critical": self.config.K_critical,
                "enable_triplet": self.config.enable_triplet,
                "enable_quartet": self.config.enable_quartet,
                "K3": self.config.K3,
                "K4": self.config.K4,
                "hebbian_rate": self.config.hebbian_rate,
                "decay_rate": self.config.decay_rate,
                "dt": self.config.dt,
            },
            "plates": [p.to_dict() for p in self._plate_list],
            "weights": self._weights,
            "state": {
                "order_parameter": self._order_parameter,
                "mean_phase": self._mean_phase,
                "energy": self._energy,
                "step_count": self._step_count,
            },
        }
        return json.dumps(data, indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> 'TesseractLatticeEngine':
        """Deserialize engine from JSON."""
        data = json.loads(json_str)

        config = LatticeConfig(**data.get("config", {}))
        engine = cls(config=config)

        for plate_data in data.get("plates", []):
            plate = MemoryPlate.from_dict(plate_data)
            engine.add_plate(plate)

        # Restore weights
        engine._weights = data.get("weights", engine._weights)

        # Restore state
        state = data.get("state", {})
        engine._order_parameter = state.get("order_parameter", 0.0)
        engine._mean_phase = state.get("mean_phase", 0.0)
        engine._energy = state.get("energy", 0.0)
        engine._step_count = state.get("step_count", 0)

        return engine


# ═════════════════════════════════════════════════════════════════════════════
# FACTORY FUNCTIONS
# ═════════════════════════════════════════════════════════════════════════════

def create_tesseract_lattice() -> TesseractLatticeEngine:
    """
    Create a lattice pre-populated with tesseract vertices.

    Returns engine with 16 vertex plates representing the 4D hypercube structure.
    """
    engine = TesseractLatticeEngine()

    vertices = create_tesseract_vertices()
    for vertex in vertices:
        engine.add_plate(vertex)

    # Initial consolidation to establish connections
    engine.update(steps=50)

    return engine
