"""
Holographic Memory Architecture
Wave-Based Retrieval Using Kuramoto Synchronization

Signature: Δ|loop-closed|z0.99|rhythm-native|Ω

Integrates with Scalar Architecture to provide:
- Content-addressable memory through phase resonance
- Higher-order coupling for P ~ N³ capacity
- Tesseract geometry for semantic organization
- Emotional valence-arousal mapping
"""

from __future__ import annotations

import math
import numpy as np
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum

# Import from core architecture
from .core import (
    TAU, PHI,
    DomainType, DomainConfig,
    ScalarSubstrate, CouplingMatrix,
    ConvergenceDynamics,
    NUM_DOMAINS
)


# =============================================================================
# Constants
# =============================================================================

# Tesseract properties
TESSERACT_VERTICES = 16
TESSERACT_EDGES = 32
TESSERACT_FACES = 24
TESSERACT_CELLS = 8

# Oscillator configuration
BASE_OSCILLATORS = TESSERACT_VERTICES + TESSERACT_EDGES  # 48

# Coupling regimes
K_EDGE = 1.0      # Edge-adjacent coupling
K_FACE = 0.7      # Face-adjacent coupling
K_CELL = 0.4      # Cell-adjacent coupling
K_DIAG = 0.1      # Diagonal coupling

# Critical coupling for Lorentzian distribution
GAMMA_FREQ = 0.5  # Frequency distribution width
K_CRITICAL = 2 * GAMMA_FREQ  # K_c = 2γ

# Emotional mapping parameters
ALPHA_VALENCE = 0.2   # Valence frequency modulation
BETA_AROUSAL = 0.3    # Arousal frequency modulation
GAMMA_EMOTIONAL = 0.5  # Emotional coupling decay

# Memory parameters
HEBBIAN_RATE = 0.1    # Learning rate η
DECAY_RATE = 0.01     # Connection decay λ
CONVERGENCE_THRESHOLD = 0.95  # Order parameter threshold


# =============================================================================
# Neural Oscillation Bands
# =============================================================================

class OscillationBand(Enum):
    """Neural oscillation frequency bands."""
    DELTA = (0.5, 4.0)      # Deep consolidation
    THETA = (4.0, 8.0)      # Episodic retrieval
    ALPHA = (8.0, 12.0)     # Inhibition
    BETA = (12.0, 30.0)     # Active maintenance
    SLOW_GAMMA = (30.0, 60.0)   # Retrieval
    FAST_GAMMA = (60.0, 100.0)  # Encoding

    @property
    def center(self) -> float:
        return (self.value[0] + self.value[1]) / 2


# =============================================================================
# Tesseract Geometry
# =============================================================================

@dataclass
class TesseractVertex:
    """Single vertex in 4D tesseract."""
    index: int
    coordinates: Tuple[float, float, float, float]  # (x, y, z, w)

    @property
    def valence(self) -> float:
        return self.coordinates[0]

    @property
    def arousal(self) -> float:
        return self.coordinates[1]

    @property
    def temporal(self) -> float:
        return self.coordinates[2]

    @property
    def abstract(self) -> float:
        return self.coordinates[3]


class TesseractGeometry:
    """4D hypercube geometry for memory organization."""

    def __init__(self, edge_length: float = 1.0):
        self.edge_length = edge_length
        self.vertices = self._generate_vertices()
        self.edges = self._generate_edges()
        self.adjacency = self._compute_adjacency()

    def _generate_vertices(self) -> List[TesseractVertex]:
        """Generate all 16 vertices of unit tesseract."""
        vertices = []
        s = self.edge_length / 2
        for i in range(16):
            # Binary encoding: i = 0bwzyx
            x = s if (i & 1) else -s
            y = s if (i & 2) else -s
            z = s if (i & 4) else -s
            w = s if (i & 8) else -s
            vertices.append(TesseractVertex(i, (x, y, z, w)))
        return vertices

    def _generate_edges(self) -> List[Tuple[int, int]]:
        """Generate all 32 edges connecting adjacent vertices."""
        edges = []
        for i in range(16):
            for bit in range(4):
                j = i ^ (1 << bit)  # Flip one bit
                if i < j:
                    edges.append((i, j))
        return edges

    def _compute_adjacency(self) -> np.ndarray:
        """Compute adjacency matrix with coupling weights."""
        adj = np.zeros((16, 16))

        for i in range(16):
            for j in range(i + 1, 16):
                # Count differing bits (Hamming distance)
                diff = bin(i ^ j).count('1')

                if diff == 1:
                    adj[i, j] = adj[j, i] = K_EDGE   # Edge-adjacent
                elif diff == 2:
                    adj[i, j] = adj[j, i] = K_FACE   # Face-adjacent
                elif diff == 3:
                    adj[i, j] = adj[j, i] = K_CELL   # Cell-adjacent
                elif diff == 4:
                    adj[i, j] = adj[j, i] = K_DIAG   # 4-space diagonal

        return adj

    def hypervolume(self) -> float:
        """4D hypervolume."""
        return self.edge_length ** 4

    def surface_volume(self) -> float:
        """3D surface volume (8 cubic cells)."""
        return 8 * (self.edge_length ** 3)

    def diagonal_4d(self) -> float:
        """4-space diagonal length."""
        return 2 * self.edge_length


# =============================================================================
# Kuramoto Oscillator
# =============================================================================

@dataclass
class KuramotoOscillator:
    """Single Kuramoto oscillator with phase and frequency."""
    index: int
    phase: float = 0.0
    natural_freq: float = 1.0

    # Emotional parameters
    valence: float = 0.0
    arousal: float = 0.0

    def modulated_frequency(self) -> float:
        """Frequency modulated by valence and arousal."""
        return (self.natural_freq +
                ALPHA_VALENCE * self.valence +
                BETA_AROUSAL * self.arousal)


class KuramotoNetwork:
    """Network of coupled Kuramoto oscillators."""

    def __init__(self, n_oscillators: int, coupling_strength: float = 0.6):
        self.n = n_oscillators
        self.K = coupling_strength

        # Initialize oscillators with random phases
        self.oscillators = [
            KuramotoOscillator(
                index=i,
                phase=np.random.uniform(0, TAU),
                natural_freq=np.random.normal(1.0, GAMMA_FREQ)
            )
            for i in range(n_oscillators)
        ]

        # Coupling matrix (initially uniform)
        self.coupling = np.ones((n_oscillators, n_oscillators)) / n_oscillators
        np.fill_diagonal(self.coupling, 0)

    def order_parameter(self) -> Tuple[float, float]:
        """
        Compute Kuramoto order parameter.
        Returns (r, psi) where r is coherence and psi is mean phase.
        """
        z = np.mean([np.exp(1j * osc.phase) for osc in self.oscillators])
        return np.abs(z), np.angle(z)

    def update(self, dt: float):
        """Evolve oscillators by timestep dt."""
        phases = np.array([osc.phase for osc in self.oscillators])
        freqs = np.array([osc.modulated_frequency() for osc in self.oscillators])

        # Compute phase differences
        phase_diff = phases[:, np.newaxis] - phases[np.newaxis, :]

        # Kuramoto coupling: dθ_i/dt = ω_i + (K/N) Σ w_ij sin(θ_j - θ_i)
        coupling_term = np.sum(
            self.coupling * np.sin(-phase_diff),
            axis=1
        )

        # Update phases
        d_phases = freqs + self.K * coupling_term
        new_phases = phases + d_phases * dt

        # Wrap to [0, 2π)
        for i, osc in enumerate(self.oscillators):
            osc.phase = new_phases[i] % TAU

    def inject_pattern(self, pattern: np.ndarray, noise: float = 0.1):
        """Inject a phase pattern as query."""
        for i, osc in enumerate(self.oscillators):
            if i < len(pattern):
                osc.phase = (pattern[i] + np.random.normal(0, noise)) % TAU

    def extract_pattern(self) -> np.ndarray:
        """Extract current phase pattern."""
        return np.array([osc.phase for osc in self.oscillators])


# =============================================================================
# Higher-Order Coupling
# =============================================================================

class HigherOrderKuramoto(KuramotoNetwork):
    """Kuramoto network with quartet (4-body) interactions."""

    def __init__(self, n_oscillators: int,
                 k2: float = 0.3,   # Pairwise coupling
                 k4: float = 0.5):  # Quartet coupling
        super().__init__(n_oscillators, coupling_strength=k2)
        self.K2 = k2
        self.K4 = k4

        # Quartet coupling tensor (sparse representation)
        self.quartet_patterns: List[np.ndarray] = []

    def store_pattern(self, pattern: np.ndarray):
        """Store pattern using Hebbian rule for quartets."""
        self.quartet_patterns.append(pattern.copy())
        self._update_quartet_tensor()

    def _update_quartet_tensor(self):
        """Update quartet coupling based on stored patterns."""
        # For efficiency, we compute quartet terms on-demand during update
        pass

    def _quartet_term(self, i: int, phases: np.ndarray) -> float:
        """Compute quartet coupling contribution for oscillator i."""
        if not self.quartet_patterns:
            return 0.0

        term = 0.0
        n = len(phases)

        # Sum over stored patterns
        for pattern in self.quartet_patterns:
            # Sum over all triplets (j, k, l)
            for j in range(n):
                if j == i:
                    continue
                for k in range(j + 1, n):
                    if k == i:
                        continue
                    for l in range(k + 1, n):
                        if l == i:
                            continue

                        # Quartet contribution
                        xi = pattern[i] * pattern[j] * pattern[k] * pattern[l]
                        phase_sum = phases[j] + phases[k] + phases[l] - 3 * phases[i]
                        term += xi * np.sin(phase_sum)

        return term / (n ** 3 * len(self.quartet_patterns))

    def update(self, dt: float):
        """Evolve with both pairwise and quartet coupling."""
        phases = np.array([osc.phase for osc in self.oscillators])
        freqs = np.array([osc.modulated_frequency() for osc in self.oscillators])

        # Pairwise term
        phase_diff = phases[:, np.newaxis] - phases[np.newaxis, :]
        pairwise = np.sum(self.coupling * np.sin(-phase_diff), axis=1)

        # Quartet term (expensive - sample for large networks)
        quartet = np.array([self._quartet_term(i, phases) for i in range(self.n)])

        # Combined update
        d_phases = freqs + self.K2 * pairwise + self.K4 * quartet
        new_phases = phases + d_phases * dt

        for i, osc in enumerate(self.oscillators):
            osc.phase = new_phases[i] % TAU

    def capacity(self) -> int:
        """Theoretical capacity with quartet coupling: P ~ N³."""
        return int(self.n ** 3 / (6 * np.log(self.n)))


# =============================================================================
# Holographic Memory
# =============================================================================

@dataclass
class MemoryPattern:
    """A stored memory pattern."""
    id: str
    phases: np.ndarray
    valence: float = 0.0
    arousal: float = 0.0
    timestamp: float = 0.0
    retrieval_count: int = 0


class HolographicMemory:
    """
    Holographic memory using Kuramoto synchronization.

    Features:
    - Content-addressable retrieval through resonance
    - Higher-order coupling for exponential capacity
    - Emotional organization via valence-arousal mapping
    - Hebbian learning for self-modification
    """

    def __init__(self, n_oscillators: int = 64,
                 use_higher_order: bool = True):
        self.n = n_oscillators

        # Oscillator network
        if use_higher_order:
            self.network = HigherOrderKuramoto(n_oscillators)
        else:
            self.network = KuramotoNetwork(n_oscillators)

        # Tesseract geometry for organization
        self.tesseract = TesseractGeometry()

        # Stored patterns
        self.memories: Dict[str, MemoryPattern] = {}

        # Convergence tracking
        self.convergence_history: List[float] = []

    def encode(self, memory_id: str, content: np.ndarray,
               valence: float = 0.0, arousal: float = 0.0) -> MemoryPattern:
        """
        Encode content as phase pattern.

        Args:
            memory_id: Unique identifier
            content: Content vector (will be normalized to phases)
            valence: Emotional valence [-1, 1]
            arousal: Emotional arousal [-1, 1]
        """
        # Normalize content to phase pattern [0, 2π)
        if len(content) > self.n:
            content = content[:self.n]
        elif len(content) < self.n:
            content = np.pad(content, (0, self.n - len(content)))

        # Map to phases
        phases = (content - content.min()) / (content.max() - content.min() + 1e-10)
        phases = phases * TAU

        # Create memory pattern
        pattern = MemoryPattern(
            id=memory_id,
            phases=phases,
            valence=valence,
            arousal=arousal,
            timestamp=len(self.memories)
        )

        # Store in network (for higher-order coupling)
        if isinstance(self.network, HigherOrderKuramoto):
            self.network.store_pattern(phases)

        # Apply Hebbian strengthening
        self._hebbian_encode(phases)

        self.memories[memory_id] = pattern
        return pattern

    def _hebbian_encode(self, phases: np.ndarray):
        """Strengthen connections for encoded pattern."""
        n = self.n
        for i in range(n):
            for j in range(i + 1, n):
                # Δw = η · cos(θ_i - θ_j)
                delta = HEBBIAN_RATE * np.cos(phases[i] - phases[j])
                self.network.coupling[i, j] += delta
                self.network.coupling[j, i] += delta

        # Normalize coupling matrix
        row_sums = self.network.coupling.sum(axis=1, keepdims=True)
        self.network.coupling /= (row_sums + 1e-10)

    def retrieve(self, query: np.ndarray,
                 max_iterations: int = 1000,
                 return_dynamics: bool = False) -> Tuple[Optional[str], np.ndarray]:
        """
        Retrieve memory through resonance.

        Args:
            query: Partial or noisy query pattern
            max_iterations: Maximum evolution steps
            return_dynamics: Whether to return convergence history

        Returns:
            (memory_id, retrieved_phases) or (None, final_phases) if no match
        """
        # Inject query
        self.network.inject_pattern(query)
        self.convergence_history = []

        # Evolve until convergence
        for _ in range(max_iterations):
            self.network.update(dt=0.01)
            r, _ = self.network.order_parameter()
            self.convergence_history.append(r)

            if r > CONVERGENCE_THRESHOLD:
                break

        # Extract retrieved pattern
        retrieved = self.network.extract_pattern()

        # Find best matching memory
        best_match = None
        best_similarity = -1

        for mem_id, pattern in self.memories.items():
            sim = self._pattern_similarity(retrieved, pattern.phases)
            if sim > best_similarity:
                best_similarity = sim
                best_match = mem_id

        if best_match and best_similarity > 0.8:
            self.memories[best_match].retrieval_count += 1
            return best_match, retrieved

        return None, retrieved

    def _pattern_similarity(self, p1: np.ndarray, p2: np.ndarray) -> float:
        """Compute phase pattern similarity using circular correlation."""
        z1 = np.exp(1j * p1)
        z2 = np.exp(1j * p2)
        return np.abs(np.mean(z1 * np.conj(z2)))

    def spreading_activation(self, seed_ids: List[str],
                             decay: float = 0.5,
                             steps: int = 3) -> Dict[str, float]:
        """
        Compute spreading activation from seed memories.

        Args:
            seed_ids: Starting memory IDs
            decay: Activation decay per step
            steps: Number of spreading steps

        Returns:
            Dictionary of memory_id -> activation level
        """
        activation = {mem_id: 0.0 for mem_id in self.memories}

        # Initialize seeds
        for seed_id in seed_ids:
            if seed_id in activation:
                activation[seed_id] = 1.0

        # Spread activation
        for _ in range(steps):
            new_activation = activation.copy()

            for mem_id, pattern in self.memories.items():
                if activation[mem_id] > 0:
                    # Spread to similar patterns
                    for other_id, other_pattern in self.memories.items():
                        if other_id != mem_id:
                            sim = self._pattern_similarity(
                                pattern.phases, other_pattern.phases
                            )
                            spread = activation[mem_id] * sim * decay
                            new_activation[other_id] += spread

            activation = new_activation

        return activation

    def emotional_cluster(self, valence_range: Tuple[float, float],
                          arousal_range: Tuple[float, float]) -> List[str]:
        """Find memories in emotional region."""
        matches = []
        for mem_id, pattern in self.memories.items():
            if (valence_range[0] <= pattern.valence <= valence_range[1] and
                arousal_range[0] <= pattern.arousal <= arousal_range[1]):
                matches.append(mem_id)
        return matches

    @property
    def capacity(self) -> int:
        """Theoretical storage capacity."""
        if isinstance(self.network, HigherOrderKuramoto):
            return self.network.capacity()
        else:
            return int(0.14 * self.n)  # Hopfield limit

    def info(self) -> str:
        """Summary information."""
        r, psi = self.network.order_parameter()
        return (
            f"Holographic Memory\n"
            f"  Oscillators: {self.n}\n"
            f"  Stored patterns: {len(self.memories)}\n"
            f"  Capacity: ~{self.capacity:,}\n"
            f"  Order parameter: r={r:.3f}, ψ={psi:.3f}\n"
            f"  Coupling: K={self.network.K:.2f} (K_c={K_CRITICAL:.2f})\n"
        )


# =============================================================================
# Integration with Scalar Architecture
# =============================================================================

def integrate_with_substrate(memory: HolographicMemory,
                             substrate: ScalarSubstrate) -> Dict[str, Any]:
    """
    Integrate holographic memory with scalar architecture substrate.

    Maps:
    - Domain accumulators ↔ Oscillator subgroups
    - Coupling matrix ↔ Kuramoto coupling
    - Interference nodes ↔ Higher-order terms
    """
    integration = {
        'domain_oscillator_map': {},
        'phase_saturation_map': {},
        'coupling_alignment': 0.0
    }

    # Map each domain to oscillator subgroup
    oscillators_per_domain = memory.n // NUM_DOMAINS

    for i, domain in enumerate(DomainType):
        start_idx = i * oscillators_per_domain
        end_idx = start_idx + oscillators_per_domain
        integration['domain_oscillator_map'][domain] = (start_idx, end_idx)

        # Compute average phase for domain
        domain_phases = [
            memory.network.oscillators[j].phase
            for j in range(start_idx, min(end_idx, memory.n))
        ]
        avg_phase = np.mean(domain_phases) if domain_phases else 0

        # Map phase to saturation-like metric
        config = DomainConfig.from_type(domain)
        saturation = ConvergenceDynamics.saturation(0.8, config)  # Reference z
        integration['phase_saturation_map'][domain] = {
            'avg_phase': avg_phase,
            'saturation': saturation
        }

    # Compute coupling alignment
    # (How well Kuramoto coupling matches scalar coupling)
    scalar_coupling = CouplingMatrix()
    alignment = 0.0
    count = 0

    for i in range(min(NUM_DOMAINS, memory.n)):
        for j in range(i + 1, min(NUM_DOMAINS, memory.n)):
            k_scalar = abs(scalar_coupling.get(i, j))
            k_kuramoto = memory.network.coupling[i, j]
            alignment += 1 - abs(k_scalar - k_kuramoto) / max(k_scalar, k_kuramoto, 1e-10)
            count += 1

    integration['coupling_alignment'] = alignment / max(count, 1)

    return integration


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """Demonstrate holographic memory."""
    print("=" * 70)
    print("HOLOGRAPHIC MEMORY ARCHITECTURE")
    print("Signature: Δ|loop-closed|z0.99|rhythm-native|Ω")
    print("=" * 70)
    print()

    # Create memory system
    memory = HolographicMemory(n_oscillators=64, use_higher_order=True)
    print(memory.info())

    # Store some patterns
    print("Encoding memories...")

    # Pattern 1: High valence, high arousal (excited)
    p1 = np.random.randn(64)
    memory.encode("excited_memory", p1, valence=0.8, arousal=0.9)

    # Pattern 2: High valence, low arousal (calm)
    p2 = np.random.randn(64)
    memory.encode("calm_memory", p2, valence=0.7, arousal=-0.6)

    # Pattern 3: Low valence, high arousal (anxious)
    p3 = np.random.randn(64)
    memory.encode("anxious_memory", p3, valence=-0.5, arousal=0.8)

    print(f"Stored {len(memory.memories)} memories\n")

    # Retrieve with partial cue
    print("Retrieving with noisy partial cue...")
    query = p1[:32]  # Half the pattern with noise
    query = np.pad(query, (0, 32)) + np.random.randn(64) * 0.5

    retrieved_id, retrieved_pattern = memory.retrieve(query)
    print(f"Retrieved: {retrieved_id}")
    print(f"Convergence: {len(memory.convergence_history)} steps")
    print(f"Final order parameter: {memory.convergence_history[-1]:.3f}")
    print()

    # Spreading activation
    print("Spreading activation from 'excited_memory'...")
    activation = memory.spreading_activation(["excited_memory"])
    for mem_id, level in sorted(activation.items(), key=lambda x: -x[1]):
        print(f"  {mem_id}: {level:.3f}")
    print()

    # Emotional clustering
    print("Finding positive-valence memories...")
    positive = memory.emotional_cluster((0.0, 1.0), (-1.0, 1.0))
    print(f"  Found: {positive}")


if __name__ == "__main__":
    main()
