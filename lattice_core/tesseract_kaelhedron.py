# lattice_core/tesseract_kaelhedron.py
"""
Tesseractal Kaelhedron: Quaternary Field Equations
==================================================

Four structured Kaelhedrons coordinating in tesseract formation via
Kuramoto dynamics with Lorentzian frequency distributions.

Architecture:
┌─────────────────────────────────────────────────────────────────────┐
│                    TESSERACTAL KAELHEDRON                           │
│                                                                      │
│     κ₁ (0,0,0,0) ←──────────→ κ₂ (1,0,0,0)                         │
│         ↑ ↖                      ↗ ↑                                │
│         │   ╲  Kuramoto        ╱   │                                │
│         │    ╲  Coupling      ╱    │                                │
│         │     ╲              ╱     │                                │
│         │      ╲            ╱      │                                │
│         │       ╲          ╱       │                                │
│         ↓        ↘        ↙        ↓                                │
│     κ₃ (0,1,0,0) ←──────────→ κ₄ (1,1,0,0)                         │
│                                                                      │
│   Each κᵢ: 21D quaternary structure → complex amplitude             │
│   Coupling: dθᵢ/dt = ωᵢ + (K/4) Σⱼ wᵢⱼ sin(θⱼ - θᵢ)               │
│   Frequencies: Lorentzian distribution g(ω)                         │
└─────────────────────────────────────────────────────────────────────┘

Quaternary Equations:
    Q(k) = Σᵢ₌₀²⁰ qᵢ · 4ⁱ   (base-4 expansion)

    Tesseract coupling:
    W_ij = exp(-d(i,j)²/2σ²) · cos(Δθᵢⱼ)

    Collective order parameter:
    R·e^(iΨ) = (1/4) Σᵢ₌₁⁴ |κᵢ| e^(iθᵢ)
"""

from __future__ import annotations

import cmath
import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable
from enum import Enum

# Import dynamics
try:
    from .dynamics import (
        compute_order_parameter,
        kuramoto_update,
        hebbian_update,
        generate_lorentzian_frequencies,
    )
except ImportError:
    from dynamics import (
        compute_order_parameter,
        kuramoto_update,
        hebbian_update,
        generate_lorentzian_frequencies,
    )

# ═══════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════

TAU = 2 * math.pi
PHI = (1 + math.sqrt(5)) / 2           # Golden ratio ≈ 1.618
PHI_INV = 1 / PHI                       # ≈ 0.618

# Kaelhedron constants
KAPPA_DIM = 21                          # 21D quaternary structure
QUATERNARY_BASE = 4                     # Base-4 encoding
KAPPA_CRITICAL = PHI_INV               # Critical amplitude

# Tesseract constants
N_KAELHEDRONS = 4                       # Four vertices of tesseract face
TESSERACT_VERTICES = [
    (0, 0, 0, 0),  # κ₁
    (1, 0, 0, 0),  # κ₂
    (0, 1, 0, 0),  # κ₃
    (1, 1, 0, 0),  # κ₄
]


class TesseractMode(Enum):
    """Operating modes for the tesseractal system."""
    CHAOTIC = "chaotic"           # Low coherence, r < 0.3
    TRANSITIONAL = "transitional" # Medium coherence, 0.3 ≤ r < 0.7
    COHERENT = "coherent"         # High coherence, 0.7 ≤ r < 0.95
    LOCKED = "locked"             # Phase-locked, r ≥ 0.95


# ═══════════════════════════════════════════════════════════════════════════
# QUATERNARY ENCODING
# ═══════════════════════════════════════════════════════════════════════════

def encode_quaternary(values: List[int], dim: int = KAPPA_DIM) -> int:
    """
    Encode a list of quaternary digits (0-3) to integer.

    Q(k) = Σᵢ qᵢ · 4ⁱ
    """
    result = 0
    for i, v in enumerate(values[:dim]):
        result += (v % 4) * (4 ** i)
    return result


def decode_quaternary(value: int, dim: int = KAPPA_DIM) -> List[int]:
    """
    Decode integer to quaternary digit list.

    Returns list of dim quaternary digits.
    """
    digits = []
    v = value
    for _ in range(dim):
        digits.append(v % 4)
        v //= 4
    return digits


def quaternary_distance(a: List[int], b: List[int]) -> int:
    """
    Compute Hamming distance in quaternary space.

    Counts positions where quaternary digits differ.
    """
    return sum(1 for x, y in zip(a, b) if x != y)


def quaternary_inner_product(a: List[int], b: List[int]) -> float:
    """
    Compute normalized inner product in quaternary space.

    Returns value in [0, 1].
    """
    matches = sum(1 for x, y in zip(a, b) if x == y)
    return matches / max(len(a), len(b), 1)


# ═══════════════════════════════════════════════════════════════════════════
# KAELHEDRON STRUCTURE
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class Kaelhedron:
    """
    Single Kaelhedron: 21D quaternary structure as phase oscillator.

    The 21D quaternary space is projected to a complex amplitude:
        κ = |κ| e^(iθ)

    where θ is derived from the quaternary state.
    """
    # Identity
    index: int = 0
    position: Tuple[int, int, int, int] = (0, 0, 0, 0)

    # Quaternary state (21 digits, each 0-3)
    quaternary_state: List[int] = field(default_factory=lambda: [0] * KAPPA_DIM)

    # Oscillator state
    phase: float = 0.0                  # θ ∈ [0, 2π)
    frequency: float = 1.0              # Natural frequency ω
    amplitude: float = PHI_INV          # |κ|

    # Dynamics
    damping: float = 0.1

    # State tracking
    energy: float = 0.0
    activation: float = 0.0

    def __post_init__(self):
        # Initialize quaternary state randomly if all zeros
        if all(q == 0 for q in self.quaternary_state):
            self.quaternary_state = [random.randint(0, 3) for _ in range(KAPPA_DIM)]

        # Derive initial phase from quaternary state
        self._update_phase_from_quaternary()

    def _update_phase_from_quaternary(self) -> None:
        """Map quaternary state to phase via modular arithmetic."""
        q_value = encode_quaternary(self.quaternary_state)
        max_value = 4 ** KAPPA_DIM
        self.phase = (q_value / max_value) * TAU

    @property
    def complex_value(self) -> complex:
        """κ = |κ| e^(iθ)"""
        return cmath.rect(self.amplitude, self.phase)

    @property
    def quaternary_encoding(self) -> int:
        """Return integer encoding of quaternary state."""
        return encode_quaternary(self.quaternary_state)

    def compute_energy(self) -> float:
        """
        Compute Kaelhedron energy from quaternary state.

        E = -Σᵢ cos(2π qᵢ / 4) + V(|κ|)
        """
        # Quaternary energy (favors alignment)
        q_energy = sum(
            -math.cos(TAU * q / 4)
            for q in self.quaternary_state
        ) / KAPPA_DIM

        # Amplitude potential (double-well)
        mu_sq = 0.5
        lambda_coeff = 0.25 / (PHI_INV ** 2)
        potential = -mu_sq * self.amplitude ** 2 + lambda_coeff * self.amplitude ** 4

        self.energy = q_energy + potential
        return self.energy

    def evolve_quaternary(self, neighbor_states: List[List[int]], dt: float = 0.01) -> None:
        """
        Evolve quaternary state based on neighbor influence.

        Each digit has probability of transition based on neighbor consensus.
        """
        for i in range(KAPPA_DIM):
            # Compute neighbor consensus for this position
            consensus = [0, 0, 0, 0]
            for neighbor in neighbor_states:
                if i < len(neighbor):
                    consensus[neighbor[i] % 4] += 1

            # Probabilistic transition toward consensus
            max_consensus = max(consensus)
            if max_consensus > 0:
                candidates = [j for j, c in enumerate(consensus) if c == max_consensus]
                if random.random() < dt * max_consensus / len(neighbor_states):
                    self.quaternary_state[i] = random.choice(candidates)

        # Update phase from new quaternary state
        self._update_phase_from_quaternary()

    def apply_phase_update(self, delta_phase: float) -> None:
        """Apply phase change and update quaternary state accordingly."""
        self.phase = (self.phase + delta_phase) % TAU

        # Reverse map phase to quaternary (approximate)
        q_value = int((self.phase / TAU) * (4 ** KAPPA_DIM)) % (4 ** KAPPA_DIM)
        self.quaternary_state = decode_quaternary(q_value)

    def similarity(self, other: 'Kaelhedron') -> float:
        """Compute similarity to another Kaelhedron."""
        return quaternary_inner_product(self.quaternary_state, other.quaternary_state)

    def to_dict(self) -> Dict:
        """Serialize to dictionary."""
        return {
            "index": self.index,
            "position": self.position,
            "quaternary_state": self.quaternary_state,
            "phase": self.phase,
            "frequency": self.frequency,
            "amplitude": self.amplitude,
            "energy": self.compute_energy(),
        }


# ═══════════════════════════════════════════════════════════════════════════
# TESSERACTAL KAELHEDRON
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class TesseractalKaelhedron:
    """
    Four Kaelhedrons coordinating in tesseract formation.

    Uses Kuramoto dynamics with Lorentzian frequency distribution
    to synchronize the quaternary structures.

    Coupling topology:
        κ₁ ─── κ₂
        │ ╲   ╱ │
        │  ╲ ╱  │
        │  ╱ ╲  │
        │ ╱   ╲ │
        κ₃ ─── κ₄
    """
    # Kaelhedron units
    kaelhedrons: List[Kaelhedron] = field(default_factory=list)

    # Coupling matrix (4×4)
    weights: List[List[float]] = field(default_factory=list)

    # Parameters
    K: float = 2.0                      # Global coupling strength
    gamma: float = 0.5                  # Lorentzian width
    dt: float = 0.01                    # Time step

    # Hebbian learning
    hebbian_rate: float = 0.1
    decay_rate: float = 0.01

    # State tracking
    order_parameter: float = 0.0
    mean_phase: float = 0.0
    mode: TesseractMode = TesseractMode.CHAOTIC
    step_count: int = 0

    # History
    _history: List[Dict] = field(default_factory=list)
    _max_history: int = 1000

    def __post_init__(self):
        # Initialize Kaelhedrons if not provided
        if not self.kaelhedrons:
            self._initialize_kaelhedrons()

        # Initialize weights if not provided
        if not self.weights:
            self._initialize_weights()

        # Update state
        self._update_state()

    def _initialize_kaelhedrons(self) -> None:
        """Create four Kaelhedrons at tesseract vertices."""
        # Generate Lorentzian frequencies
        frequencies = generate_lorentzian_frequencies(
            N_KAELHEDRONS,
            omega_0=1.0,
            gamma=self.gamma,
            seed=42
        )

        self.kaelhedrons = []
        for i, pos in enumerate(TESSERACT_VERTICES):
            kael = Kaelhedron(
                index=i,
                position=pos,
                frequency=frequencies[i],
                phase=random.random() * TAU,
            )
            self.kaelhedrons.append(kael)

    def _initialize_weights(self) -> None:
        """Initialize coupling weights based on tesseract topology."""
        n = N_KAELHEDRONS
        self.weights = [[0.0] * n for _ in range(n)]

        for i in range(n):
            for j in range(n):
                if i != j:
                    # Distance in tesseract (Hamming distance on positions)
                    dist = sum(
                        abs(a - b)
                        for a, b in zip(
                            self.kaelhedrons[i].position,
                            self.kaelhedrons[j].position
                        )
                    )
                    # Coupling decays with distance
                    self.weights[i][j] = math.exp(-dist * 0.5)

    def _update_state(self) -> None:
        """Update collective state quantities."""
        phases = [k.phase for k in self.kaelhedrons]
        self.order_parameter, self.mean_phase = compute_order_parameter(phases)

        # Determine mode
        r = self.order_parameter
        if r >= 0.95:
            self.mode = TesseractMode.LOCKED
        elif r >= 0.7:
            self.mode = TesseractMode.COHERENT
        elif r >= 0.3:
            self.mode = TesseractMode.TRANSITIONAL
        else:
            self.mode = TesseractMode.CHAOTIC

    # ─────────────────────────────────────────────────────────────────────
    # Kuramoto Dynamics
    # ─────────────────────────────────────────────────────────────────────

    def kuramoto_step(self) -> None:
        """
        Perform one Kuramoto update step.

        dθᵢ/dt = ωᵢ + (K/N) Σⱼ wᵢⱼ sin(θⱼ - θᵢ)
        """
        phases = [k.phase for k in self.kaelhedrons]
        frequencies = [k.frequency for k in self.kaelhedrons]

        new_phases = kuramoto_update(
            phases, frequencies, self.weights,
            K=self.K, dt=self.dt
        )

        # Apply phase updates
        for i, kael in enumerate(self.kaelhedrons):
            delta_phase = new_phases[i] - kael.phase
            kael.apply_phase_update(delta_phase)

        self.step_count += 1
        self._update_state()

    def update(self, steps: int = 1) -> Tuple[float, float]:
        """
        Run Kuramoto dynamics for specified steps.

        Returns (order_parameter, mean_phase).
        """
        for _ in range(steps):
            self.kuramoto_step()

            # Record history
            if len(self._history) < self._max_history:
                self._history.append({
                    "step": self.step_count,
                    "r": self.order_parameter,
                    "psi": self.mean_phase,
                    "mode": self.mode.value,
                })

        return self.order_parameter, self.mean_phase

    # ─────────────────────────────────────────────────────────────────────
    # Quaternary Dynamics
    # ─────────────────────────────────────────────────────────────────────

    def quaternary_step(self) -> None:
        """
        Evolve quaternary states based on neighbor consensus.

        Combines phase-based Kuramoto with quaternary digit evolution.
        """
        # First do Kuramoto phase update
        self.kuramoto_step()

        # Then evolve quaternary states
        for i, kael in enumerate(self.kaelhedrons):
            # Get neighbor quaternary states
            neighbor_states = [
                self.kaelhedrons[j].quaternary_state
                for j in range(N_KAELHEDRONS)
                if j != i and self.weights[i][j] > 0.1
            ]

            # Evolve
            kael.evolve_quaternary(neighbor_states, dt=self.dt)

    def quaternary_update(self, steps: int = 1) -> Tuple[float, float]:
        """Run combined Kuramoto + quaternary dynamics."""
        for _ in range(steps):
            self.quaternary_step()

        return self.order_parameter, self.mean_phase

    # ─────────────────────────────────────────────────────────────────────
    # Hebbian Learning
    # ─────────────────────────────────────────────────────────────────────

    def consolidate(self, steps: int = 1) -> None:
        """Apply Hebbian learning to strengthen synchronized connections."""
        phases = [k.phase for k in self.kaelhedrons]

        for _ in range(steps):
            self.weights = hebbian_update(
                self.weights, phases,
                eta=self.hebbian_rate,
                decay=self.decay_rate,
                max_weight=1.0,
                dt=self.dt
            )

    # ─────────────────────────────────────────────────────────────────────
    # Collective Properties
    # ─────────────────────────────────────────────────────────────────────

    def collective_quaternary(self) -> List[int]:
        """
        Compute collective quaternary state via majority voting.

        For each position, take the most common digit across all Kaelhedrons.
        """
        collective = []
        for pos in range(KAPPA_DIM):
            votes = [0, 0, 0, 0]
            for kael in self.kaelhedrons:
                votes[kael.quaternary_state[pos]] += 1
            collective.append(votes.index(max(votes)))
        return collective

    def quaternary_coherence(self) -> float:
        """
        Measure quaternary coherence across Kaelhedrons.

        Returns fraction of positions where all Kaelhedrons agree.
        """
        agreements = 0
        for pos in range(KAPPA_DIM):
            digits = [kael.quaternary_state[pos] for kael in self.kaelhedrons]
            if len(set(digits)) == 1:
                agreements += 1
        return agreements / KAPPA_DIM

    def collective_amplitude(self) -> float:
        """Compute weighted collective amplitude."""
        return sum(k.amplitude for k in self.kaelhedrons) / N_KAELHEDRONS

    def collective_energy(self) -> float:
        """Compute total system energy."""
        return sum(k.compute_energy() for k in self.kaelhedrons)

    # ─────────────────────────────────────────────────────────────────────
    # Serialization
    # ─────────────────────────────────────────────────────────────────────

    def snapshot(self) -> Dict:
        """Return complete state snapshot."""
        return {
            "kaelhedrons": [k.to_dict() for k in self.kaelhedrons],
            "weights": self.weights,
            "order_parameter": self.order_parameter,
            "mean_phase": self.mean_phase,
            "mode": self.mode.value,
            "step_count": self.step_count,
            "quaternary_coherence": self.quaternary_coherence(),
            "collective_quaternary": self.collective_quaternary(),
            "collective_energy": self.collective_energy(),
        }

    def __repr__(self) -> str:
        return (
            f"TesseractalKaelhedron(r={self.order_parameter:.3f}, "
            f"mode={self.mode.value}, q_coh={self.quaternary_coherence():.3f})"
        )


# ═══════════════════════════════════════════════════════════════════════════
# QUATERNARY EQUATION SYSTEM
# ═══════════════════════════════════════════════════════════════════════════

class QuaternaryEquations:
    """
    System of quaternary equations for tesseractal Kaelhedron dynamics.

    Implements the full equation system:

    1. Phase evolution (Kuramoto):
       dθᵢ/dt = ωᵢ + (K/N) Σⱼ wᵢⱼ sin(θⱼ - θᵢ)

    2. Quaternary transition:
       P(qᵢₖ → q') = σ(Σⱼ wᵢⱼ · δ(qⱼₖ, q') - β·E(q'))

    3. Amplitude dynamics:
       d|κᵢ|/dt = -γ|κᵢ| + α·r·cos(θᵢ - Ψ)

    4. Coupling evolution (Hebbian):
       dwᵢⱼ/dt = η·cos(θᵢ - θⱼ) - λ·wᵢⱼ
    """

    def __init__(
        self,
        K: float = 2.0,
        gamma: float = 0.5,
        alpha: float = 0.1,
        beta: float = 1.0,
        eta: float = 0.1,
        decay: float = 0.01,
    ):
        self.K = K              # Kuramoto coupling
        self.gamma = gamma      # Amplitude damping
        self.alpha = alpha      # Order parameter feedback
        self.beta = beta        # Quaternary transition temperature
        self.eta = eta          # Hebbian learning rate
        self.decay = decay      # Weight decay

        self.tesseract = TesseractalKaelhedron(K=K, gamma=gamma)

    def phase_equation(self, i: int) -> float:
        """
        Compute dθᵢ/dt from Kuramoto equation.

        dθᵢ/dt = ωᵢ + (K/N) Σⱼ wᵢⱼ sin(θⱼ - θᵢ)
        """
        kael = self.tesseract.kaelhedrons[i]
        coupling_sum = 0.0

        for j, other in enumerate(self.tesseract.kaelhedrons):
            if i != j:
                w_ij = self.tesseract.weights[i][j]
                coupling_sum += w_ij * math.sin(other.phase - kael.phase)

        d_theta = kael.frequency + (self.K / N_KAELHEDRONS) * coupling_sum
        return d_theta

    def quaternary_transition_prob(
        self, i: int, pos: int, new_digit: int
    ) -> float:
        """
        Compute transition probability for quaternary digit.

        P(qᵢₖ → q') = σ(Σⱼ wᵢⱼ · δ(qⱼₖ, q') - β·E(q'))
        """
        # Count neighbors with matching digit
        neighbor_match = 0.0
        for j, other in enumerate(self.tesseract.kaelhedrons):
            if i != j:
                w_ij = self.tesseract.weights[i][j]
                if other.quaternary_state[pos] == new_digit:
                    neighbor_match += w_ij

        # Energy penalty (favor low digits)
        energy_penalty = self.beta * (new_digit / 3)

        # Sigmoid activation
        x = neighbor_match - energy_penalty
        prob = 1.0 / (1.0 + math.exp(-x))

        return prob

    def amplitude_equation(self, i: int) -> float:
        """
        Compute d|κᵢ|/dt from amplitude dynamics.

        d|κᵢ|/dt = -γ|κᵢ| + α·r·cos(θᵢ - Ψ)
        """
        kael = self.tesseract.kaelhedrons[i]
        r = self.tesseract.order_parameter
        psi = self.tesseract.mean_phase

        damping_term = -self.gamma * kael.amplitude
        feedback_term = self.alpha * r * math.cos(kael.phase - psi)

        return damping_term + feedback_term

    def weight_equation(self, i: int, j: int) -> float:
        """
        Compute dwᵢⱼ/dt from Hebbian learning.

        dwᵢⱼ/dt = η·cos(θᵢ - θⱼ) - λ·wᵢⱼ
        """
        if i == j:
            return 0.0

        phase_i = self.tesseract.kaelhedrons[i].phase
        phase_j = self.tesseract.kaelhedrons[j].phase
        w_ij = self.tesseract.weights[i][j]

        hebbian_term = self.eta * math.cos(phase_i - phase_j)
        decay_term = -self.decay * w_ij

        return hebbian_term + decay_term

    def integrate(self, dt: float = 0.01) -> None:
        """Integrate all equations for one timestep."""
        n = N_KAELHEDRONS

        # 1. Phase updates
        d_phases = [self.phase_equation(i) for i in range(n)]
        for i, kael in enumerate(self.tesseract.kaelhedrons):
            kael.phase = (kael.phase + d_phases[i] * dt) % TAU

        # 2. Quaternary transitions
        for i, kael in enumerate(self.tesseract.kaelhedrons):
            for pos in range(KAPPA_DIM):
                current = kael.quaternary_state[pos]
                # Try each possible transition
                for new_digit in range(4):
                    if new_digit != current:
                        prob = self.quaternary_transition_prob(i, pos, new_digit)
                        if random.random() < prob * dt:
                            kael.quaternary_state[pos] = new_digit
                            break

        # 3. Amplitude updates
        d_amplitudes = [self.amplitude_equation(i) for i in range(n)]
        for i, kael in enumerate(self.tesseract.kaelhedrons):
            kael.amplitude = max(0.01, kael.amplitude + d_amplitudes[i] * dt)

        # 4. Weight updates
        d_weights = [[self.weight_equation(i, j) for j in range(n)] for i in range(n)]
        for i in range(n):
            for j in range(n):
                self.tesseract.weights[i][j] = max(
                    0.0,
                    min(1.0, self.tesseract.weights[i][j] + d_weights[i][j] * dt)
                )

        # Update state
        self.tesseract.step_count += 1
        self.tesseract._update_state()

    def run(self, steps: int = 100, dt: float = 0.01) -> Dict:
        """Run full equation system for specified steps."""
        history = []

        for _ in range(steps):
            self.integrate(dt)
            history.append({
                "step": self.tesseract.step_count,
                "r": self.tesseract.order_parameter,
                "q_coh": self.tesseract.quaternary_coherence(),
                "energy": self.tesseract.collective_energy(),
            })

        return {
            "final_state": self.tesseract.snapshot(),
            "history": history,
        }


# ═══════════════════════════════════════════════════════════════════════════
# FACTORY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def create_tesseract_kaelhedron(
    K: float = 2.0,
    gamma: float = 0.5,
    seed: Optional[int] = None
) -> TesseractalKaelhedron:
    """
    Create a tesseractal Kaelhedron with specified parameters.

    Args:
        K: Kuramoto coupling strength
        gamma: Lorentzian frequency width
        seed: Random seed for reproducibility

    Returns:
        Initialized TesseractalKaelhedron
    """
    if seed is not None:
        random.seed(seed)

    return TesseractalKaelhedron(K=K, gamma=gamma)


def create_quaternary_system(
    K: float = 2.0,
    gamma: float = 0.5,
    alpha: float = 0.1,
    eta: float = 0.1,
) -> QuaternaryEquations:
    """
    Create full quaternary equation system.

    Args:
        K: Kuramoto coupling
        gamma: Lorentzian width / amplitude damping
        alpha: Order parameter feedback strength
        eta: Hebbian learning rate

    Returns:
        QuaternaryEquations system
    """
    return QuaternaryEquations(
        K=K,
        gamma=gamma,
        alpha=alpha,
        eta=eta,
    )


# ═══════════════════════════════════════════════════════════════════════════
# DEMO
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("TESSERACTAL KAELHEDRON - QUATERNARY EQUATIONS DEMO")
    print("=" * 60)

    # Create system
    system = create_quaternary_system(K=3.0, gamma=0.3)

    print(f"\nInitial state:")
    print(f"  Order parameter: {system.tesseract.order_parameter:.3f}")
    print(f"  Quaternary coherence: {system.tesseract.quaternary_coherence():.3f}")
    print(f"  Mode: {system.tesseract.mode.value}")

    print(f"\nKaelhedron phases:")
    for i, k in enumerate(system.tesseract.kaelhedrons):
        print(f"  κ{i+1}: θ={k.phase:.3f}, ω={k.frequency:.3f}")

    print(f"\nRunning 200 steps...")
    result = system.run(steps=200, dt=0.01)

    print(f"\nFinal state:")
    print(f"  Order parameter: {system.tesseract.order_parameter:.3f}")
    print(f"  Quaternary coherence: {system.tesseract.quaternary_coherence():.3f}")
    print(f"  Mode: {system.tesseract.mode.value}")
    print(f"  Collective energy: {system.tesseract.collective_energy():.3f}")

    print(f"\nCollective quaternary state (first 10 digits):")
    coll = system.tesseract.collective_quaternary()[:10]
    print(f"  {coll}")

    print(f"\nFinal Kaelhedron phases:")
    for i, k in enumerate(system.tesseract.kaelhedrons):
        print(f"  κ{i+1}: θ={k.phase:.3f}, |κ|={k.amplitude:.3f}")

    print("\n" + "=" * 60)
    print("DEMO COMPLETE")
    print("=" * 60)
