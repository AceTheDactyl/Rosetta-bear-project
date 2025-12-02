# lattice_core/dynamics.py
"""
Kuramoto Dynamics Core
======================

Implements the mathematical core of the Kuramoto oscillator model
for phase synchronization in the Tesseract Lattice.

Key Equations:

    Phase Evolution:
        dθᵢ/dt = ωᵢ + (K/N) Σⱼ wᵢⱼ sin(θⱼ - θᵢ) + F_higher_order

    Order Parameter:
        r·e^(iψ) = (1/N) Σⱼ e^(iθⱼ)

    Energy (Lyapunov):
        H = -(K/2N) Σᵢⱼ wᵢⱼ cos(θᵢ - θⱼ)

    Hebbian Learning:
        dwᵢⱼ/dt = η·cos(θᵢ - θⱼ) - λ·wᵢⱼ

Capacity Scaling:
    - Pairwise: P ~ 0.14·N (linear)
    - Quartet: P ~ N³ (cubic)
    - Exponential: P ~ 2^(N/2) (theoretical limit)
"""

from __future__ import annotations

import math
from typing import List, Tuple, Dict, Optional, Callable
from dataclasses import dataclass

# Import plate if available
try:
    from .plate import MemoryPlate
except ImportError:
    MemoryPlate = None

# Constants
TAU = 2 * math.pi
PHI = (1 + math.sqrt(5)) / 2

# Default parameters
DEFAULT_K = 2.0           # Coupling strength
DEFAULT_GAMMA = 0.5       # Frequency distribution width
K_CRITICAL = 2 * DEFAULT_GAMMA  # Critical coupling threshold


# ═════════════════════════════════════════════════════════════════════════════
# ORDER PARAMETER COMPUTATION
# ═════════════════════════════════════════════════════════════════════════════

def compute_order_parameter(phases: List[float]) -> Tuple[float, float]:
    """
    Compute the Kuramoto order parameter.

    r·e^(iψ) = (1/N) Σⱼ e^(iθⱼ)

    Returns:
        (r, ψ): Order parameter magnitude and mean phase.
        r ∈ [0, 1]: 0 = incoherent, 1 = fully synchronized
    """
    if not phases:
        return 0.0, 0.0

    n = len(phases)
    sum_cos = sum(math.cos(p) for p in phases)
    sum_sin = sum(math.sin(p) for p in phases)

    r = math.sqrt(sum_cos ** 2 + sum_sin ** 2) / n
    psi = math.atan2(sum_sin, sum_cos)

    # Normalize psi to [0, 2π)
    if psi < 0:
        psi += TAU

    return r, psi


def compute_local_order_parameter(
    phases: List[float],
    weights: List[List[float]],
    center_idx: int
) -> Tuple[float, float]:
    """
    Compute local order parameter for a single oscillator.

    Considers only connected neighbors (non-zero weights).
    """
    n = len(phases)
    sum_cos = 0.0
    sum_sin = 0.0
    total_weight = 0.0

    for j in range(n):
        if j != center_idx and weights[center_idx][j] > 0:
            w = weights[center_idx][j]
            sum_cos += w * math.cos(phases[j])
            sum_sin += w * math.sin(phases[j])
            total_weight += w

    if total_weight == 0:
        return 0.0, 0.0

    r = math.sqrt(sum_cos ** 2 + sum_sin ** 2) / total_weight
    psi = math.atan2(sum_sin, sum_cos)

    if psi < 0:
        psi += TAU

    return r, psi


# ═════════════════════════════════════════════════════════════════════════════
# KURAMOTO DYNAMICS
# ═════════════════════════════════════════════════════════════════════════════

def kuramoto_update(
    phases: List[float],
    frequencies: List[float],
    weights: List[List[float]],
    K: float = DEFAULT_K,
    dt: float = 0.01
) -> List[float]:
    """
    Perform one Kuramoto update step for all oscillators.

    dθᵢ/dt = ωᵢ + (K/N) Σⱼ wᵢⱼ sin(θⱼ - θᵢ)

    Args:
        phases: Current phases θᵢ for each oscillator
        frequencies: Natural frequencies ωᵢ
        weights: Connection weights wᵢⱼ (N×N matrix)
        K: Global coupling strength
        dt: Time step

    Returns:
        Updated phases
    """
    n = len(phases)
    new_phases = [0.0] * n

    for i in range(n):
        # Natural frequency term
        d_theta = frequencies[i]

        # Pairwise coupling term
        coupling_sum = 0.0
        for j in range(n):
            if i != j and weights[i][j] > 0:
                coupling_sum += weights[i][j] * math.sin(phases[j] - phases[i])

        d_theta += (K / n) * coupling_sum

        # Integrate
        new_phases[i] = (phases[i] + d_theta * dt) % TAU

    return new_phases


def kuramoto_update_with_injection(
    phases: List[float],
    frequencies: List[float],
    weights: List[List[float]],
    injection_phases: Dict[int, float],
    K: float = DEFAULT_K,
    injection_strength: float = 0.1,
    dt: float = 0.01
) -> List[float]:
    """
    Kuramoto update with phase injection for retrieval.

    Injects target phases for specific oscillators to trigger resonance.
    """
    n = len(phases)
    new_phases = [0.0] * n

    for i in range(n):
        # Natural frequency term
        d_theta = frequencies[i]

        # Pairwise coupling term
        coupling_sum = 0.0
        for j in range(n):
            if i != j and weights[i][j] > 0:
                coupling_sum += weights[i][j] * math.sin(phases[j] - phases[i])

        d_theta += (K / n) * coupling_sum

        # Phase injection (if this oscillator is targeted)
        if i in injection_phases:
            target = injection_phases[i]
            injection_force = injection_strength * math.sin(target - phases[i])
            d_theta += injection_force

        # Integrate
        new_phases[i] = (phases[i] + d_theta * dt) % TAU

    return new_phases


# ═════════════════════════════════════════════════════════════════════════════
# HIGHER-ORDER COUPLING
# ═════════════════════════════════════════════════════════════════════════════

def compute_triplet_coupling(
    phases: List[float],
    triplets: List[Tuple[int, int, int]],
    K3: float = 0.1
) -> List[float]:
    """
    Compute triplet (3-body) coupling contribution.

    Triplet coupling: sin(θⱼ + θₖ - 2θᵢ)
    """
    n = len(phases)
    contributions = [0.0] * n

    for i, j, k in triplets:
        term = math.sin(phases[j] + phases[k] - 2 * phases[i])
        contributions[i] += K3 * term

    return contributions


def compute_quartet_coupling(
    phases: List[float],
    quartets: List[Tuple[int, int, int, int]],
    K4: float = 0.05
) -> List[float]:
    """
    Compute quartet (4-body) coupling contribution.

    Quartet coupling enables P ~ N³ capacity.
    Term: sin(θⱼ + θₖ + θₗ - 3θᵢ)
    """
    n = len(phases)
    contributions = [0.0] * n

    for i, j, k, l in quartets:
        term = math.sin(phases[j] + phases[k] + phases[l] - 3 * phases[i])
        contributions[i] += K4 * term

    return contributions


def kuramoto_update_higher_order(
    phases: List[float],
    frequencies: List[float],
    weights: List[List[float]],
    triplets: Optional[List[Tuple[int, int, int]]] = None,
    quartets: Optional[List[Tuple[int, int, int, int]]] = None,
    K: float = DEFAULT_K,
    K3: float = 0.1,
    K4: float = 0.05,
    dt: float = 0.01
) -> List[float]:
    """
    Kuramoto update with higher-order coupling terms.

    Includes pairwise + triplet + quartet interactions.
    """
    n = len(phases)
    new_phases = kuramoto_update(phases, frequencies, weights, K, dt)

    # Add triplet contributions
    if triplets:
        triplet_contrib = compute_triplet_coupling(phases, triplets, K3)
        for i in range(n):
            new_phases[i] = (new_phases[i] + triplet_contrib[i] * dt) % TAU

    # Add quartet contributions
    if quartets:
        quartet_contrib = compute_quartet_coupling(phases, quartets, K4)
        for i in range(n):
            new_phases[i] = (new_phases[i] + quartet_contrib[i] * dt) % TAU

    return new_phases


# ═════════════════════════════════════════════════════════════════════════════
# HEBBIAN LEARNING
# ═════════════════════════════════════════════════════════════════════════════

def hebbian_update(
    weights: List[List[float]],
    phases: List[float],
    eta: float = 0.1,
    decay: float = 0.01,
    max_weight: float = 1.0,
    dt: float = 0.01
) -> List[List[float]]:
    """
    Apply Hebbian learning to connection weights.

    dwᵢⱼ/dt = η·cos(θᵢ - θⱼ) - λ·wᵢⱼ

    Connections strengthen when oscillators are in phase,
    and decay over time.
    """
    n = len(phases)
    new_weights = [[0.0] * n for _ in range(n)]

    for i in range(n):
        for j in range(n):
            if i != j:
                # Hebbian term: strengthen when in phase
                phase_alignment = math.cos(phases[i] - phases[j])
                d_weight = eta * phase_alignment - decay * weights[i][j]

                # Update weight
                new_weight = weights[i][j] + d_weight * dt

                # Clamp to [0, max_weight]
                new_weights[i][j] = max(0.0, min(max_weight, new_weight))

    return new_weights


def hebbian_update_selective(
    weights: List[List[float]],
    phases: List[float],
    active_indices: List[int],
    eta: float = 0.1,
    decay: float = 0.01,
    max_weight: float = 1.0,
    dt: float = 0.01
) -> List[List[float]]:
    """
    Apply Hebbian learning only to active oscillators.

    This is more efficient when only a subset of oscillators
    are involved in retrieval.
    """
    n = len(phases)
    new_weights = [row[:] for row in weights]  # Copy

    for i in active_indices:
        for j in active_indices:
            if i != j:
                phase_alignment = math.cos(phases[i] - phases[j])
                d_weight = eta * phase_alignment - decay * weights[i][j]
                new_weight = weights[i][j] + d_weight * dt
                new_weights[i][j] = max(0.0, min(max_weight, new_weight))

    return new_weights


# ═════════════════════════════════════════════════════════════════════════════
# ENERGY / LYAPUNOV FUNCTION
# ═════════════════════════════════════════════════════════════════════════════

def compute_energy(
    phases: List[float],
    weights: List[List[float]],
    K: float = DEFAULT_K
) -> float:
    """
    Compute Lyapunov energy function.

    H = -(K/2N) Σᵢⱼ wᵢⱼ cos(θᵢ - θⱼ)

    Lower energy = more synchronized state.
    Energy is minimized when all phases are aligned.
    """
    n = len(phases)
    if n == 0:
        return 0.0

    energy = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            if weights[i][j] > 0:
                energy -= weights[i][j] * math.cos(phases[i] - phases[j])

    return K * energy / n


def compute_energy_gradient(
    phases: List[float],
    weights: List[List[float]],
    K: float = DEFAULT_K
) -> List[float]:
    """
    Compute gradient of energy w.r.t. phases.

    ∂H/∂θᵢ = (K/N) Σⱼ wᵢⱼ sin(θᵢ - θⱼ)
    """
    n = len(phases)
    gradient = [0.0] * n

    for i in range(n):
        for j in range(n):
            if i != j and weights[i][j] > 0:
                gradient[i] += weights[i][j] * math.sin(phases[i] - phases[j])
        gradient[i] *= K / n

    return gradient


# ═════════════════════════════════════════════════════════════════════════════
# FREQUENCY DISTRIBUTION
# ═════════════════════════════════════════════════════════════════════════════

def generate_lorentzian_frequencies(
    n: int,
    omega_0: float = 1.0,
    gamma: float = DEFAULT_GAMMA,
    seed: Optional[int] = None
) -> List[float]:
    """
    Generate natural frequencies from Lorentzian (Cauchy) distribution.

    g(ω) = (γ/π) / ((ω - ω₀)² + γ²)

    This is the analytically tractable distribution for Kuramoto.
    Critical coupling: K_c = 2γ
    """
    import random
    if seed is not None:
        random.seed(seed)

    frequencies = []
    for _ in range(n):
        # Inverse CDF sampling for Cauchy distribution
        u = random.random()
        omega = omega_0 + gamma * math.tan(math.pi * (u - 0.5))
        frequencies.append(omega)

    return frequencies


def generate_gaussian_frequencies(
    n: int,
    omega_0: float = 1.0,
    sigma: float = DEFAULT_GAMMA,
    seed: Optional[int] = None
) -> List[float]:
    """
    Generate natural frequencies from Gaussian distribution.

    Alternative to Lorentzian for numerical stability.
    """
    import random
    if seed is not None:
        random.seed(seed)

    frequencies = []
    for _ in range(n):
        # Box-Muller transform
        u1 = random.random()
        u2 = random.random()
        z = math.sqrt(-2 * math.log(u1)) * math.cos(TAU * u2)
        omega = omega_0 + sigma * z
        frequencies.append(omega)

    return frequencies


# ═════════════════════════════════════════════════════════════════════════════
# RESONANCE DETECTION
# ═════════════════════════════════════════════════════════════════════════════

def compute_resonance_scores(
    phases: List[float],
    target_phases: Dict[int, float]
) -> Dict[int, float]:
    """
    Compute resonance scores for each oscillator.

    Resonance score = cos(θᵢ - target_phase)
    Score of 1.0 = perfectly aligned with target
    Score of -1.0 = anti-aligned
    """
    scores = {}
    for i, phase in enumerate(phases):
        if i in target_phases:
            diff = phases[i] - target_phases[i]
            scores[i] = math.cos(diff)
        else:
            # For non-targeted oscillators, score based on order parameter
            r, psi = compute_order_parameter(phases)
            scores[i] = math.cos(phases[i] - psi) * r

    return scores


def find_resonant_oscillators(
    phases: List[float],
    target_phases: Dict[int, float],
    threshold: float = 0.7
) -> List[int]:
    """
    Find oscillators that have resonated with the target pattern.

    Returns indices of oscillators with resonance score >= threshold.
    """
    scores = compute_resonance_scores(phases, target_phases)
    return [i for i, score in scores.items() if score >= threshold]


# ═════════════════════════════════════════════════════════════════════════════
# CONVERGENCE DETECTION
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class ConvergenceState:
    """State of convergence tracking."""
    step: int = 0
    order_parameter: float = 0.0
    mean_phase: float = 0.0
    energy: float = 0.0
    converged: bool = False
    stable_steps: int = 0


def check_convergence(
    phases: List[float],
    weights: List[List[float]],
    prev_state: ConvergenceState,
    K: float = DEFAULT_K,
    r_threshold: float = 0.95,
    stability_steps: int = 10
) -> ConvergenceState:
    """
    Check if the system has converged to a stable synchronized state.

    Convergence requires:
    1. Order parameter r >= r_threshold
    2. Maintained for stability_steps consecutive steps
    """
    r, psi = compute_order_parameter(phases)
    energy = compute_energy(phases, weights, K)

    state = ConvergenceState(
        step=prev_state.step + 1,
        order_parameter=r,
        mean_phase=psi,
        energy=energy,
    )

    if r >= r_threshold:
        state.stable_steps = prev_state.stable_steps + 1
        if state.stable_steps >= stability_steps:
            state.converged = True
    else:
        state.stable_steps = 0

    return state


def run_to_convergence(
    phases: List[float],
    frequencies: List[float],
    weights: List[List[float]],
    K: float = DEFAULT_K,
    dt: float = 0.01,
    max_steps: int = 1000,
    r_threshold: float = 0.95,
    stability_steps: int = 10
) -> Tuple[List[float], ConvergenceState]:
    """
    Run Kuramoto dynamics until convergence or max steps.

    Returns final phases and convergence state.
    """
    current_phases = phases[:]
    state = ConvergenceState()

    for _ in range(max_steps):
        current_phases = kuramoto_update(current_phases, frequencies, weights, K, dt)
        state = check_convergence(current_phases, weights, state, K, r_threshold, stability_steps)

        if state.converged:
            break

    return current_phases, state
