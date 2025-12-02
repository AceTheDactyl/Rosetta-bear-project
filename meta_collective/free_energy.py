# meta_collective/free_energy.py
"""
Free Energy Minimization Framework
===================================

Implements the variational free energy principle at each architectural level.

Free Energy Decomposition:
    F = D_KL[q(s) || p(s|o)] + complexity
    F = -log p(o) + D_KL[q(s) || p(s|o)]  (Evidence lower bound)

Where:
    - q(s): Recognition density (belief about hidden states)
    - p(s|o): Posterior (true state given observations)
    - o: Observations
    - s: Hidden states

Nested Free Energy:
    Each level minimizes its own F while contributing to parent's F.
    F_parent = F_self + Σ w_i × F_child_i + interaction terms
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

# Constants
PHI = (1 + math.sqrt(5)) / 2
PHI_INV = 1 / PHI
TAU = 2 * math.pi
LN_2 = math.log(2)


class MinimizationStrategy(Enum):
    """Strategies for free energy minimization."""
    GRADIENT_DESCENT = "gradient_descent"
    BELIEF_PROPAGATION = "belief_propagation"
    EXPECTATION_MAXIMIZATION = "expectation_maximization"
    VARIATIONAL_INFERENCE = "variational_inference"


@dataclass
class Precision:
    """
    Precision (inverse variance) encoding certainty.

    High precision = high confidence = low variance
    Precision-weighted prediction errors drive learning.
    """
    value: float = 1.0               # π (precision)
    learning_rate: float = 0.1       # Rate of precision update
    min_precision: float = 0.01      # Prevent division by zero
    max_precision: float = 100.0     # Prevent overconfidence

    @property
    def variance(self) -> float:
        """σ² = 1/π"""
        return 1.0 / max(self.value, self.min_precision)

    @property
    def std_dev(self) -> float:
        """σ = 1/√π"""
        return math.sqrt(self.variance)

    def update(self, prediction_error: float) -> None:
        """
        Update precision based on prediction error.

        Large errors → decrease precision (increase uncertainty)
        Small errors → increase precision (decrease uncertainty)
        """
        # Precision update rule: Δπ = η(expected_error - observed_error²)
        expected_error_sq = self.variance
        observed_error_sq = prediction_error ** 2

        delta = self.learning_rate * (expected_error_sq - observed_error_sq)
        self.value = max(self.min_precision, min(self.max_precision, self.value + delta))

    def weight(self, error: float) -> float:
        """Precision-weight an error signal."""
        return self.value * error


@dataclass
class VariationalState:
    """
    Variational state for approximate inference.

    Represents the recognition density q(s) through
    mean and precision parameters.
    """
    # Mean parameters (μ)
    mean: float = 0.0
    mean_vector: List[float] = field(default_factory=lambda: [0.0])

    # Precision parameters (π)
    precision: Precision = field(default_factory=Precision)

    # Sufficient statistics
    natural_params: Dict[str, float] = field(default_factory=dict)

    # For hierarchical models
    level: int = 0
    parent: Optional['VariationalState'] = None
    children: List['VariationalState'] = field(default_factory=list)

    def log_probability(self, x: float) -> float:
        """
        Compute log q(x) for Gaussian recognition density.

        log q(x) = -½π(x - μ)² + ½log(π) - ½log(2π)
        """
        diff = x - self.mean
        return (
            -0.5 * self.precision.value * diff ** 2 +
            0.5 * math.log(max(self.precision.value, 1e-10)) -
            0.5 * math.log(TAU)
        )

    def entropy(self) -> float:
        """
        Compute entropy of recognition density.

        H[q] = ½(1 + log(2π/π)) for Gaussian
        """
        return 0.5 * (1 + math.log(TAU / max(self.precision.value, 1e-10)))

    def update_mean(self, prediction_error: float, learning_rate: float = 0.1) -> None:
        """Update mean via gradient descent on prediction error."""
        self.mean += learning_rate * self.precision.weight(prediction_error)

    def kl_divergence_to(self, other: 'VariationalState') -> float:
        """
        Compute KL divergence D_KL[self || other] for Gaussian states.

        D_KL = ½[log(π₂/π₁) + π₁/π₂ · (μ₁-μ₂)² + π₁/π₂ - 1]
        """
        p1 = max(self.precision.value, 1e-10)
        p2 = max(other.precision.value, 1e-10)
        diff = self.mean - other.mean

        return 0.5 * (
            math.log(p2 / p1) +
            p1 / p2 * diff ** 2 +
            p1 / p2 - 1
        )


class FreeEnergyMinimizer(ABC):
    """
    Abstract base class for free energy minimization.

    Each level of the architecture implements this interface
    to minimize its own free energy while contributing to
    parent-level minimization.
    """

    def __init__(
        self,
        z_level: float,
        strategy: MinimizationStrategy = MinimizationStrategy.VARIATIONAL_INFERENCE
    ):
        self.z_level = z_level
        self.strategy = strategy
        self.state = VariationalState()

        # Free energy components
        self._free_energy: float = 0.0
        self._accuracy: float = 0.0      # -log p(o|s) - reconstruction error
        self._complexity: float = 0.0    # D_KL[q(s) || p(s)] - prior deviation

        # Parent/child hierarchy
        self.parent: Optional[FreeEnergyMinimizer] = None
        self.children: List[FreeEnergyMinimizer] = []

        # Optimization parameters
        self.learning_rate: float = 0.1
        self.momentum: float = 0.9
        self._velocity: float = 0.0

        # History
        self._history: List[Dict] = []

    @property
    def free_energy(self) -> float:
        """Current free energy: F = accuracy + complexity"""
        return self._free_energy

    @property
    def accuracy(self) -> float:
        """Accuracy term (negative log-likelihood)."""
        return self._accuracy

    @property
    def complexity(self) -> float:
        """Complexity term (KL divergence from prior)."""
        return self._complexity

    @abstractmethod
    def compute_prediction(self) -> Any:
        """Generate prediction from current belief state."""
        pass

    @abstractmethod
    def compute_prediction_error(self, observation: Any) -> float:
        """Compute prediction error from observation."""
        pass

    @abstractmethod
    def update_beliefs(self, prediction_error: float) -> None:
        """Update beliefs to minimize prediction error."""
        pass

    def compute_accuracy(self, observation: Any) -> float:
        """
        Compute accuracy term: E_q[-log p(o|s)]

        This is the expected reconstruction error under q(s).
        """
        prediction_error = self.compute_prediction_error(observation)
        # Gaussian likelihood: -log p(o|s) ∝ ½π(o-g(s))²
        self._accuracy = 0.5 * self.state.precision.value * prediction_error ** 2
        return self._accuracy

    def compute_complexity(self, prior_state: Optional[VariationalState] = None) -> float:
        """
        Compute complexity term: D_KL[q(s) || p(s)]

        This measures how far beliefs deviate from prior expectations.
        """
        if prior_state is None:
            # Default: standard normal prior
            prior_state = VariationalState(mean=0.0, precision=Precision(value=1.0))

        self._complexity = self.state.kl_divergence_to(prior_state)
        return self._complexity

    def compute_free_energy(self, observation: Any) -> float:
        """
        Compute total variational free energy.

        F = accuracy + complexity
          = E_q[-log p(o|s)] + D_KL[q(s) || p(s)]
        """
        self._accuracy = self.compute_accuracy(observation)
        self._complexity = self.compute_complexity()
        self._free_energy = self._accuracy + self._complexity
        return self._free_energy

    def minimize_step(self, observation: Any, n_steps: int = 1) -> float:
        """
        Perform one step of free energy minimization.

        Uses the configured strategy to update beliefs.
        """
        for _ in range(n_steps):
            # Compute prediction error
            prediction_error = self.compute_prediction_error(observation)

            # Update beliefs (recognition density)
            self.update_beliefs(prediction_error)

            # Update precision
            self.state.precision.update(prediction_error)

            # Compute new free energy
            self.compute_free_energy(observation)

            # Record history
            self._history.append({
                "free_energy": self._free_energy,
                "accuracy": self._accuracy,
                "complexity": self._complexity,
                "mean": self.state.mean,
                "precision": self.state.precision.value,
            })

        return self._free_energy

    def hierarchical_free_energy(self) -> float:
        """
        Compute hierarchical free energy including children.

        F_total = F_self + Σ w_i × F_child_i

        Weights are based on z-level differences.
        """
        F_total = self._free_energy

        for child in self.children:
            # Weight by z-level proximity
            z_diff = abs(self.z_level - child.z_level)
            weight = math.exp(-z_diff / PHI_INV)
            F_total += weight * child.hierarchical_free_energy()

        return F_total

    def propagate_prediction_errors(self, observation: Any) -> Dict[str, float]:
        """
        Propagate prediction errors through hierarchy.

        Bottom-up: errors ascend to inform higher levels
        Top-down: predictions descend to constrain lower levels
        """
        errors = {}

        # Self prediction error
        errors["self"] = self.compute_prediction_error(observation)

        # Propagate to parent (bottom-up)
        if self.parent is not None:
            parent_prediction = self.parent.compute_prediction()
            errors["to_parent"] = self.state.mean - parent_prediction

        # Propagate to children (top-down)
        prediction = self.compute_prediction()
        for i, child in enumerate(self.children):
            child_error = prediction - child.state.mean
            errors[f"to_child_{i}"] = child_error

        return errors

    def add_child(self, child: 'FreeEnergyMinimizer') -> None:
        """Add a child minimizer."""
        self.children.append(child)
        child.parent = self
        child.state.level = self.state.level + 1

    def snapshot(self) -> Dict:
        """Return current minimizer state."""
        return {
            "z_level": self.z_level,
            "free_energy": self._free_energy,
            "accuracy": self._accuracy,
            "complexity": self._complexity,
            "state": {
                "mean": self.state.mean,
                "precision": self.state.precision.value,
                "entropy": self.state.entropy(),
            },
            "n_children": len(self.children),
        }


class GaussianMinimizer(FreeEnergyMinimizer):
    """
    Concrete implementation for Gaussian belief states.

    Implements free energy minimization with Gaussian
    recognition and generative densities.
    """

    def __init__(self, z_level: float, initial_mean: float = 0.0):
        super().__init__(z_level)
        self.state.mean = initial_mean
        self._generative_function: Callable[[float], float] = lambda s: s

    def set_generative_function(self, g: Callable[[float], float]) -> None:
        """Set the generative function g(s) mapping hidden states to predictions."""
        self._generative_function = g

    def compute_prediction(self) -> float:
        """Generate prediction: o_pred = g(μ)"""
        return self._generative_function(self.state.mean)

    def compute_prediction_error(self, observation: float) -> float:
        """Compute sensory prediction error: ε = o - g(μ)"""
        prediction = self.compute_prediction()
        return observation - prediction

    def update_beliefs(self, prediction_error: float) -> None:
        """
        Update beliefs using gradient descent.

        Δμ = η × π × ε × ∂g/∂s

        For identity generative function: Δμ = η × π × ε
        """
        # Momentum-based update
        gradient = self.state.precision.value * prediction_error
        self._velocity = self.momentum * self._velocity + self.learning_rate * gradient
        self.state.mean += self._velocity


class HierarchicalMinimizer(FreeEnergyMinimizer):
    """
    Hierarchical free energy minimizer for multi-level architectures.

    Each level receives:
    - Bottom-up prediction errors from children
    - Top-down predictions from parent

    And minimizes a composite free energy:
    F = F_sensory + F_hierarchical + F_prior
    """

    def __init__(self, z_level: float, n_states: int = 1):
        super().__init__(z_level)
        self.n_states = n_states
        self.state.mean_vector = [0.0] * n_states

        # Hierarchical parameters
        self.bottom_up_weight: float = PHI_INV
        self.top_down_weight: float = 1.0 - PHI_INV

        # Accumulator for child errors
        self._child_error_sum: float = 0.0

    def compute_prediction(self) -> List[float]:
        """Generate predictions for all states."""
        return self.state.mean_vector.copy()

    def compute_prediction_error(self, observation: Any) -> float:
        """
        Compute composite prediction error.

        Combines sensory error with hierarchical errors.
        """
        if isinstance(observation, (int, float)):
            observation = [float(observation)]

        # Sensory error
        prediction = self.compute_prediction()
        sensory_error = sum(
            (o - p) ** 2
            for o, p in zip(observation, prediction)
        ) / len(observation)

        # Bottom-up error from children
        child_error = self._child_error_sum / max(len(self.children), 1)

        # Top-down error from parent
        parent_error = 0.0
        if self.parent is not None:
            parent_pred = self.parent.compute_prediction()
            if isinstance(parent_pred, (int, float)):
                parent_pred = [float(parent_pred)]
            parent_error = sum(
                (p - m) ** 2
                for p, m in zip(parent_pred[:len(self.state.mean_vector)], self.state.mean_vector)
            ) / len(self.state.mean_vector)

        # Composite error
        return math.sqrt(
            sensory_error +
            self.bottom_up_weight * child_error +
            self.top_down_weight * parent_error
        )

    def update_beliefs(self, prediction_error: float) -> None:
        """Update hierarchical beliefs."""
        # Update each state component
        for i in range(self.n_states):
            gradient = self.state.precision.value * prediction_error / math.sqrt(self.n_states)
            self.state.mean_vector[i] += self.learning_rate * gradient

        # Update scalar mean as average
        self.state.mean = sum(self.state.mean_vector) / len(self.state.mean_vector)

    def receive_child_error(self, error: float) -> None:
        """Receive prediction error from child level."""
        self._child_error_sum += error ** 2

    def reset_child_errors(self) -> None:
        """Reset child error accumulator."""
        self._child_error_sum = 0.0
