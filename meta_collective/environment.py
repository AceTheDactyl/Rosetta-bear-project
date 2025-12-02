#!/usr/bin/env python3
"""
ENVIRONMENT INTERFACE - Active Inference Environment Abstraction
================================================================

Provides abstract environment interface for Tool agents to interact with.
Enables perception-action loop closure through observation generation.

Physics Integration:
    - Observations carry precision (confidence) estimates
    - Actions map to physical system operations
    - Reward signals guide policy learning

Signature: Δ|environment|z0.867|interface|Ω
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, TypeVar
from enum import Enum
import numpy as np

# Type variable for generic action types
ActionT = TypeVar('ActionT')


class EnvironmentState(Enum):
    """Environment operational states."""
    IDLE = "idle"                       # Awaiting action
    PROCESSING = "processing"           # Executing action
    RESPONDING = "responding"           # Generating observation
    ERROR = "error"                     # Error state


@dataclass
class Observation:
    """
    Observation received from environment after action execution.
    Carries sensory data with precision estimate for weighting.
    """
    data: np.ndarray                    # Observation vector (N-dimensional)
    precision: float                    # Confidence in observation [0, 1]
    timestamp: float                    # When observation occurred
    source: str                         # Environment region/component
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def dimension(self) -> int:
        """Dimensionality of observation vector."""
        return len(self.data)

    def weighted_data(self) -> np.ndarray:
        """Return precision-weighted observation."""
        return self.data * self.precision


@dataclass
class ActionResult:
    """
    Result of executing an action in the environment.
    Contains observation, reward, and episode status.
    """
    observation: Observation            # Resulting observation
    reward: float                       # Scalar reward signal
    done: bool                          # Episode termination flag
    truncated: bool = False             # Time limit reached
    info: Dict[str, Any] = field(default_factory=dict)


class Environment(ABC):
    """
    Abstract base class for active inference environments.

    Subclass this to create specific environment types that Tools
    can interact with through the perception-action cycle.

    Implementation Requirements:
        - reset(): Initialize environment, return first observation
        - step(): Execute action, return (observation, reward, done)
        - get_observation_space(): Describe observation structure
        - get_action_space(): Describe valid actions
    """

    def __init__(self):
        """Initialize environment state tracking."""
        self.state = EnvironmentState.IDLE
        self._step_count = 0
        self._episode_count = 0

    @abstractmethod
    def reset(self, seed: Optional[int] = None) -> Observation:
        """
        Reset environment to initial state.

        Args:
            seed: Random seed for reproducibility

        Returns:
            Initial observation from reset state
        """
        pass

    @abstractmethod
    def step(self, action: ActionT) -> ActionResult:
        """
        Execute action and return result.

        The core interface method - takes an action from an agent,
        applies it to the environment, and returns the resulting
        observation along with reward and status information.

        Args:
            action: Action to execute (type depends on environment)

        Returns:
            ActionResult with observation, reward, done flag
        """
        pass

    @abstractmethod
    def get_observation_space(self) -> Dict[str, Any]:
        """
        Return specification of observation space.

        Should describe:
            - type: "continuous", "discrete", "hybrid"
            - shape: Observation vector dimensions
            - low/high: Value bounds (for continuous)
            - values: Valid values (for discrete)
        """
        pass

    @abstractmethod
    def get_action_space(self) -> Dict[str, Any]:
        """
        Return specification of valid actions.

        Should describe:
            - type: "continuous", "discrete", "hybrid"
            - actions: List of action names/types
            - parameters: Per-action parameter specs
        """
        pass

    def render(self, mode: str = "human") -> Optional[Any]:
        """
        Render environment state for visualization.
        Optional - override in subclasses that support rendering.
        """
        return None

    def close(self) -> None:
        """
        Clean up environment resources.
        Override if environment holds external resources.
        """
        pass

    @property
    def step_count(self) -> int:
        """Number of steps taken in current episode."""
        return self._step_count

    @property
    def episode_count(self) -> int:
        """Number of completed episodes."""
        return self._episode_count


class CompositeEnvironment(Environment):
    """
    Environment composed of multiple sub-environments.
    Aggregates observations and routes actions appropriately.
    """

    def __init__(self, environments: Dict[str, Environment]):
        """
        Initialize with named sub-environments.

        Args:
            environments: Dict mapping names to Environment instances
        """
        super().__init__()
        self.environments = environments

    def reset(self, seed: Optional[int] = None) -> Observation:
        """Reset all sub-environments and combine observations."""
        observations = {}
        for name, env in self.environments.items():
            obs = env.reset(seed=seed)
            observations[name] = obs.data

        # Concatenate all observation vectors
        combined_data = np.concatenate(list(observations.values()))
        mean_precision = np.mean([
            self.environments[n].reset().precision
            for n in self.environments
        ])

        self._step_count = 0
        return Observation(
            data=combined_data,
            precision=mean_precision,
            timestamp=0.0,
            source="composite",
            metadata={"sub_observations": observations},
        )

    def step(self, action: Dict[str, Any]) -> ActionResult:
        """
        Route action to appropriate sub-environment.

        Args:
            action: Dict with 'target' (env name) and 'action' (actual action)
        """
        target = action.get("target", list(self.environments.keys())[0])
        sub_action = action.get("action")

        if target not in self.environments:
            raise ValueError(f"Unknown environment: {target}")

        result = self.environments[target].step(sub_action)
        self._step_count += 1

        return result

    def get_observation_space(self) -> Dict[str, Any]:
        """Return combined observation space."""
        spaces = {
            name: env.get_observation_space()
            for name, env in self.environments.items()
        }
        return {"type": "composite", "spaces": spaces}

    def get_action_space(self) -> Dict[str, Any]:
        """Return combined action space."""
        spaces = {
            name: env.get_action_space()
            for name, env in self.environments.items()
        }
        return {"type": "composite", "spaces": spaces}


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_observation(
    data: np.ndarray,
    precision: float = 1.0,
    source: str = "unknown",
) -> Observation:
    """
    Create observation with default values.

    Args:
        data: Observation vector
        precision: Confidence level
        source: Origin identifier
    """
    return Observation(
        data=data,
        precision=precision,
        timestamp=0.0,
        source=source,
    )


def create_action_result(
    observation: Observation,
    reward: float = 0.0,
    done: bool = False,
) -> ActionResult:
    """
    Create action result with default values.

    Args:
        observation: Resulting observation
        reward: Reward signal
        done: Episode termination
    """
    return ActionResult(
        observation=observation,
        reward=reward,
        done=done,
    )
