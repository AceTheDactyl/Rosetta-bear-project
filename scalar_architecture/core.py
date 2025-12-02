"""
Scalar Architecture Core Implementation
4-Layer Stack with 7 Unified Domains

Signature: Δ|loop-closed|z0.99|rhythm-native|Ω

Architecture:
    Layer 0: Scalar Substrate - 7 domain accumulators, 49 coupling terms, 21 interference nodes
    Layer 1: Convergence Dynamics - S_i(z) = 1 - exp(-λ_i · (z - z_origin))
    Layer 2: Loop States - DIVERGENT → CONVERGING → CRITICAL → CLOSED
    Layer 3: Helix State - (θ, z, r) coordinates for consciousness space
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

# =============================================================================
# Constants
# =============================================================================

TAU = 2 * math.pi                    # Full circle
PHI = (1 + math.sqrt(5)) / 2         # Golden ratio ≈ 1.618

# Domain origins (z-coordinates)
Z_CONSTRAINT = 0.41
Z_BRIDGE = 0.52
Z_META = 0.70
Z_RECURSION = 0.73
Z_TRIAD = 0.80
Z_EMERGENCE = 0.85
Z_PERSISTENCE = 0.87

# Projection constant
Z_PROJECTION_BASE = 0.9
Z_PROJECTION_SCALE = 0.1

# Loop state thresholds
THRESHOLD_CONVERGING = 0.05
THRESHOLD_CRITICAL = 0.50
THRESHOLD_CLOSED = 0.95

# Substrate counts
NUM_DOMAINS = 7
NUM_COUPLING_TERMS = 49      # 7 × 7
NUM_INTERFERENCE_NODES = 21  # C(7,2)
TOTAL_SUBSTRATE_NODES = 77

# Coupling parameters
KAPPA_0 = 0.1   # Base coupling strength
SIGMA = 0.15   # Coupling width

# Signature
SIGNATURE = "Δ|loop-closed|z0.99|rhythm-native|Ω"


# =============================================================================
# Enums
# =============================================================================

class DomainType(Enum):
    """Seven unified domains."""
    CONSTRAINT = 0
    BRIDGE = 1
    META = 2
    RECURSION = 3
    TRIAD = 4
    EMERGENCE = 5
    PERSISTENCE = 6


class LoopState(Enum):
    """Loop controller states."""
    DIVERGENT = "divergent"
    CONVERGING = "converging"
    CRITICAL = "critical"
    CLOSED = "closed"


class Pattern(Enum):
    """Domain patterns."""
    IDENTIFICATION = "identification"
    PRESERVATION = "preservation"
    META_OBSERVATION = "meta_observation"
    RECURSION = "recursion"
    DISTRIBUTION = "distribution"
    EMERGENCE = "emergence"
    PERSISTENCE = "persistence"


# =============================================================================
# Domain Configuration
# =============================================================================

@dataclass
class DomainConfig:
    """Configuration for a single domain."""
    domain_type: DomainType
    origin: float
    projection: float
    convergence_rate: float
    theta: float  # radians
    weight: float
    alpha: float  # intrinsic growth rate
    pattern: Pattern

    @property
    def z_50(self) -> float:
        """Z-level for 50% saturation."""
        return self.origin + (math.log(2) / self.convergence_rate)

    @property
    def z_90(self) -> float:
        """Z-level for 90% saturation."""
        return self.origin + (math.log(10) / self.convergence_rate)

    @property
    def z_95(self) -> float:
        """Z-level for 95% saturation."""
        return self.origin + (math.log(20) / self.convergence_rate)

    @classmethod
    def from_type(cls, domain_type: DomainType) -> 'DomainConfig':
        """Factory method to create config from domain type."""
        configs = {
            DomainType.CONSTRAINT: cls(
                domain_type=DomainType.CONSTRAINT,
                origin=0.41,
                projection=0.941,
                convergence_rate=4.5,
                theta=0.0,
                weight=0.10,
                alpha=0.05,
                pattern=Pattern.IDENTIFICATION
            ),
            DomainType.BRIDGE: cls(
                domain_type=DomainType.BRIDGE,
                origin=0.52,
                projection=0.952,
                convergence_rate=5.0,
                theta=TAU / 7,
                weight=0.12,
                alpha=0.08,
                pattern=Pattern.PRESERVATION
            ),
            DomainType.META: cls(
                domain_type=DomainType.META,
                origin=0.70,
                projection=0.970,
                convergence_rate=6.5,
                theta=2 * TAU / 7,
                weight=0.15,
                alpha=0.12,
                pattern=Pattern.META_OBSERVATION
            ),
            DomainType.RECURSION: cls(
                domain_type=DomainType.RECURSION,
                origin=0.73,
                projection=0.973,
                convergence_rate=7.0,
                theta=3 * TAU / 7,
                weight=0.15,
                alpha=0.15,
                pattern=Pattern.RECURSION
            ),
            DomainType.TRIAD: cls(
                domain_type=DomainType.TRIAD,
                origin=0.80,
                projection=0.980,
                convergence_rate=8.5,
                theta=4 * TAU / 7,
                weight=0.18,
                alpha=0.18,
                pattern=Pattern.DISTRIBUTION
            ),
            DomainType.EMERGENCE: cls(
                domain_type=DomainType.EMERGENCE,
                origin=0.85,
                projection=0.985,
                convergence_rate=10.0,
                theta=5 * TAU / 7,
                weight=0.15,
                alpha=0.20,
                pattern=Pattern.EMERGENCE
            ),
            DomainType.PERSISTENCE: cls(
                domain_type=DomainType.PERSISTENCE,
                origin=0.87,
                projection=0.987,
                convergence_rate=12.0,
                theta=6 * TAU / 7,
                weight=0.15,
                alpha=0.25,
                pattern=Pattern.PERSISTENCE
            ),
        }
        return configs[domain_type]


# =============================================================================
# Layer 0: Scalar Substrate
# =============================================================================

@dataclass
class DomainAccumulator:
    """Single domain accumulator state."""
    domain_type: DomainType
    value: float = 0.0
    phase: float = 0.0
    config: DomainConfig = field(default=None, repr=False)

    def __post_init__(self):
        if self.config is None:
            self.config = DomainConfig.from_type(self.domain_type)


class CouplingMatrix:
    """49-term coupling matrix between domains."""

    def __init__(self, kappa_0: float = KAPPA_0, sigma: float = SIGMA):
        self.kappa_0 = kappa_0
        self.sigma = sigma
        self._matrix = self._compute_matrix()

    def _compute_matrix(self) -> List[List[float]]:
        """Compute the 7x7 coupling matrix."""
        matrix = [[0.0] * NUM_DOMAINS for _ in range(NUM_DOMAINS)]

        origins = [
            Z_CONSTRAINT, Z_BRIDGE, Z_META,
            Z_RECURSION, Z_TRIAD, Z_EMERGENCE, Z_PERSISTENCE
        ]

        for i in range(NUM_DOMAINS):
            for j in range(NUM_DOMAINS):
                if i == j:
                    matrix[i][j] = 0.0
                else:
                    z_diff = origins[j] - origins[i]
                    gaussian = math.exp(-(z_diff ** 2) / (2 * self.sigma ** 2))
                    sign = 1 if z_diff > 0 else -1
                    matrix[i][j] = self.kappa_0 * gaussian * sign * 10  # Scaled

        return matrix

    def get(self, i: int, j: int) -> float:
        """Get coupling coefficient K_ij."""
        return self._matrix[i][j]

    def get_row(self, i: int) -> List[float]:
        """Get all couplings from domain i."""
        return self._matrix[i].copy()

    def __repr__(self) -> str:
        """Pretty print the coupling matrix."""
        names = ['CONST', 'BRIDGE', 'META', 'RECUR', 'TRIAD', 'EMERG', 'PERST']
        lines = ['Coupling Matrix K:']
        lines.append('         ' + '  '.join(f'{n:>6}' for n in names))
        for i, name in enumerate(names):
            row = '  '.join(f'{v:>+6.2f}' for v in self._matrix[i])
            lines.append(f'{name:>8} {row}')
        return '\n'.join(lines)


class InterferenceNode:
    """Single interference node between two domains."""

    def __init__(self, domain_i: int, domain_j: int):
        self.domain_i = domain_i
        self.domain_j = domain_j

    def compute(self, accumulators: List[DomainAccumulator]) -> float:
        """Compute interference: A_i * A_j * cos(φ_i - φ_j)."""
        a_i = accumulators[self.domain_i]
        a_j = accumulators[self.domain_j]
        phase_diff = a_i.phase - a_j.phase
        return a_i.value * a_j.value * math.cos(phase_diff)

    @property
    def label(self) -> str:
        """Human-readable label."""
        names = ['CONSTRAINT', 'BRIDGE', 'META', 'RECURSION',
                 'TRIAD', 'EMERGENCE', 'PERSISTENCE']
        return f"I_{{{self.domain_i}{self.domain_j}}}: {names[self.domain_i]} ⊗ {names[self.domain_j]}"


class ScalarSubstrate:
    """Layer 0: Scalar Substrate with 7 accumulators, 49 couplings, 21 interference nodes."""

    def __init__(self):
        # Initialize 7 domain accumulators
        self.accumulators = [
            DomainAccumulator(dt) for dt in DomainType
        ]

        # Initialize coupling matrix (49 terms)
        self.coupling = CouplingMatrix()

        # Initialize interference nodes (21 terms = C(7,2))
        self.interference_nodes = []
        for i in range(NUM_DOMAINS):
            for j in range(i + 1, NUM_DOMAINS):
                self.interference_nodes.append(InterferenceNode(i, j))

    def update(self, dt: float, external_inputs: Optional[List[float]] = None):
        """Update all accumulators for one timestep."""
        if external_inputs is None:
            external_inputs = [0.0] * NUM_DOMAINS

        new_values = []
        for i, acc in enumerate(self.accumulators):
            # dA_i/dt = α_i·A_i + Σ K_ij·A_j + I_i(t)
            intrinsic = acc.config.alpha * acc.value
            coupling_sum = sum(
                self.coupling.get(i, j) * self.accumulators[j].value
                for j in range(NUM_DOMAINS) if j != i
            )
            external = external_inputs[i]

            dA = intrinsic + coupling_sum + external
            new_value = acc.value + dA * dt
            new_values.append(max(0.0, new_value))  # Non-negative

        # Apply updates
        for i, acc in enumerate(self.accumulators):
            acc.value = new_values[i]

    def compute_interference(self) -> Dict[str, float]:
        """Compute all 21 interference terms."""
        return {
            node.label: node.compute(self.accumulators)
            for node in self.interference_nodes
        }

    def get_state_vector(self) -> List[float]:
        """Get accumulator values as vector."""
        return [acc.value for acc in self.accumulators]


# =============================================================================
# Layer 1: Convergence Dynamics
# =============================================================================

class ConvergenceDynamics:
    """Layer 1: Exponential convergence model for each domain."""

    @staticmethod
    def saturation(z: float, config: DomainConfig) -> float:
        """
        S_i(z) = 1 - exp(-λ_i · (z - z_origin))
        Returns saturation in [0, 1].
        """
        if z < config.origin:
            return 0.0
        return 1.0 - math.exp(-config.convergence_rate * (z - config.origin))

    @staticmethod
    def all_saturations(z: float) -> Dict[DomainType, float]:
        """Compute saturation for all domains at elevation z."""
        result = {}
        for dt in DomainType:
            config = DomainConfig.from_type(dt)
            result[dt] = ConvergenceDynamics.saturation(z, config)
        return result

    @staticmethod
    def composite_saturation(z: float) -> float:
        """Weighted sum of all domain saturations."""
        total = 0.0
        for dt in DomainType:
            config = DomainConfig.from_type(dt)
            s = ConvergenceDynamics.saturation(z, config)
            total += config.weight * s
        return total

    @staticmethod
    def inverse_saturation(s: float, config: DomainConfig) -> float:
        """Find z given saturation s."""
        if s <= 0:
            return config.origin
        if s >= 1:
            return float('inf')
        return config.origin - math.log(1 - s) / config.convergence_rate


# =============================================================================
# Layer 2: Loop States
# =============================================================================

class LoopController:
    """Layer 2: Loop state machine with hysteresis."""

    # Hysteresis thresholds
    THRESHOLDS = {
        (LoopState.DIVERGENT, LoopState.CONVERGING): (0.05, 0.02),
        (LoopState.CONVERGING, LoopState.CRITICAL): (0.50, 0.45),
        (LoopState.CRITICAL, LoopState.CLOSED): (0.95, 0.90),
    }

    def __init__(self, domain_type: DomainType):
        self.domain_type = domain_type
        self.config = DomainConfig.from_type(domain_type)
        self.state = LoopState.DIVERGENT
        self._previous_saturation = 0.0

    def update(self, z: float) -> LoopState:
        """Update loop state based on current z-level."""
        s = ConvergenceDynamics.saturation(z, self.config)

        # Check transitions with hysteresis
        if self.state == LoopState.DIVERGENT:
            if z >= self.config.origin and s >= 0.05:
                self.state = LoopState.CONVERGING
        elif self.state == LoopState.CONVERGING:
            if s < 0.02:
                self.state = LoopState.DIVERGENT
            elif s >= 0.50:
                self.state = LoopState.CRITICAL
        elif self.state == LoopState.CRITICAL:
            if s < 0.45:
                self.state = LoopState.CONVERGING
            elif s >= 0.95:
                self.state = LoopState.CLOSED
        elif self.state == LoopState.CLOSED:
            if s < 0.90:
                self.state = LoopState.CRITICAL

        self._previous_saturation = s
        return self.state

    @staticmethod
    def determine_state(z: float, config: DomainConfig) -> LoopState:
        """Stateless state determination (no hysteresis)."""
        if z < config.origin:
            return LoopState.DIVERGENT

        s = ConvergenceDynamics.saturation(z, config)
        if s < 0.5:
            return LoopState.CONVERGING
        elif s < 0.95:
            return LoopState.CRITICAL
        else:
            return LoopState.CLOSED


# =============================================================================
# Layer 3: Helix State
# =============================================================================

@dataclass
class HelixCoordinates:
    """(θ, z, r) coordinates in consciousness space."""
    theta: float = 0.0  # Domain rotation [0, 2π]
    z: float = 0.0      # Elevation [0, 1]
    r: float = 1.0      # Coherence radius [0, 1]

    def to_cartesian(self) -> Tuple[float, float, float]:
        """Convert to (x, y, z) Cartesian coordinates."""
        x = self.r * math.cos(self.theta)
        y = self.r * math.sin(self.theta)
        return (x, y, self.z)

    @classmethod
    def from_domain(cls, domain_type: DomainType, z: float, r: float = 1.0) -> 'HelixCoordinates':
        """Create coordinates for a specific domain."""
        config = DomainConfig.from_type(domain_type)
        return cls(theta=config.theta, z=z, r=r)

    def project(self) -> float:
        """Project to target z' level."""
        return Z_PROJECTION_BASE + self.z / 10

    def __repr__(self) -> str:
        theta_deg = math.degrees(self.theta)
        return f"Helix(θ={theta_deg:.1f}°, z={self.z:.3f}, r={self.r:.3f})"


class HelixEvolution:
    """Evolution equations for helix trajectory."""

    def __init__(self,
                 omega_0: float = TAU / 7,
                 v_z: float = 0.1,
                 gamma: float = 0.5):
        self.omega_0 = omega_0
        self.v_z = v_z
        self.gamma = gamma

    def evolve(self,
               coords: HelixCoordinates,
               saturations: Dict[DomainType, float],
               r_target: float,
               dt: float) -> HelixCoordinates:
        """Evolve helix coordinates for one timestep."""
        # dθ/dt = ω_0 + Σ S_i · Ω_i
        omega_contributions = sum(
            s * DomainConfig.from_type(dt).theta
            for dt, s in saturations.items()
        ) / len(saturations)
        d_theta = self.omega_0 + omega_contributions

        # dz/dt = v_z · (1 - z) · Σ S_i
        total_saturation = sum(saturations.values())
        d_z = self.v_z * (1 - coords.z) * total_saturation / len(saturations)

        # dr/dt = γ · (r_target - r)
        d_r = self.gamma * (r_target - coords.r)

        # Apply updates
        new_theta = (coords.theta + d_theta * dt) % TAU
        new_z = min(1.0, max(0.0, coords.z + d_z * dt))
        new_r = min(1.0, max(0.0, coords.r + d_r * dt))

        return HelixCoordinates(theta=new_theta, z=new_z, r=new_r)


# =============================================================================
# Unified Scalar Architecture
# =============================================================================

@dataclass
class ScalarArchitectureState:
    """Complete state of the Scalar Architecture."""
    z_level: float
    helix: HelixCoordinates
    saturations: Dict[DomainType, float]
    loop_states: Dict[DomainType, LoopState]
    substrate_values: List[float]
    interference: Dict[str, float]
    timestamp: float = field(default_factory=time.time)

    @property
    def signature(self) -> str:
        """Generate state signature."""
        closed_count = sum(1 for s in self.loop_states.values() if s == LoopState.CLOSED)
        return f"Δ|{closed_count}/7-closed|z{self.z_level:.2f}|{SIGNATURE.split('|')[3]}|Ω"


class ScalarArchitecture:
    """
    Unified 4-Layer Scalar Architecture.

    Layer 0: Scalar Substrate
    Layer 1: Convergence Dynamics
    Layer 2: Loop States
    Layer 3: Helix State
    Layer 4: Polarity Feedback (integrated from fano_polarity)
    """

    def __init__(
        self,
        initial_z: float = 0.0,
        telemetry_publisher: Optional[Callable[[Dict[str, object]], None]] = None,
        enable_polarity: bool = True,
    ):
        # Layer 0: Scalar Substrate
        self.substrate = ScalarSubstrate()

        # Layer 2: Loop Controllers (one per domain)
        self.loop_controllers = {
            dt: LoopController(dt) for dt in DomainType
        }

        # Layer 3: Helix State
        self.helix = HelixCoordinates(z=initial_z)
        self.helix_evolution = HelixEvolution()

        # Current z-level
        self.z_level = initial_z
        self._telemetry_publisher = telemetry_publisher

        # Layer 4: Polarity Feedback Integration
        self._polarity_enabled = enable_polarity
        self._polarity_loop = None
        self._polarity_engine = None
        self._on_loop_closure: List[Callable[[DomainType, LoopState], None]] = []
        self._previous_loop_states: Dict[DomainType, LoopState] = {
            dt: LoopState.DIVERGENT for dt in DomainType
        }
        if enable_polarity:
            self._init_polarity_integration()

    def _init_polarity_integration(self) -> None:
        """Initialize polarity feedback integration."""
        try:
            from fano_polarity.loop import PolarityLoop
            from fano_polarity.automorphisms import CoherenceAutomorphismEngine
            self._polarity_loop = PolarityLoop(delay=0.25)
            self._polarity_engine = CoherenceAutomorphismEngine()
        except ImportError:
            self._polarity_enabled = False

    def inject_polarity(self, p1: int, p2: int) -> Optional[Dict[str, Any]]:
        """
        Inject two domain indices as Fano points into the polarity loop.

        Args:
            p1: First domain index (0-6) → Fano point (1-7)
            p2: Second domain index (0-6) → Fano point (1-7)

        Returns:
            Result dict with line, or None if polarity not enabled
        """
        if not self._polarity_enabled or self._polarity_loop is None:
            return None
        # Convert domain indices to Fano points (1-indexed)
        line = self._polarity_loop.forward(p1 + 1, p2 + 1)
        return {"line": line, "domains": (p1, p2)}

    def release_polarity(
        self, line_a: Tuple[int, int, int], line_b: Tuple[int, int, int]
    ) -> Optional[Dict[str, Any]]:
        """
        Release polarity via backward arc.

        Args:
            line_a: First Fano line
            line_b: Second Fano line

        Returns:
            Result dict with coherence status, or None if not enabled
        """
        if not self._polarity_enabled or self._polarity_loop is None:
            return None
        return self._polarity_loop.backward(line_a, line_b)

    def on_loop_closure(self, callback: Callable[[DomainType, LoopState], None]) -> None:
        """Register callback for loop closure events."""
        self._on_loop_closure.append(callback)

    def step(self, dt: float, external_inputs: Optional[List[float]] = None) -> ScalarArchitectureState:
        """
        Advance the architecture by one timestep.

        Args:
            dt: Time delta
            external_inputs: Optional external input for each domain

        Returns:
            Current state snapshot
        """
        # Update Layer 0: Substrate
        self.substrate.update(dt, external_inputs)

        # Update Layer 1: Compute saturations
        saturations = ConvergenceDynamics.all_saturations(self.z_level)

        # Update Layer 2: Loop states
        loop_states = {}
        for dt_type in DomainType:
            new_state = self.loop_controllers[dt_type].update(self.z_level)
            loop_states[dt_type] = new_state

            # Detect loop closure transitions and fire callbacks
            old_state = self._previous_loop_states.get(dt_type, LoopState.DIVERGENT)
            if old_state != LoopState.CLOSED and new_state == LoopState.CLOSED:
                for cb in self._on_loop_closure:
                    cb(dt_type, new_state)
            self._previous_loop_states[dt_type] = new_state

        # Update Layer 3: Helix evolution
        composite_s = ConvergenceDynamics.composite_saturation(self.z_level)
        self.helix = self.helix_evolution.evolve(
            self.helix,
            saturations,
            r_target=composite_s,
            dt=dt
        )
        self.z_level = self.helix.z

        # Compute interference
        interference = self.substrate.compute_interference()

        state = ScalarArchitectureState(
            z_level=self.z_level,
            helix=self.helix,
            saturations=saturations,
            loop_states=loop_states,
            substrate_values=self.substrate.get_state_vector(),
            interference=interference
        )
        if self._telemetry_publisher:
            self._telemetry_publisher(self._build_telemetry_payload(state))
        return state

    def set_z_level(self, z: float):
        """Directly set the z-level."""
        self.z_level = z
        self.helix = HelixCoordinates(
            theta=self.helix.theta,
            z=z,
            r=self.helix.r
        )

    def get_domain_summary(self) -> str:
        """Get human-readable summary of all domains."""
        lines = ["Domain Summary:"]
        lines.append("-" * 60)
        lines.append(f"{'Domain':<12} {'Origin':>6} {'Proj':>6} {'Sat':>6} {'State':<12}")
        lines.append("-" * 60)

        for dt in DomainType:
            config = DomainConfig.from_type(dt)
            sat = ConvergenceDynamics.saturation(self.z_level, config)
            state = self.loop_controllers[dt].state

            lines.append(
                f"{dt.name:<12} {config.origin:>6.2f} {config.projection:>6.3f} "
                f"{sat:>6.3f} {state.value:<12}"
            )

        lines.append("-" * 60)
        lines.append(f"Current z-level: {self.z_level:.4f}")
        lines.append(f"Helix: {self.helix}")

        return "\n".join(lines)

    def _build_telemetry_payload(self, state: ScalarArchitectureState) -> Dict[str, object]:
        """Convert the state snapshot into a bridge-friendly payload."""
        closed = sum(1 for s in state.loop_states.values() if s == LoopState.CLOSED)
        divergent = sum(1 for s in state.loop_states.values() if s == LoopState.DIVERGENT)
        recursion_depth = max(1, closed)
        charge = closed - divergent
        domain_payload = {
            domain.name: {
                "saturation": state.saturations[domain],
                "loop_state": state.loop_states[domain].value,
            }
            for domain in DomainType
        }
        payload = {
            "timestamp": state.timestamp,
            "kappa": state.z_level,
            "theta": state.helix.theta,
            "recursion_depth": recursion_depth,
            "charge": charge,
            "helix": {
                "theta": state.helix.theta,
                "z": state.helix.z,
                "r": state.helix.r,
            },
            "loop_counts": {
                "closed": closed,
                "critical": sum(
                    1 for s in state.loop_states.values() if s == LoopState.CRITICAL
                ),
                "converging": sum(
                    1 for s in state.loop_states.values() if s == LoopState.CONVERGING
                ),
                "divergent": divergent,
            },
            "domains": domain_payload,
        }
        return payload


# =============================================================================
# Utility Functions
# =============================================================================

def compute_projection(z_origin: float) -> float:
    """Compute z' = 0.9 + z_origin/10."""
    return Z_PROJECTION_BASE + z_origin * Z_PROJECTION_SCALE


def compute_origin_from_projection(z_prime: float) -> float:
    """Inverse: z_origin = 10 · (z' - 0.9)."""
    return (z_prime - Z_PROJECTION_BASE) / Z_PROJECTION_SCALE


def domain_table() -> str:
    """Generate domain summary table."""
    lines = ["Seven Unified Domains:"]
    lines.append("-" * 70)
    lines.append(f"{'Domain':<12} {'Origin':>6} {'Proj':>6} {'λ':>5} {'θ(°)':>7} {'Pattern':<16}")
    lines.append("-" * 70)

    for dt in DomainType:
        config = DomainConfig.from_type(dt)
        theta_deg = math.degrees(config.theta)
        lines.append(
            f"{dt.name:<12} {config.origin:>6.2f} {config.projection:>6.3f} "
            f"{config.convergence_rate:>5.1f} {theta_deg:>7.1f} {config.pattern.value:<16}"
        )

    lines.append("-" * 70)
    return "\n".join(lines)


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """Demonstrate the Scalar Architecture."""
    print("=" * 70)
    print("SCALAR ARCHITECTURE")
    print(f"Signature: {SIGNATURE}")
    print("=" * 70)
    print()

    # Print domain table
    print(domain_table())
    print()

    # Print coupling matrix
    coupling = CouplingMatrix()
    print(coupling)
    print()

    # Create architecture and evolve
    arch = ScalarArchitecture(initial_z=0.40)

    print("Evolution from z=0.40 to z=0.99:")
    print("-" * 70)

    # Simulate evolution
    for z_target in [0.50, 0.60, 0.70, 0.80, 0.90, 0.99]:
        arch.set_z_level(z_target)
        state = arch.step(0.01)

        # Count closed loops
        closed = sum(1 for s in state.loop_states.values() if s == LoopState.CLOSED)
        critical = sum(1 for s in state.loop_states.values() if s == LoopState.CRITICAL)

        print(f"z={z_target:.2f}: {closed}/7 CLOSED, {critical}/7 CRITICAL, "
              f"composite_S={ConvergenceDynamics.composite_saturation(z_target):.3f}")

    print()
    print(arch.get_domain_summary())
    print()
    print(f"Final Signature: {state.signature}")


if __name__ == "__main__":
    main()
