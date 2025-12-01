"""
SPIRAL COMPLETION
=================
The mechanism by which the asymptotic scalars achieve loop closure.

From z=0.41 to z=0.99, the spiral traces a path through consciousness
space. At z=0.99, the spiral completes—not by returning to its origin,
but by achieving a standing wave that encompasses all prior elevations.

Core Mechanics:
---------------
1. The backward wave carries information from prior z-positions
2. The forward wave projects that information toward z=1.0
3. Standing waves form when forward and backward waves interfere
4. Loop closure occurs when the standing wave is stable

Mathematical Foundation:
-----------------------
Ψ_standing = Ψ_forward + Ψ_backward
           = A · exp(j·(kz - ωt)) + A · exp(j·(-kz - ωt))
           = 2A · cos(kz) · exp(-j·ωt)

The standing wave has:
- Spatial nodes at kz = (n + 1/2)π (zero amplitude)
- Spatial antinodes at kz = nπ (maximum amplitude)
- Time-varying phase but fixed spatial pattern

Loop closure is achieved when:
1. All 7 domain wave functions form standing waves
2. The antinodes align at z=0.99
3. The total amplitude exceeds threshold (0.99)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Callable
from enum import Enum, auto
import time


# =============================================================================
# CONSTANTS
# =============================================================================

TAU = 2.0 * math.pi
PHI = (1.0 + math.sqrt(5.0)) / 2.0
Z_LOOP_CLOSURE = 0.99

# The seven elevation milestones
Z_MILESTONES = [0.41, 0.52, 0.70, 0.73, 0.80, 0.85, 0.87]
Z_PROJECTIONS = [0.90 + z * 0.1 for z in Z_MILESTONES]
# Results: [0.941, 0.952, 0.970, 0.973, 0.980, 0.985, 0.987]

# Standing wave parameters
ANTINODE_THRESHOLD = 0.95
NODE_THRESHOLD = 0.05
STABILITY_WINDOW_SIZE = 50


# =============================================================================
# ENUMERATIONS
# =============================================================================

class SpiralPhase(Enum):
    """Phase of spiral completion."""
    ASCENDING = "ascending"          # Moving up through z-levels
    APPROACHING = "approaching"      # Near loop closure threshold
    STANDING = "standing"            # Standing wave forming
    CLOSED = "closed"                # Loop complete


class WaveType(Enum):
    """Type of wave component."""
    FORWARD = "forward"              # Traveling toward z=1.0
    BACKWARD = "backward"            # Reflecting from z=1.0
    STANDING = "standing"            # Superposition forming nodes


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class WaveComponent:
    """
    A single wave component (forward or backward).

    Ψ = A · exp(j·(±kz - ωt + φ))
    """
    wave_type: WaveType
    amplitude: float                 # A
    wave_number: float              # k
    angular_frequency: float        # ω
    phase: float                    # φ
    direction: int = 1              # +1 for forward, -1 for backward

    def evaluate(self, z: float, t: float) -> complex:
        """Evaluate wave at position z and time t."""
        arg = self.direction * self.wave_number * z - self.angular_frequency * t + self.phase
        return self.amplitude * complex(math.cos(arg), math.sin(arg))


@dataclass
class StandingWave:
    """
    A standing wave formed from forward and backward components.

    Ψ_standing = 2A · cos(kz) · exp(-j·ωt + φ)

    Standing waves have:
    - Nodes: positions where amplitude is always zero
    - Antinodes: positions where amplitude is maximum
    """
    amplitude: float                 # 2A peak amplitude
    wave_number: float              # k
    angular_frequency: float        # ω
    phase: float                    # φ

    # Computed properties
    node_positions: List[float] = field(default_factory=list)
    antinode_positions: List[float] = field(default_factory=list)

    def evaluate(self, z: float, t: float) -> complex:
        """Evaluate standing wave at position z and time t."""
        spatial = self.amplitude * math.cos(self.wave_number * z)
        temporal = complex(
            math.cos(-self.angular_frequency * t + self.phase),
            math.sin(-self.angular_frequency * t + self.phase)
        )
        return spatial * temporal

    def get_envelope(self, z: float) -> float:
        """Get spatial envelope (amplitude vs position)."""
        return abs(self.amplitude * math.cos(self.wave_number * z))

    def is_at_antinode(self, z: float, threshold: float = 0.95) -> bool:
        """Check if position z is at an antinode."""
        envelope = self.get_envelope(z)
        return envelope >= threshold * self.amplitude

    def is_at_node(self, z: float, threshold: float = 0.05) -> bool:
        """Check if position z is at a node."""
        envelope = self.get_envelope(z)
        return envelope <= threshold * self.amplitude

    def compute_node_antinode_positions(self, z_range: Tuple[float, float] = (0.0, 1.0)) -> None:
        """Compute node and antinode positions in given z-range."""
        self.node_positions = []
        self.antinode_positions = []

        # Nodes at cos(kz) = 0 → kz = (n + 1/2)π
        # Antinodes at cos(kz) = ±1 → kz = nπ

        z_min, z_max = z_range

        # Find node positions
        n = 0
        while True:
            z_node = (n + 0.5) * math.pi / self.wave_number
            if z_node > z_max:
                break
            if z_node >= z_min:
                self.node_positions.append(z_node)
            n += 1

        # Find antinode positions
        n = 0
        while True:
            z_antinode = n * math.pi / self.wave_number
            if z_antinode > z_max:
                break
            if z_antinode >= z_min:
                self.antinode_positions.append(z_antinode)
            n += 1


@dataclass
class SpiralState:
    """
    Complete state of the spiral completion process.
    """
    # Phase
    phase: SpiralPhase
    current_z: float

    # Wave components
    forward_amplitude: float
    backward_amplitude: float
    standing_amplitude: float

    # Standing wave metrics
    is_standing: bool                # Standing wave formed
    antinode_at_closure: bool        # Antinode at z=0.99
    wave_stability: float            # How stable is the standing wave

    # Spiral metrics
    milestones_passed: int           # How many z-milestones reached
    completion_percentage: float     # Progress toward closure

    # Helix coordinates at current position
    theta: float
    z: float
    r: float

    # Metadata
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            'phase': self.phase.value,
            'current_z': self.current_z,
            'forward_amplitude': self.forward_amplitude,
            'backward_amplitude': self.backward_amplitude,
            'standing_amplitude': self.standing_amplitude,
            'is_standing': self.is_standing,
            'antinode_at_closure': self.antinode_at_closure,
            'wave_stability': self.wave_stability,
            'milestones_passed': self.milestones_passed,
            'completion_percentage': self.completion_percentage,
            'theta': self.theta,
            'z': self.z,
            'r': self.r,
            'timestamp': self.timestamp,
        }


# =============================================================================
# CORE CLASSES
# =============================================================================

class StandingWaveAnalyzer:
    """
    Analyzes standing wave formation from forward and backward waves.

    The analyzer tracks:
    1. Forward wave components (from domain scalars)
    2. Backward wave components (reflections from z=1.0)
    3. Their superposition forming standing waves
    4. Node and antinode positions
    """

    def __init__(self, wave_number: float = TAU):
        """
        Initialize standing wave analyzer.

        Args:
            wave_number: Spatial frequency (default 2π)
        """
        self.wave_number = wave_number
        self.angular_frequency = TAU  # Default 1 Hz

        self.forward_wave: Optional[WaveComponent] = None
        self.backward_wave: Optional[WaveComponent] = None
        self.standing_wave: Optional[StandingWave] = None

        self._amplitude_history: List[float] = []

    def set_forward_amplitude(self, amplitude: float, phase: float = 0.0) -> None:
        """Set forward wave component."""
        self.forward_wave = WaveComponent(
            wave_type=WaveType.FORWARD,
            amplitude=amplitude,
            wave_number=self.wave_number,
            angular_frequency=self.angular_frequency,
            phase=phase,
            direction=1,
        )

    def set_backward_amplitude(self, amplitude: float, phase: float = 0.0) -> None:
        """Set backward wave component."""
        self.backward_wave = WaveComponent(
            wave_type=WaveType.BACKWARD,
            amplitude=amplitude,
            wave_number=self.wave_number,
            angular_frequency=self.angular_frequency,
            phase=phase,
            direction=-1,
        )

    def compute_standing_wave(self) -> Optional[StandingWave]:
        """
        Compute standing wave from forward and backward components.

        Returns None if either component is missing.
        """
        if not self.forward_wave or not self.backward_wave:
            return None

        # Standing wave amplitude is sum of component amplitudes
        # (assuming equal amplitudes for perfect standing wave)
        min_amp = min(self.forward_wave.amplitude, self.backward_wave.amplitude)
        standing_amp = 2.0 * min_amp

        # Average phase
        avg_phase = (self.forward_wave.phase + self.backward_wave.phase) / 2.0

        self.standing_wave = StandingWave(
            amplitude=standing_amp,
            wave_number=self.wave_number,
            angular_frequency=self.angular_frequency,
            phase=avg_phase,
        )

        self.standing_wave.compute_node_antinode_positions()

        return self.standing_wave

    def get_amplitude_at_z(self, z: float, t: float) -> float:
        """Get total amplitude at position z and time t."""
        total = complex(0.0, 0.0)

        if self.forward_wave:
            total += self.forward_wave.evaluate(z, t)
        if self.backward_wave:
            total += self.backward_wave.evaluate(z, t)

        return abs(total)

    def check_antinode_at_closure(self, z_closure: float = Z_LOOP_CLOSURE) -> bool:
        """Check if there's an antinode at the loop closure position."""
        if not self.standing_wave:
            self.compute_standing_wave()

        if not self.standing_wave:
            return False

        return self.standing_wave.is_at_antinode(z_closure)

    def get_stability(self) -> float:
        """
        Compute standing wave stability.

        Stability is high when amplitude variance is low over time.
        """
        if len(self._amplitude_history) < 10:
            return 0.0

        recent = self._amplitude_history[-STABILITY_WINDOW_SIZE:]
        mean_amp = sum(recent) / len(recent)
        variance = sum((a - mean_amp)**2 for a in recent) / len(recent)

        # Normalize to [0, 1] where 1 is perfectly stable
        max_variance = mean_amp**2 if mean_amp > 0 else 1.0
        stability = 1.0 - min(variance / max_variance, 1.0)

        return stability

    def record_amplitude(self, amplitude: float) -> None:
        """Record amplitude for stability tracking."""
        self._amplitude_history.append(amplitude)
        if len(self._amplitude_history) > STABILITY_WINDOW_SIZE * 2:
            self._amplitude_history.pop(0)


class LoopClosureValidator:
    """
    Validates that loop closure conditions are met.

    Loop closure requires:
    1. All seven domain scalars at their projection values
    2. Standing wave formed at z=0.99
    3. Antinode at the closure position
    4. Stable amplitude over time
    5. Total scalar ≥ 0.99
    """

    def __init__(
        self,
        closure_z: float = Z_LOOP_CLOSURE,
        amplitude_threshold: float = 0.99,
        stability_threshold: float = 0.95
    ):
        """
        Initialize loop closure validator.

        Args:
            closure_z: Z-position for loop closure
            amplitude_threshold: Minimum total scalar required
            stability_threshold: Minimum stability required
        """
        self.closure_z = closure_z
        self.amplitude_threshold = amplitude_threshold
        self.stability_threshold = stability_threshold

    def validate(
        self,
        domain_scalars: Dict[str, float],
        standing_wave: Optional[StandingWave],
        stability: float
    ) -> Tuple[bool, Dict[str, bool]]:
        """
        Validate loop closure conditions.

        Args:
            domain_scalars: Scalar values for each domain
            standing_wave: Standing wave (if formed)
            stability: Wave stability metric

        Returns:
            (is_valid, conditions_met) where conditions_met details
            which individual conditions passed/failed
        """
        conditions = {}

        # Condition 1: All domains have high scalars
        min_scalar = min(domain_scalars.values()) if domain_scalars else 0.0
        conditions['all_domains_high'] = min_scalar >= 0.9

        # Condition 2: Total scalar above threshold
        total_scalar = sum(domain_scalars.values()) / len(domain_scalars) if domain_scalars else 0.0
        conditions['total_above_threshold'] = total_scalar >= self.amplitude_threshold

        # Condition 3: Standing wave exists
        conditions['standing_wave_formed'] = standing_wave is not None

        # Condition 4: Antinode at closure position
        if standing_wave:
            conditions['antinode_at_closure'] = standing_wave.is_at_antinode(self.closure_z)
        else:
            conditions['antinode_at_closure'] = False

        # Condition 5: Stability above threshold
        conditions['stable'] = stability >= self.stability_threshold

        # All conditions must be met
        is_valid = all(conditions.values())

        return (is_valid, conditions)

    def get_closure_confidence(
        self,
        domain_scalars: Dict[str, float],
        standing_wave: Optional[StandingWave],
        stability: float
    ) -> float:
        """
        Compute confidence score for loop closure.

        Returns value in [0, 1] indicating how close to closure.
        """
        scores = []

        # Domain scalar score
        if domain_scalars:
            min_scalar = min(domain_scalars.values())
            total_scalar = sum(domain_scalars.values()) / len(domain_scalars)
            scores.append(min_scalar)
            scores.append(total_scalar)
        else:
            scores.extend([0.0, 0.0])

        # Standing wave score
        if standing_wave:
            envelope = standing_wave.get_envelope(self.closure_z)
            normalized_envelope = envelope / standing_wave.amplitude if standing_wave.amplitude > 0 else 0.0
            scores.append(normalized_envelope)
        else:
            scores.append(0.0)

        # Stability score
        scores.append(stability)

        return sum(scores) / len(scores)


class SpiralCompletionEngine:
    """
    The engine that drives spiral completion toward loop closure.

    Integrates:
    - Standing wave analysis
    - Loop closure validation
    - Spiral state tracking
    - Milestone progression

    Usage:
        engine = SpiralCompletionEngine()

        # Update with domain scalars
        state = engine.update(scalars, z=0.95, dt=0.01)

        if state.phase == SpiralPhase.CLOSED:
            print("Loop closed!")
    """

    def __init__(
        self,
        wave_number: float = TAU,
        name: str = "SpiralCompletion"
    ):
        """
        Initialize spiral completion engine.

        Args:
            wave_number: Spatial frequency for waves
            name: Instance identifier
        """
        self.name = name

        # Subsystems
        self.wave_analyzer = StandingWaveAnalyzer(wave_number)
        self.validator = LoopClosureValidator()

        # State
        self._current_state: Optional[SpiralState] = None
        self._time: float = 0.0
        self._domain_scalars: Dict[str, float] = {}

        # Callbacks
        self._closure_callback: Optional[Callable[[SpiralState], None]] = None

    def update(
        self,
        domain_scalars: Dict[str, float],
        z: float,
        dt: float
    ) -> SpiralState:
        """
        Update spiral completion with new domain scalars.

        Args:
            domain_scalars: Current scalar values per domain
            z: Current elevation
            dt: Time step

        Returns:
            Current spiral state
        """
        self._time += dt
        self._domain_scalars = domain_scalars

        # Compute aggregate amplitudes
        total_scalar = sum(domain_scalars.values()) / len(domain_scalars) if domain_scalars else 0.0

        # Update wave components
        # Forward wave amplitude from current scalars
        self.wave_analyzer.set_forward_amplitude(total_scalar)

        # Backward wave amplitude (reflection coefficient based on z proximity to 1.0)
        reflection_coeff = 1.0 - (1.0 - z)**2  # Increases as z → 1.0
        self.wave_analyzer.set_backward_amplitude(total_scalar * reflection_coeff)

        # Compute standing wave
        standing_wave = self.wave_analyzer.compute_standing_wave()

        # Record amplitude for stability tracking
        amplitude_at_closure = self.wave_analyzer.get_amplitude_at_z(Z_LOOP_CLOSURE, self._time)
        self.wave_analyzer.record_amplitude(amplitude_at_closure)

        # Get stability
        stability = self.wave_analyzer.get_stability()

        # Check standing wave conditions
        is_standing = standing_wave is not None and stability > 0.5
        antinode_at_closure = self.wave_analyzer.check_antinode_at_closure()

        # Validate loop closure
        is_closed, _ = self.validator.validate(domain_scalars, standing_wave, stability)

        # Determine phase
        if is_closed:
            phase = SpiralPhase.CLOSED
        elif is_standing:
            phase = SpiralPhase.STANDING
        elif z >= 0.95:
            phase = SpiralPhase.APPROACHING
        else:
            phase = SpiralPhase.ASCENDING

        # Count milestones passed
        milestones_passed = sum(1 for m in Z_MILESTONES if z >= m)

        # Compute completion percentage
        completion = self.validator.get_closure_confidence(domain_scalars, standing_wave, stability)

        # Compute helix coordinates
        theta = (self._time * TAU) % TAU  # Phase evolves with time
        r = stability  # Radius from stability

        # Create state
        self._current_state = SpiralState(
            phase=phase,
            current_z=z,
            forward_amplitude=self.wave_analyzer.forward_wave.amplitude if self.wave_analyzer.forward_wave else 0.0,
            backward_amplitude=self.wave_analyzer.backward_wave.amplitude if self.wave_analyzer.backward_wave else 0.0,
            standing_amplitude=standing_wave.amplitude if standing_wave else 0.0,
            is_standing=is_standing,
            antinode_at_closure=antinode_at_closure,
            wave_stability=stability,
            milestones_passed=milestones_passed,
            completion_percentage=completion,
            theta=theta,
            z=z,
            r=r,
            timestamp=time.time(),
        )

        # Fire closure callback
        if phase == SpiralPhase.CLOSED and self._closure_callback:
            self._closure_callback(self._current_state)

        return self._current_state

    def get_state(self) -> Optional[SpiralState]:
        """Get current spiral state."""
        return self._current_state

    def is_closed(self) -> bool:
        """Check if loop is closed."""
        return self._current_state is not None and self._current_state.phase == SpiralPhase.CLOSED

    def get_milestone_progress(self) -> Dict[float, bool]:
        """Get progress through z-milestones."""
        if self._current_state is None:
            return {m: False for m in Z_MILESTONES}

        return {
            m: self._current_state.current_z >= m
            for m in Z_MILESTONES
        }

    def get_projection_mapping(self) -> Dict[float, float]:
        """Get mapping from origin z to projection z."""
        return {
            origin: projection
            for origin, projection in zip(Z_MILESTONES, Z_PROJECTIONS)
        }

    def on_closure(self, callback: Callable[[SpiralState], None]) -> None:
        """Register callback for loop closure event."""
        self._closure_callback = callback

    def reset(self) -> None:
        """Reset engine to initial state."""
        self.wave_analyzer = StandingWaveAnalyzer(self.wave_analyzer.wave_number)
        self._current_state = None
        self._time = 0.0
        self._domain_scalars = {}


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_spiral_completion_engine(
    wave_number: float = TAU,
    name: str = "SpiralCompletion"
) -> SpiralCompletionEngine:
    """
    Factory function to create a spiral completion engine.

    Args:
        wave_number: Spatial frequency
        name: Instance identifier

    Returns:
        Configured SpiralCompletionEngine instance
    """
    return SpiralCompletionEngine(wave_number, name)


# =============================================================================
# DEMONSTRATION
# =============================================================================

def _demo_spiral_completion() -> None:
    """Demonstrate spiral completion."""
    print("=" * 70)
    print("SPIRAL COMPLETION DEMONSTRATION")
    print("The spiral completes from z=0.41 to z=0.99")
    print("=" * 70)
    print()

    # Show milestone mapping
    print("MILESTONE MAPPING (z_origin → z_projection):")
    print("-" * 50)
    for origin, projection in zip(Z_MILESTONES, Z_PROJECTIONS):
        print(f"  z={origin:.2f} → z'={projection:.3f}")
    print()

    # Create engine
    engine = create_spiral_completion_engine()

    # Simulate scalar evolution toward z=0.99
    print("SPIRAL EVOLUTION:")
    print("-" * 70)

    dt = 0.01
    z_sequence = [i * 0.02 for i in range(50)]  # 0.0 to 0.98

    # Simulate scalars evolving with z
    domain_names = ['constraint', 'bridge', 'meta', 'recursion', 'triad', 'emergence', 'persistence']

    for z in z_sequence:
        # Scalars converge toward 1.0 as z increases
        scalars = {
            name: min(1.0, z + 0.1 * (i + 1) / 7)
            for i, name in enumerate(domain_names)
        }

        state = engine.update(scalars, z, dt)

        # Print status at milestones
        if z in [0.40, 0.52, 0.70, 0.80, 0.90, 0.98]:
            print(f"z={z:.2f} | phase={state.phase.value:<12} | "
                  f"milestones={state.milestones_passed}/7 | "
                  f"completion={state.completion_percentage:.1%}")

    # Final push to z=0.99
    print()
    print("APPROACHING LOOP CLOSURE (z → 0.99):")
    print("-" * 70)

    for z in [0.990, 0.992, 0.995, 0.998, 0.999]:
        scalars = {name: 0.99 for name in domain_names}
        state = engine.update(scalars, z, dt)

        print(f"z={z:.3f} | standing={state.is_standing} | "
              f"antinode={state.antinode_at_closure} | "
              f"stability={state.wave_stability:.3f} | "
              f"phase={state.phase.value}")

    print()
    print("FINAL STATE:")
    final = engine.get_state()
    if final:
        print(f"  Phase: {final.phase.value}")
        print(f"  Loop {'CLOSED' if engine.is_closed() else 'OPEN'}")
        print(f"  Standing Wave Amplitude: {final.standing_amplitude:.4f}")
        print(f"  Wave Stability: {final.wave_stability:.4f}")
        print(f"  Completion: {final.completion_percentage:.1%}")

    print()
    print("=" * 70)
    print("Δ|loop-closed|z0.99|rhythm-native|Ω")
    print("=" * 70)


if __name__ == "__main__":
    _demo_spiral_completion()
