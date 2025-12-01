"""
UNIFIED DOMAIN BRIDGE
=====================
Cross-domain integration through wave function unification.

The bridge connects all seven domains into a unified field where
their asymptotic scalars can constructively interfere to achieve
loop closure.

Core Concept:
-------------
Each domain carries a wave function Ψ_i that propagates along the
z-axis. When domains are bridged, their wave functions can:
1. Reinforce each other (constructive interference)
2. Cancel each other (destructive interference)
3. Create standing waves (loop closure)

The unified field emerges when all domains achieve coherent
superposition at z=0.99.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Callable
from enum import Enum


# =============================================================================
# CONSTANTS
# =============================================================================

TAU = 2.0 * math.pi
PHI = (1.0 + math.sqrt(5.0)) / 2.0

# Domain identifiers and their intrinsic frequencies (Hz)
E = math.e  # Euler's number
DOMAIN_FREQUENCIES = {
    'constraint': 1.0,       # Base frequency
    'bridge': PHI,           # Golden ratio frequency
    'meta': 2.0,             # Octave
    'recursion': PHI * PHI,  # PHI^2
    'triad': 3.0,            # Perfect fifth
    'emergence': E,          # Natural frequency
    'persistence': TAU,      # Full cycle frequency
}

# Bridge coupling strengths (how strongly domains connect)
DEFAULT_BRIDGE_STRENGTH = 0.5


# =============================================================================
# ENUMERATIONS
# =============================================================================

class BridgeState(Enum):
    """State of a domain bridge."""
    DISCONNECTED = "disconnected"  # No coupling
    WEAK = "weak"                  # Coupling < 0.3
    MODERATE = "moderate"          # 0.3 ≤ coupling < 0.7
    STRONG = "strong"              # 0.7 ≤ coupling < 0.9
    UNIFIED = "unified"            # coupling ≥ 0.9


class InterferenceType(Enum):
    """Type of interference between domains."""
    CONSTRUCTIVE = "constructive"
    DESTRUCTIVE = "destructive"
    MIXED = "mixed"


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class DomainWaveFunction:
    """
    Wave function representation for a single domain.

    Ψ_i(z, t) = A_i · exp(j·(k_i·z - ω_i·t + φ_i))

    Where:
    - A_i: amplitude (scalar value)
    - k_i: wave number (determines spatial frequency)
    - ω_i: angular frequency
    - φ_i: initial phase
    """
    domain_name: str
    amplitude: float = 0.0          # A_i: scalar value
    wave_number: float = 1.0        # k_i: spatial frequency
    angular_frequency: float = 1.0  # ω_i: temporal frequency
    phase: float = 0.0              # φ_i: initial phase

    def evaluate(self, z: float, t: float) -> complex:
        """
        Evaluate wave function at position z and time t.

        Returns complex value: A · exp(j·(k·z - ω·t + φ))
        """
        argument = self.wave_number * z - self.angular_frequency * t + self.phase
        return self.amplitude * complex(math.cos(argument), math.sin(argument))

    def get_real(self, z: float, t: float) -> float:
        """Get real part of wave function."""
        return self.evaluate(z, t).real

    def get_imag(self, z: float, t: float) -> float:
        """Get imaginary part of wave function."""
        return self.evaluate(z, t).imag

    def get_magnitude(self, z: float, t: float) -> float:
        """Get magnitude of wave function."""
        return abs(self.evaluate(z, t))


@dataclass
class BridgeConnection:
    """
    Connection between two domains.

    The bridge mediates energy transfer and phase synchronization
    between domains.
    """
    domain_a: str
    domain_b: str
    strength: float = DEFAULT_BRIDGE_STRENGTH  # [0, 1]
    phase_offset: float = 0.0                  # Phase difference at bridge
    state: BridgeState = BridgeState.DISCONNECTED
    energy_flow: float = 0.0                   # Net energy transfer A→B

    def compute_coupling(
        self,
        wave_a: DomainWaveFunction,
        wave_b: DomainWaveFunction,
        z: float,
        t: float
    ) -> float:
        """
        Compute effective coupling between two domains at given position.

        Coupling = strength · |Ψ_a| · |Ψ_b| · cos(Δφ)
        """
        psi_a = wave_a.evaluate(z, t)
        psi_b = wave_b.evaluate(z, t)

        # Phase difference
        phase_a = math.atan2(psi_a.imag, psi_a.real)
        phase_b = math.atan2(psi_b.imag, psi_b.real)
        delta_phase = phase_a - phase_b + self.phase_offset

        coupling = self.strength * abs(psi_a) * abs(psi_b) * math.cos(delta_phase)

        # Update state based on coupling strength
        if abs(coupling) < 0.01:
            self.state = BridgeState.DISCONNECTED
        elif abs(coupling) < 0.3:
            self.state = BridgeState.WEAK
        elif abs(coupling) < 0.7:
            self.state = BridgeState.MODERATE
        elif abs(coupling) < 0.9:
            self.state = BridgeState.STRONG
        else:
            self.state = BridgeState.UNIFIED

        # Energy flow is proportional to coupling and phase gradient
        self.energy_flow = coupling * (wave_a.amplitude - wave_b.amplitude)

        return coupling


@dataclass
class UnifiedFieldState:
    """
    State of the unified domain field.
    """
    # Wave functions per domain
    wave_functions: Dict[str, DomainWaveFunction]

    # Superposition
    total_amplitude: float           # |Ψ_total|
    total_phase: float              # arg(Ψ_total)

    # Field metrics
    coherence: float                # How aligned are all domains
    energy_density: float           # Total energy in field
    interference_type: InterferenceType

    # Bridge summary
    active_bridges: int             # Number of connected bridges
    mean_coupling: float            # Average coupling strength

    # Position
    z: float
    t: float

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            'wave_functions': {
                name: {
                    'amplitude': wf.amplitude,
                    'phase': wf.phase,
                }
                for name, wf in self.wave_functions.items()
            },
            'total_amplitude': self.total_amplitude,
            'total_phase': self.total_phase,
            'coherence': self.coherence,
            'energy_density': self.energy_density,
            'interference_type': self.interference_type.value,
            'active_bridges': self.active_bridges,
            'mean_coupling': self.mean_coupling,
            'z': self.z,
            't': self.t,
        }


# =============================================================================
# CORE CLASSES
# =============================================================================

class DomainBridge:
    """
    Bridge connecting multiple domains for wave function integration.

    The bridge manages:
    1. Wave function representation for each domain
    2. Pairwise coupling between domains
    3. Phase synchronization dynamics
    4. Energy transfer between domains
    """

    def __init__(
        self,
        domains: Optional[List[str]] = None,
        bridge_strength: float = DEFAULT_BRIDGE_STRENGTH
    ):
        """
        Initialize domain bridge.

        Args:
            domains: List of domain names (defaults to all 7)
            bridge_strength: Default coupling strength
        """
        self.domains = domains or list(DOMAIN_FREQUENCIES.keys())
        self.bridge_strength = bridge_strength

        # Initialize wave functions
        self.wave_functions: Dict[str, DomainWaveFunction] = {}
        for domain in self.domains:
            freq = DOMAIN_FREQUENCIES.get(domain, 1.0)
            self.wave_functions[domain] = DomainWaveFunction(
                domain_name=domain,
                amplitude=0.0,
                wave_number=freq,
                angular_frequency=TAU * freq,
                phase=0.0,
            )

        # Initialize bridge connections (all pairs)
        self.bridges: List[BridgeConnection] = []
        for i, d1 in enumerate(self.domains):
            for d2 in self.domains[i+1:]:
                self.bridges.append(BridgeConnection(
                    domain_a=d1,
                    domain_b=d2,
                    strength=bridge_strength,
                ))

    def set_amplitude(self, domain: str, amplitude: float) -> None:
        """Set amplitude for a domain's wave function."""
        if domain in self.wave_functions:
            self.wave_functions[domain].amplitude = amplitude

    def set_phase(self, domain: str, phase: float) -> None:
        """Set phase for a domain's wave function."""
        if domain in self.wave_functions:
            self.wave_functions[domain].phase = phase % TAU

    def update_from_scalars(self, scalars: Dict[str, float]) -> None:
        """
        Update wave function amplitudes from scalar values.

        Args:
            scalars: Dictionary mapping domain names to scalar values
        """
        for domain, scalar in scalars.items():
            if domain in self.wave_functions:
                self.wave_functions[domain].amplitude = scalar

    def compute_superposition(self, z: float, t: float) -> complex:
        """
        Compute total wave function superposition.

        Ψ_total = Σ Ψ_i(z, t)
        """
        total = complex(0.0, 0.0)
        for wf in self.wave_functions.values():
            total += wf.evaluate(z, t)
        return total

    def compute_coupling_matrix(self, z: float, t: float) -> Dict[Tuple[str, str], float]:
        """
        Compute coupling between all domain pairs.

        Returns dictionary mapping (domain_a, domain_b) to coupling strength.
        """
        couplings = {}
        for bridge in self.bridges:
            wave_a = self.wave_functions[bridge.domain_a]
            wave_b = self.wave_functions[bridge.domain_b]
            coupling = bridge.compute_coupling(wave_a, wave_b, z, t)
            couplings[(bridge.domain_a, bridge.domain_b)] = coupling
        return couplings

    def get_coherence(self, z: float, t: float) -> float:
        """
        Compute field coherence (how aligned are all domains).

        Coherence = |Ψ_total|² / Σ|Ψ_i|²
        """
        total = self.compute_superposition(z, t)
        sum_of_squares = sum(
            abs(wf.evaluate(z, t))**2
            for wf in self.wave_functions.values()
        )

        if sum_of_squares < 1e-10:
            return 0.0

        return abs(total)**2 / (len(self.wave_functions) * sum_of_squares)

    def get_field_state(self, z: float, t: float) -> UnifiedFieldState:
        """
        Get complete unified field state at position (z, t).
        """
        # Compute superposition
        total = self.compute_superposition(z, t)
        total_amplitude = abs(total)
        total_phase = math.atan2(total.imag, total.real) % TAU

        # Compute coherence
        coherence = self.get_coherence(z, t)

        # Compute energy density (sum of squared amplitudes)
        energy_density = sum(
            abs(wf.evaluate(z, t))**2
            for wf in self.wave_functions.values()
        )

        # Count active bridges and compute mean coupling
        couplings = self.compute_coupling_matrix(z, t)
        active_bridges = sum(1 for c in couplings.values() if abs(c) > 0.1)
        mean_coupling = sum(abs(c) for c in couplings.values()) / len(couplings) if couplings else 0.0

        # Determine interference type
        max_possible = sum(wf.amplitude for wf in self.wave_functions.values())
        if total_amplitude > 0.8 * max_possible:
            interference_type = InterferenceType.CONSTRUCTIVE
        elif total_amplitude < 0.2 * max_possible:
            interference_type = InterferenceType.DESTRUCTIVE
        else:
            interference_type = InterferenceType.MIXED

        return UnifiedFieldState(
            wave_functions=self.wave_functions.copy(),
            total_amplitude=total_amplitude,
            total_phase=total_phase,
            coherence=coherence,
            energy_density=energy_density,
            interference_type=interference_type,
            active_bridges=active_bridges,
            mean_coupling=mean_coupling,
            z=z,
            t=t,
        )


class UnifiedDomainField:
    """
    The unified field that emerges from domain bridge integration.

    This represents the complete state of all domains as they
    evolve toward loop closure. The field tracks:

    1. Individual domain wave functions
    2. Cross-domain coupling dynamics
    3. Standing wave formation
    4. Loop closure conditions
    """

    def __init__(self, bridge: Optional[DomainBridge] = None):
        """
        Initialize unified domain field.

        Args:
            bridge: Optional pre-configured domain bridge
        """
        self.bridge = bridge or DomainBridge()
        self._time: float = 0.0
        self._z: float = 0.0
        self._state_history: List[UnifiedFieldState] = []

        # Callbacks
        self._coherence_callback: Optional[Callable[[float], None]] = None

    def advance(self, z: float, dt: float) -> UnifiedFieldState:
        """
        Advance field by one time step.

        Args:
            z: Current elevation
            dt: Time step

        Returns:
            Current field state
        """
        self._time += dt
        self._z = z

        # Get current state
        state = self.bridge.get_field_state(z, self._time)

        # Track history (keep last 100 states)
        self._state_history.append(state)
        if len(self._state_history) > 100:
            self._state_history.pop(0)

        # Fire coherence callback if set
        if self._coherence_callback and state.coherence > 0.9:
            self._coherence_callback(state.coherence)

        return state

    def update_scalars(self, scalars: Dict[str, float]) -> None:
        """Update domain scalars from external source."""
        self.bridge.update_from_scalars(scalars)

    def get_standing_wave_amplitude(self) -> float:
        """
        Compute standing wave amplitude.

        A standing wave forms when domains interfere constructively
        at fixed positions (nodes and antinodes).
        """
        if len(self._state_history) < 10:
            return 0.0

        # Get amplitude variation over recent history
        amplitudes = [s.total_amplitude for s in self._state_history[-10:]]
        mean_amp = sum(amplitudes) / len(amplitudes)
        variance = sum((a - mean_amp)**2 for a in amplitudes) / len(amplitudes)

        # Low variance indicates standing wave
        if variance < 0.01 * mean_amp**2:
            return mean_amp
        return 0.0

    def check_loop_closure_condition(self) -> Tuple[bool, float]:
        """
        Check if loop closure condition is met.

        Loop closure requires:
        1. High coherence (> 0.99)
        2. Standing wave formation
        3. All domains active (amplitude > 0.9)

        Returns:
            (is_closed, closure_confidence)
        """
        if not self._state_history:
            return (False, 0.0)

        state = self._state_history[-1]

        # Check coherence
        coherence_ok = state.coherence >= 0.99

        # Check standing wave
        standing_amplitude = self.get_standing_wave_amplitude()
        standing_ok = standing_amplitude > 0.5

        # Check all domains active
        all_active = all(
            wf.amplitude > 0.9
            for wf in state.wave_functions.values()
        )

        is_closed = coherence_ok and standing_ok and all_active

        # Compute confidence
        confidence = (
            (state.coherence / 0.99) * 0.4 +
            (standing_amplitude / 1.0) * 0.3 +
            (sum(wf.amplitude for wf in state.wave_functions.values()) /
             len(state.wave_functions)) * 0.3
        )
        confidence = min(confidence, 1.0)

        return (is_closed, confidence)

    def on_high_coherence(self, callback: Callable[[float], None]) -> None:
        """Register callback for high coherence events."""
        self._coherence_callback = callback

    @property
    def current_state(self) -> Optional[UnifiedFieldState]:
        """Get most recent state."""
        return self._state_history[-1] if self._state_history else None

    @property
    def time(self) -> float:
        """Get current time."""
        return self._time

    @property
    def z(self) -> float:
        """Get current elevation."""
        return self._z


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_domain_bridge(
    domains: Optional[List[str]] = None,
    bridge_strength: float = DEFAULT_BRIDGE_STRENGTH
) -> DomainBridge:
    """
    Factory function to create a domain bridge.

    Args:
        domains: List of domain names (defaults to all 7)
        bridge_strength: Default coupling strength

    Returns:
        Configured DomainBridge instance
    """
    return DomainBridge(domains, bridge_strength)


def create_unified_field(
    bridge: Optional[DomainBridge] = None
) -> UnifiedDomainField:
    """
    Factory function to create a unified domain field.

    Args:
        bridge: Optional pre-configured domain bridge

    Returns:
        Configured UnifiedDomainField instance
    """
    return UnifiedDomainField(bridge)


# =============================================================================
# DEMONSTRATION
# =============================================================================

def _demo_domain_unifier() -> None:
    """Demonstrate domain unification."""
    print("=" * 60)
    print("DOMAIN UNIFIER DEMONSTRATION")
    print("=" * 60)
    print()

    # Create bridge and field
    bridge = create_domain_bridge()
    field = create_unified_field(bridge)

    # Set initial amplitudes (simulating scalar values at z=0.99)
    scalars = {
        'constraint': 0.95,
        'bridge': 0.96,
        'meta': 0.97,
        'recursion': 0.97,
        'triad': 0.98,
        'emergence': 0.985,
        'persistence': 0.987,
    }
    field.update_scalars(scalars)

    # Evolve field
    print("FIELD EVOLUTION:")
    print("-" * 60)
    z = 0.99
    dt = 0.01

    for i in range(20):
        state = field.advance(z, dt)

        if i % 4 == 0:
            print(f"t={field.time:5.2f} | "
                  f"|Ψ|={state.total_amplitude:.4f} | "
                  f"coherence={state.coherence:.4f} | "
                  f"interference={state.interference_type.value}")

    print()

    # Check loop closure
    is_closed, confidence = field.check_loop_closure_condition()
    print(f"Loop Closure: {'CLOSED' if is_closed else 'OPEN'}")
    print(f"Closure Confidence: {confidence:.4f}")
    print()

    # Show final state
    final_state = field.current_state
    if final_state:
        print("FINAL FIELD STATE:")
        for name, wf in final_state.wave_functions.items():
            print(f"  {name:<12}: amplitude={wf.amplitude:.4f}, phase={wf.phase:.4f}")

    print()
    print("=" * 60)


if __name__ == "__main__":
    _demo_domain_unifier()
