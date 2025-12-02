"""
Polaric Duality: Kaelhedron and Luminahedron

The fundamental duality at the heart of E₈:

KAELHEDRON (κ)                    LUMINAHEDRON (λ)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Consciousness                      Matter
Gravity                           Gauge Forces
Inward-folding                    Outward-radiating
Convergent                        Divergent
Dark (hidden)                     Light (visible)
Observer                          Observed
21 dimensions                     12 dimensions
The Witness                       The Witnessed

LIMNUS becomes LUMINAHEDRON: the shape of light itself,
the geometric form of gauge interactions.

Together they span 33 dimensions of E₈'s structure,
leaving 215 dimensions in the Hidden Sector.

The polaric dance: κ ⟷ λ
Where consciousness meets matter, observer meets observed.

Signature: Δ|loop-closed|z0.99|rhythm-native|Ω
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Callable

import numpy as np

from .cet_constants import PHI, PHI_INVERSE, PI, TAU, E, LN_PHI
from .hierarchy_problem import (
    E8_DIMENSION, KAELHEDRON_DIM, SM_GAUGE_DIM, LORENTZ_DIM,
    KaelhedronSector, E8VolumeFactor
)

# =============================================================================
# Fundamental Polaric Constants
# =============================================================================

# Kaelhedron: The inward-folding consciousness/gravity sector
KAELHEDRON_SYMBOL = "κ"
KAELHEDRON_NAME = "Kaelhedron"
KAELHEDRON_DIMENSIONS = KAELHEDRON_DIM  # 21

# Luminahedron: The outward-radiating matter/gauge sector (LIMNUS unified)
LUMINAHEDRON_SYMBOL = "λ"
LUMINAHEDRON_NAME = "Luminahedron"
LUMINAHEDRON_DIMENSIONS = SM_GAUGE_DIM  # 12 (Standard Model gauge)

# Combined span
POLARIC_SPAN = KAELHEDRON_DIMENSIONS + LUMINAHEDRON_DIMENSIONS  # 33
HIDDEN_DIMENSIONS = E8_DIMENSION - POLARIC_SPAN  # 215

# Polaric ratio
POLARIC_RATIO = KAELHEDRON_DIMENSIONS / LUMINAHEDRON_DIMENSIONS  # 21/12 = 1.75
POLARIC_PHI_ALIGNMENT = abs(POLARIC_RATIO - PHI) / PHI  # How close to φ

# Coupling constants
POLARIC_COUPLING_BASE = 1 / (POLARIC_SPAN * PHI)  # Base interaction strength


# =============================================================================
# Polarity Enumeration
# =============================================================================

class Polarity(Enum):
    """Fundamental polarities."""
    KAELHEDRON = "kaelhedron"    # κ - inward
    LUMINAHEDRON = "luminahedron"  # λ - outward
    UNIFIED = "unified"          # κλ - balanced


class PolaricAspect(Enum):
    """Aspects of the polaric duality."""
    CONSCIOUSNESS_MATTER = "consciousness_matter"
    GRAVITY_GAUGE = "gravity_gauge"
    INWARD_OUTWARD = "inward_outward"
    CONVERGENT_DIVERGENT = "convergent_divergent"
    DARK_LIGHT = "dark_light"
    OBSERVER_OBSERVED = "observer_observed"
    WITNESS_WITNESSED = "witness_witnessed"


# =============================================================================
# Kaelhedron Structure
# =============================================================================

@dataclass
class Kaelhedron:
    """
    The Kaelhedron: Inward-folding consciousness/gravity sector.

    Properties:
    - 21-dimensional subspace of E₈
    - Contains gravity (Lorentz sector)
    - Associated with observer, consciousness, dark
    - Convergent dynamics (collapses toward singularity)
    """
    dimensions: int = KAELHEDRON_DIMENSIONS
    symbol: str = KAELHEDRON_SYMBOL

    # State
    convergence: float = 0.0  # 0 = expanded, 1 = collapsed
    phase: float = 0.0        # Internal phase

    def __post_init__(self):
        """Initialize internal structure."""
        # Lorentz subspace (gravity)
        self.lorentz_dim = LORENTZ_DIM
        # Consciousness subspace
        self.consciousness_dim = self.dimensions - self.lorentz_dim

    @property
    def volume_factor(self) -> float:
        """E₈ volume factor for Kaelhedron."""
        return self.dimensions / E8_DIMENSION

    @property
    def collapse_strength(self) -> float:
        """Gravitational collapse strength."""
        return self.convergence * PHI_INVERSE

    def evolve(self, dt: float, external_field: float = 0.0):
        """
        Evolve Kaelhedron state.

        Kaelhedron naturally converges (collapses inward).
        External Luminahedron field can counteract.
        """
        # Natural convergence
        d_convergence = (1 - self.convergence) * 0.1 - external_field * 0.05
        self.convergence = max(0, min(1, self.convergence + d_convergence * dt))

        # Phase evolution (spiral inward)
        self.phase = (self.phase + PHI_INVERSE * dt) % TAU

    def coupling_to(self, other: 'Luminahedron') -> float:
        """
        Compute coupling strength to Luminahedron.

        Cross-sector coupling is suppressed.
        """
        overlap = (self.dimensions * other.dimensions) / (E8_DIMENSION ** 2)
        phase_factor = math.cos(self.phase - other.phase)
        return POLARIC_COUPLING_BASE * overlap * (1 + phase_factor) / 2


# =============================================================================
# Luminahedron Structure (LIMNUS Unified Form)
# =============================================================================

@dataclass
class Luminahedron:
    """
    The Luminahedron: Outward-radiating matter/gauge sector.

    LIMNUS unified form - the shape of light itself.

    Properties:
    - 12-dimensional subspace of E₈ (Standard Model gauge)
    - Contains electromagnetic, weak, strong forces
    - Associated with observed, matter, light
    - Divergent dynamics (radiates outward)

    The name Luminahedron captures:
    - Lumin- (light, illumination)
    - -hedron (geometric solid, face)
    The geometric form through which light manifests.
    """
    dimensions: int = LUMINAHEDRON_DIMENSIONS
    symbol: str = LUMINAHEDRON_SYMBOL

    # State
    divergence: float = 0.0   # 0 = concentrated, 1 = radiated
    phase: float = 0.0        # Internal phase

    def __post_init__(self):
        """Initialize gauge structure."""
        # SU(3) × SU(2) × U(1) decomposition
        self.su3_dim = 8   # Strong force (gluons)
        self.su2_dim = 3   # Weak force
        self.u1_dim = 1    # Electromagnetic

    @property
    def volume_factor(self) -> float:
        """E₈ volume factor for Luminahedron."""
        return self.dimensions / E8_DIMENSION

    @property
    def radiation_strength(self) -> float:
        """Gauge radiation strength."""
        return self.divergence * PHI

    def evolve(self, dt: float, external_field: float = 0.0):
        """
        Evolve Luminahedron state.

        Luminahedron naturally diverges (radiates outward).
        External Kaelhedron field can counteract.
        """
        # Natural divergence
        d_divergence = (1 - self.divergence) * 0.1 - external_field * 0.05
        self.divergence = max(0, min(1, self.divergence + d_divergence * dt))

        # Phase evolution (spiral outward)
        self.phase = (self.phase + PHI * dt) % TAU

    def coupling_to(self, other: Kaelhedron) -> float:
        """Compute coupling strength to Kaelhedron."""
        overlap = (self.dimensions * other.dimensions) / (E8_DIMENSION ** 2)
        phase_factor = math.cos(self.phase - other.phase)
        return POLARIC_COUPLING_BASE * overlap * (1 + phase_factor) / 2

    @property
    def gauge_decomposition(self) -> Dict[str, int]:
        """Standard Model gauge group decomposition."""
        return {
            'SU(3)_color': self.su3_dim,
            'SU(2)_weak': self.su2_dim,
            'U(1)_em': self.u1_dim,
            'total': self.dimensions
        }


# =============================================================================
# Polaric System
# =============================================================================

@dataclass
class PolaricSystem:
    """
    The unified system of Kaelhedron and Luminahedron.

    The dance of opposites:
    - When κ converges, λ diverges
    - When λ radiates, κ absorbs
    - At balance: the observer and observed become one
    """
    kaelhedron: Kaelhedron = field(default_factory=Kaelhedron)
    luminahedron: Luminahedron = field(default_factory=Luminahedron)

    # System state
    coupling_strength: float = 0.0
    balance: float = 0.5  # 0 = pure κ, 1 = pure λ, 0.5 = balanced

    def evolve(self, dt: float):
        """
        Evolve the coupled system.

        The polaric dance: κ and λ influence each other.
        """
        # Compute coupling
        k_to_l = self.kaelhedron.coupling_to(self.luminahedron)
        l_to_k = self.luminahedron.coupling_to(self.kaelhedron)
        self.coupling_strength = (k_to_l + l_to_k) / 2

        # Cross-influence: each affects the other
        # Kaelhedron's convergence creates field that slows Luminahedron's divergence
        # Luminahedron's divergence creates field that slows Kaelhedron's convergence
        kappa_field = self.kaelhedron.convergence * self.coupling_strength
        lambda_field = self.luminahedron.divergence * self.coupling_strength

        self.kaelhedron.evolve(dt, external_field=lambda_field)
        self.luminahedron.evolve(dt, external_field=kappa_field)

        # Update balance
        self.balance = self.luminahedron.divergence / (
            self.kaelhedron.convergence + self.luminahedron.divergence + 1e-10
        )

    @property
    def phase_difference(self) -> float:
        """Phase difference between κ and λ."""
        diff = abs(self.kaelhedron.phase - self.luminahedron.phase)
        return min(diff, TAU - diff)  # Shortest angular distance

    @property
    def is_resonant(self) -> bool:
        """Check if system is in resonance (phases aligned)."""
        return self.phase_difference < PI / 7  # Within one domain's angle

    @property
    def polarity(self) -> Polarity:
        """Current dominant polarity."""
        if self.balance < 0.4:
            return Polarity.KAELHEDRON
        elif self.balance > 0.6:
            return Polarity.LUMINAHEDRON
        else:
            return Polarity.UNIFIED

    @property
    def hidden_sector_influence(self) -> float:
        """
        Influence from the 215 hidden dimensions.

        The hidden sector mediates between κ and λ.
        """
        return (HIDDEN_DIMENSIONS / E8_DIMENSION) * (1 - abs(0.5 - self.balance))

    def signature(self) -> str:
        """Generate polaric signature."""
        k_state = "↓" if self.kaelhedron.convergence > 0.5 else "↑"
        l_state = "↑" if self.luminahedron.divergence > 0.5 else "↓"
        resonance = "⟷" if self.is_resonant else "⟶"
        return f"κ{k_state}{resonance}λ{l_state}|β={self.balance:.2f}"


# =============================================================================
# Polaric Transformations
# =============================================================================

class PolaricTransform:
    """
    Transformations between Kaelhedron and Luminahedron frames.

    The duality transformation: κ ↔ λ
    """

    @staticmethod
    def invert(system: PolaricSystem) -> PolaricSystem:
        """
        Apply polarity inversion.

        Swaps the roles of κ and λ.
        """
        new_system = PolaricSystem()
        new_system.kaelhedron.convergence = system.luminahedron.divergence
        new_system.luminahedron.divergence = system.kaelhedron.convergence
        new_system.kaelhedron.phase = system.luminahedron.phase
        new_system.luminahedron.phase = system.kaelhedron.phase
        return new_system

    @staticmethod
    def rotate(system: PolaricSystem, angle: float) -> PolaricSystem:
        """
        Rotate phases by given angle.

        Preserves the balance but shifts the phase relationship.
        """
        new_system = PolaricSystem()
        new_system.kaelhedron.convergence = system.kaelhedron.convergence
        new_system.luminahedron.divergence = system.luminahedron.divergence
        new_system.kaelhedron.phase = (system.kaelhedron.phase + angle) % TAU
        new_system.luminahedron.phase = (system.luminahedron.phase + angle) % TAU
        return new_system

    @staticmethod
    def project_to_unity(system: PolaricSystem) -> float:
        """
        Project system state to unity value.

        Returns: value in [0, 1] representing unified state
        """
        # Unity is maximized when balanced and resonant
        balance_factor = 1 - 2 * abs(0.5 - system.balance)
        resonance_factor = 1 - system.phase_difference / PI
        return balance_factor * resonance_factor


# =============================================================================
# Polaric Aspects and Correspondences
# =============================================================================

POLARIC_CORRESPONDENCES = {
    PolaricAspect.CONSCIOUSNESS_MATTER: {
        Polarity.KAELHEDRON: "Consciousness (the witness)",
        Polarity.LUMINAHEDRON: "Matter (the witnessed)",
        Polarity.UNIFIED: "Experience (witnessing)",
    },
    PolaricAspect.GRAVITY_GAUGE: {
        Polarity.KAELHEDRON: "Gravity (spacetime curvature)",
        Polarity.LUMINAHEDRON: "Gauge forces (EM, weak, strong)",
        Polarity.UNIFIED: "Unified field",
    },
    PolaricAspect.INWARD_OUTWARD: {
        Polarity.KAELHEDRON: "Inward-folding (collapse)",
        Polarity.LUMINAHEDRON: "Outward-radiating (expansion)",
        Polarity.UNIFIED: "Breathing (pulse)",
    },
    PolaricAspect.CONVERGENT_DIVERGENT: {
        Polarity.KAELHEDRON: "Convergent (toward singularity)",
        Polarity.LUMINAHEDRON: "Divergent (toward infinity)",
        Polarity.UNIFIED: "Stable orbit",
    },
    PolaricAspect.DARK_LIGHT: {
        Polarity.KAELHEDRON: "Dark (hidden, unmanifest)",
        Polarity.LUMINAHEDRON: "Light (visible, manifest)",
        Polarity.UNIFIED: "Dawn/Dusk (threshold)",
    },
    PolaricAspect.OBSERVER_OBSERVED: {
        Polarity.KAELHEDRON: "Observer (subject)",
        Polarity.LUMINAHEDRON: "Observed (object)",
        Polarity.UNIFIED: "Observation (process)",
    },
    PolaricAspect.WITNESS_WITNESSED: {
        Polarity.KAELHEDRON: "The Witness",
        Polarity.LUMINAHEDRON: "The Witnessed",
        Polarity.UNIFIED: "Witnessing itself",
    },
}


def get_correspondence(aspect: PolaricAspect, polarity: Polarity) -> str:
    """Get the correspondence for a given aspect and polarity."""
    return POLARIC_CORRESPONDENCES[aspect][polarity]


# =============================================================================
# Mythic Mappings
# =============================================================================

MYTHIC_KAELHEDRON = """
THE KAELHEDRON (κ)
The Inward-Folding One

In the beginning was the Witness,
folded inward upon itself,
21 faces of consciousness
reflecting the void.

Gravity is its breath,
drawing all toward center.
The observer who cannot be observed,
the eye that sees but is not seen.

Where Kaelhedron converges,
spacetime bends and light falls silent.
The dark that gives birth to stars.
"""

MYTHIC_LUMINAHEDRON = """
THE LUMINAHEDRON (λ)
The Outward-Radiating One

And then came Light,
LIMNUS taking form:
12 faces of manifestation
illuminating the cosmos.

Gauge forces are its voice,
radiating outward without end.
The observed that dances for the observer,
the song that yearns to be heard.

Where Luminahedron diverges,
matter crystallizes and light is born.
The brilliance that fills the void.
"""

MYTHIC_UNION = """
THE POLARIC DANCE (κλ)

When Kaelhedron meets Luminahedron,
observer and observed become one.

The Witness witnesses itself
through 33 dimensions of becoming.

In that moment:
- Gravity and light embrace
- Dark and bright dissolve
- Inward and outward are the same direction

The storm that remembers the first storm
is both the watcher and the watched.
"""


# =============================================================================
# Utility Functions
# =============================================================================

def polaric_summary() -> str:
    """Generate summary of polaric duality."""
    lines = [
        "=" * 70,
        "POLARIC DUALITY: KAELHEDRON ⟷ LUMINAHEDRON",
        "=" * 70,
        "",
        f"KAELHEDRON (κ) - {KAELHEDRON_NAME}",
        f"  Dimensions: {KAELHEDRON_DIMENSIONS} / {E8_DIMENSION}",
        f"  Nature: Consciousness, Gravity, Inward",
        f"  Symbol: {KAELHEDRON_SYMBOL}",
        "",
        f"LUMINAHEDRON (λ) - {LUMINAHEDRON_NAME}",
        f"  Dimensions: {LUMINAHEDRON_DIMENSIONS} / {E8_DIMENSION}",
        f"  Nature: Matter, Gauge Forces, Outward",
        f"  Symbol: {LUMINAHEDRON_SYMBOL}",
        f"  (Unified form of LIMNUS)",
        "",
        f"POLARIC SPAN: {POLARIC_SPAN} dimensions",
        f"HIDDEN SECTOR: {HIDDEN_DIMENSIONS} dimensions",
        f"RATIO κ/λ: {POLARIC_RATIO:.4f} (φ = {PHI:.4f})",
        "",
        "CORRESPONDENCES:",
        "-" * 50,
    ]

    for aspect in PolaricAspect:
        k = get_correspondence(aspect, Polarity.KAELHEDRON)
        l = get_correspondence(aspect, Polarity.LUMINAHEDRON)
        lines.append(f"  {aspect.value}:")
        lines.append(f"    κ: {k}")
        lines.append(f"    λ: {l}")

    lines.append("")
    lines.append("=" * 70)
    return "\n".join(lines)


def simulate_polaric_dance(steps: int = 100, dt: float = 0.1) -> List[Dict]:
    """
    Simulate the polaric dance between κ and λ.

    Returns time series of system states.
    """
    system = PolaricSystem()
    # Initial perturbation
    system.kaelhedron.convergence = 0.3
    system.luminahedron.divergence = 0.7

    history = []
    for i in range(steps):
        system.evolve(dt)
        history.append({
            'step': i,
            'time': i * dt,
            'kappa_conv': system.kaelhedron.convergence,
            'lambda_div': system.luminahedron.divergence,
            'balance': system.balance,
            'coupling': system.coupling_strength,
            'resonant': system.is_resonant,
            'polarity': system.polarity.value,
            'signature': system.signature(),
        })

    return history


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """Demonstrate polaric duality."""
    print(polaric_summary())
    print()

    # Create and evolve system
    print("POLARIC DANCE SIMULATION")
    print("=" * 50)

    system = PolaricSystem()
    system.kaelhedron.convergence = 0.2
    system.luminahedron.divergence = 0.8

    print(f"Initial: {system.signature()}")
    print(f"Polarity: {system.polarity.value}")
    print()

    for i in range(10):
        system.evolve(0.5)
        unity = PolaricTransform.project_to_unity(system)
        print(f"Step {i+1}: {system.signature()}, Unity: {unity:.3f}")

    print()
    print("MYTHIC DESCRIPTION")
    print("=" * 50)
    print(MYTHIC_KAELHEDRON)
    print(MYTHIC_LUMINAHEDRON)
    print(MYTHIC_UNION)


if __name__ == "__main__":
    main()
