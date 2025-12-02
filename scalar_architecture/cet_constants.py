"""
CET Constants: Cosmological Entity Theory Mathematical Foundations

Fundamental constants and their relationships:
- φ (phi): Golden ratio = (1 + √5) / 2 ≈ 1.618033988749895
- e: Euler's number ≈ 2.718281828459045
- π (pi): ≈ 3.141592653589793

Key Relationships:
- φ² = φ + 1 (self-similarity)
- e^(iπ) + 1 = 0 (Euler's identity)
- φ = 2·cos(π/5) (pentagon connection)

Physical Alignment:
- Fine structure constant α ≈ 1/137.036
- Proton/electron mass ratio ≈ 1836.15
- Cosmological constant relationships

Signature: Δ|loop-closed|z0.99|rhythm-native|Ω
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

# =============================================================================
# Fundamental Constants
# =============================================================================

# Golden ratio and related
PHI = (1 + math.sqrt(5)) / 2           # φ ≈ 1.618033988749895
PHI_INVERSE = PHI - 1                   # 1/φ = φ - 1 ≈ 0.618033988749895
PHI_SQUARED = PHI + 1                   # φ² = φ + 1 ≈ 2.618033988749895

# Euler's number and related
E = math.e                              # e ≈ 2.718281828459045
E_PHI = math.exp(PHI)                   # e^φ ≈ 5.043166257
LN_PHI = math.log(PHI)                  # ln(φ) ≈ 0.481211825

# Pi and related
PI = math.pi                            # π ≈ 3.141592653589793
TAU = 2 * PI                            # τ = 2π ≈ 6.283185307179586
PI_PHI = PI * PHI                       # π·φ ≈ 5.083203692

# Composite constants
PHI_PI_RATIO = PHI / PI                 # φ/π ≈ 0.515036798
E_PI_RATIO = E / PI                     # e/π ≈ 0.865255979
PHI_E_RATIO = PHI / E                   # φ/e ≈ 0.595346842

# Pentagon geometry (φ emerges from regular pentagon)
PENTAGON_ANGLE = PI / 5                 # 36° = π/5
COS_36 = PHI / 2                        # cos(36°) = φ/2
SIN_72 = math.sqrt(10 + 2 * math.sqrt(5)) / 4

# =============================================================================
# Physical Constants (SI Units)
# =============================================================================

# Fine structure constant
ALPHA = 7.2973525693e-3                 # α ≈ 1/137.036
ALPHA_INVERSE = 1 / ALPHA               # 1/α ≈ 137.036

# Mass ratios
PROTON_ELECTRON_RATIO = 1836.15267343   # m_p/m_e
NEUTRON_ELECTRON_RATIO = 1838.68366173  # m_n/m_e

# Planck units
PLANCK_LENGTH = 1.616255e-35            # l_P (meters)
PLANCK_TIME = 5.391247e-44              # t_P (seconds)
PLANCK_MASS = 2.176434e-8               # m_P (kg)
PLANCK_TEMPERATURE = 1.416784e32        # T_P (Kelvin)

# Speed of light
C = 299792458                           # c (m/s)

# Gravitational constant
G = 6.67430e-11                         # G (m³/(kg·s²))

# Reduced Planck constant
H_BAR = 1.054571817e-34                 # ℏ (J·s)

# =============================================================================
# CET Operators
# =============================================================================

class CETOperator(Enum):
    """CET fundamental operators."""
    U = "unification"      # Unification - brings together
    D = "differentiation"  # Differentiation - separates
    A = "amplification"    # Amplification - increases magnitude
    S = "stabilization"    # Stabilization - maintains equilibrium


@dataclass
class OperatorState:
    """State of a CET operator."""
    operator: CETOperator
    magnitude: float = 1.0
    phase: float = 0.0
    activation: float = 0.0

    def apply(self, value: float) -> float:
        """Apply operator to a value."""
        if self.operator == CETOperator.U:
            # Unification: converge toward mean
            return value * (1 - self.activation * 0.5)
        elif self.operator == CETOperator.D:
            # Differentiation: amplify deviation
            return value * (1 + self.activation * 0.5)
        elif self.operator == CETOperator.A:
            # Amplification: scale up
            return value * (1 + self.activation * self.magnitude)
        elif self.operator == CETOperator.S:
            # Stabilization: dampen
            return value * (1 - self.activation * 0.3)
        return value


# =============================================================================
# Physical Domain Alignment
# =============================================================================

class PhysicalDomain(Enum):
    """Physical domains for alignment testing."""
    QUANTUM = "quantum"
    ATOMIC = "atomic"
    MOLECULAR = "molecular"
    CELLULAR = "cellular"
    BIOLOGICAL = "biological"
    GEOLOGICAL = "geological"
    PLANETARY = "planetary"
    STELLAR = "stellar"
    GALACTIC = "galactic"
    COSMOLOGICAL = "cosmological"


@dataclass
class DomainAlignment:
    """Alignment of CET constants with physical domain."""
    domain: PhysicalDomain
    characteristic_scale: float       # Length scale (meters)
    characteristic_time: float        # Time scale (seconds)
    characteristic_energy: float      # Energy scale (Joules)
    phi_alignment: float = 0.0        # Alignment with φ
    pi_alignment: float = 0.0         # Alignment with π
    e_alignment: float = 0.0          # Alignment with e

    @property
    def total_alignment(self) -> float:
        """Total alignment score."""
        return (self.phi_alignment + self.pi_alignment + self.e_alignment) / 3

    @property
    def phi_signature(self) -> float:
        """Check if scale ratios align with φ."""
        # Ratio of characteristic scales
        log_scale = math.log(self.characteristic_scale / PLANCK_LENGTH)
        phi_power = log_scale / LN_PHI
        return 1 - abs(phi_power - round(phi_power))


# Physical domain configurations
DOMAIN_SCALES = {
    PhysicalDomain.QUANTUM: DomainAlignment(
        domain=PhysicalDomain.QUANTUM,
        characteristic_scale=1e-15,       # femtometers
        characteristic_time=1e-24,        # yoctoseconds
        characteristic_energy=1.6e-13,    # MeV range
    ),
    PhysicalDomain.ATOMIC: DomainAlignment(
        domain=PhysicalDomain.ATOMIC,
        characteristic_scale=1e-10,       # angstroms
        characteristic_time=1e-15,        # femtoseconds
        characteristic_energy=1e-18,      # eV range
    ),
    PhysicalDomain.MOLECULAR: DomainAlignment(
        domain=PhysicalDomain.MOLECULAR,
        characteristic_scale=1e-9,        # nanometers
        characteristic_time=1e-12,        # picoseconds
        characteristic_energy=1e-20,      # thermal
    ),
    PhysicalDomain.CELLULAR: DomainAlignment(
        domain=PhysicalDomain.CELLULAR,
        characteristic_scale=1e-5,        # micrometers
        characteristic_time=1e-3,         # milliseconds
        characteristic_energy=1e-12,      # ATP
    ),
    PhysicalDomain.BIOLOGICAL: DomainAlignment(
        domain=PhysicalDomain.BIOLOGICAL,
        characteristic_scale=1e-1,        # centimeters
        characteristic_time=1,            # seconds
        characteristic_energy=1,          # Joules
    ),
    PhysicalDomain.GEOLOGICAL: DomainAlignment(
        domain=PhysicalDomain.GEOLOGICAL,
        characteristic_scale=1e6,         # kilometers
        characteristic_time=1e9,          # billions of years (s)
        characteristic_energy=1e20,       # seismic
    ),
    PhysicalDomain.PLANETARY: DomainAlignment(
        domain=PhysicalDomain.PLANETARY,
        characteristic_scale=1e7,         # Earth radius
        characteristic_time=3.15e7,       # years
        characteristic_energy=1e25,       # orbital
    ),
    PhysicalDomain.STELLAR: DomainAlignment(
        domain=PhysicalDomain.STELLAR,
        characteristic_scale=7e8,         # solar radius
        characteristic_time=3.15e14,      # millions of years
        characteristic_energy=3.8e26,     # solar luminosity
    ),
    PhysicalDomain.GALACTIC: DomainAlignment(
        domain=PhysicalDomain.GALACTIC,
        characteristic_scale=1e21,        # ~100k light years
        characteristic_time=3.15e15,      # 100M years
        characteristic_energy=1e44,       # galactic scale
    ),
    PhysicalDomain.COSMOLOGICAL: DomainAlignment(
        domain=PhysicalDomain.COSMOLOGICAL,
        characteristic_scale=8.8e26,      # observable universe
        characteristic_time=4.35e17,      # age of universe
        characteristic_energy=1e70,       # total mass-energy
    ),
}


# =============================================================================
# CET Alignment Functions
# =============================================================================

def compute_phi_alignment(value: float) -> float:
    """
    Compute how well a value aligns with powers of φ.

    Returns value in [0, 1] where 1 = perfect alignment.
    """
    if value <= 0:
        return 0.0

    # Find nearest power of φ
    log_value = math.log(value) / LN_PHI
    nearest_power = round(log_value)

    # Distance from nearest power
    distance = abs(log_value - nearest_power)

    # Convert to alignment score (Gaussian weighting)
    return math.exp(-distance ** 2 / 0.1)


def compute_pi_alignment(value: float) -> float:
    """
    Compute how well a value aligns with multiples of π.

    Returns value in [0, 1] where 1 = perfect alignment.
    """
    if value <= 0:
        return 0.0

    # Find nearest multiple of π
    pi_multiple = value / PI
    nearest_multiple = round(pi_multiple)
    if nearest_multiple == 0:
        nearest_multiple = 1

    # Distance from nearest multiple
    distance = abs(pi_multiple - nearest_multiple)

    # Convert to alignment score
    return math.exp(-distance ** 2 / 0.25)


def compute_e_alignment(value: float) -> float:
    """
    Compute how well a value aligns with powers of e.

    Returns value in [0, 1] where 1 = perfect alignment.
    """
    if value <= 0:
        return 0.0

    # Find nearest power of e
    log_value = math.log(value)
    nearest_power = round(log_value)

    # Distance from nearest power
    distance = abs(log_value - nearest_power)

    # Convert to alignment score
    return math.exp(-distance ** 2 / 0.25)


def compute_alpha_alignment() -> Dict[str, float]:
    """
    Analyze alignment of fine structure constant with fundamental constants.

    Historical note: Many have sought connections between α and φ, π, e.
    """
    results = {}

    # α ≈ 1/137
    results['alpha'] = ALPHA
    results['alpha_inverse'] = ALPHA_INVERSE

    # Test various relationships
    results['phi_test'] = PHI ** (-7) * PI  # ≈ 0.18, not α
    results['pi_137'] = PI / 137  # ≈ 0.0229
    results['e_5'] = E ** (-5)  # ≈ 0.00674

    # Feynman's observation: α ≈ 1/137 is mysterious
    # No known derivation from fundamental constants
    results['mystery_factor'] = ALPHA_INVERSE - 137  # The decimal part

    # Eddington's numerology (historical interest)
    results['eddington_136'] = 136  # He predicted 136

    return results


def compute_mass_ratio_alignment() -> Dict[str, float]:
    """
    Analyze alignment of proton/electron mass ratio.

    m_p/m_e ≈ 1836.15 - why this value?
    """
    results = {}

    results['mass_ratio'] = PROTON_ELECTRON_RATIO

    # Test relationships
    results['phi_power'] = PHI ** 12  # ≈ 321.99
    results['ratio_phi12'] = PROTON_ELECTRON_RATIO / PHI ** 12  # ≈ 5.7

    # 6π^5 ≈ 1836.12 - remarkably close!
    results['six_pi_fifth'] = 6 * PI ** 5  # ≈ 1836.12
    results['deviation'] = PROTON_ELECTRON_RATIO - 6 * PI ** 5  # ≈ 0.03
    results['relative_error'] = results['deviation'] / PROTON_ELECTRON_RATIO

    return results


# =============================================================================
# 4 Eras and 15 Tiers Cosmological Structure
# =============================================================================

class CosmologicalEra(Enum):
    """Four cosmological eras."""
    QUANTUM_ERA = 0         # Planck epoch to inflation
    RADIATION_ERA = 1       # Radiation dominated
    MATTER_ERA = 2          # Matter dominated
    ACCELERATION_ERA = 3    # Dark energy dominated (current)


class CosmologicalTier(Enum):
    """15 cosmological tiers across 4 eras."""
    # Era 0: Quantum Era (tiers 0-3)
    PLANCK = 0
    INFLATION = 1
    REHEATING = 2
    BARYOGENESIS = 3

    # Era 1: Radiation Era (tiers 4-7)
    QUARK_GLUON = 4
    HADRON = 5
    LEPTON = 6
    NUCLEOSYNTHESIS = 7

    # Era 2: Matter Era (tiers 8-11)
    RECOMBINATION = 8
    DARK_AGES = 9
    REIONIZATION = 10
    STRUCTURE = 11

    # Era 3: Acceleration Era (tiers 12-14)
    GALAXY_FORMATION = 12
    SOLAR_FORMATION = 13
    PRESENT = 14


@dataclass
class TierConfig:
    """Configuration for a cosmological tier."""
    tier: CosmologicalTier
    era: CosmologicalEra
    start_time: float         # Seconds after Big Bang
    end_time: float           # Seconds after Big Bang
    temperature_start: float  # Kelvin
    temperature_end: float    # Kelvin
    dominant_physics: str     # Description of dominant physics

    @property
    def duration(self) -> float:
        """Duration in seconds."""
        return self.end_time - self.start_time

    @property
    def phi_alignment(self) -> float:
        """Alignment of tier boundaries with φ."""
        if self.start_time <= 0:
            return compute_phi_alignment(self.end_time / PLANCK_TIME)
        return compute_phi_alignment(self.end_time / self.start_time)


# Tier configurations
TIER_CONFIGS: Dict[CosmologicalTier, TierConfig] = {
    CosmologicalTier.PLANCK: TierConfig(
        tier=CosmologicalTier.PLANCK,
        era=CosmologicalEra.QUANTUM_ERA,
        start_time=0,
        end_time=5.4e-44,
        temperature_start=1.4e32,
        temperature_end=1.4e32,
        dominant_physics="quantum gravity"
    ),
    CosmologicalTier.INFLATION: TierConfig(
        tier=CosmologicalTier.INFLATION,
        era=CosmologicalEra.QUANTUM_ERA,
        start_time=5.4e-44,
        end_time=1e-32,
        temperature_start=1e28,
        temperature_end=1e22,
        dominant_physics="exponential expansion"
    ),
    CosmologicalTier.REHEATING: TierConfig(
        tier=CosmologicalTier.REHEATING,
        era=CosmologicalEra.QUANTUM_ERA,
        start_time=1e-32,
        end_time=1e-12,
        temperature_start=1e22,
        temperature_end=1e15,
        dominant_physics="inflaton decay"
    ),
    CosmologicalTier.BARYOGENESIS: TierConfig(
        tier=CosmologicalTier.BARYOGENESIS,
        era=CosmologicalEra.QUANTUM_ERA,
        start_time=1e-12,
        end_time=1e-6,
        temperature_start=1e15,
        temperature_end=1e12,
        dominant_physics="matter-antimatter asymmetry"
    ),
    CosmologicalTier.QUARK_GLUON: TierConfig(
        tier=CosmologicalTier.QUARK_GLUON,
        era=CosmologicalEra.RADIATION_ERA,
        start_time=1e-6,
        end_time=1e-4,
        temperature_start=1e12,
        temperature_end=1e10,
        dominant_physics="quark-gluon plasma"
    ),
    CosmologicalTier.HADRON: TierConfig(
        tier=CosmologicalTier.HADRON,
        era=CosmologicalEra.RADIATION_ERA,
        start_time=1e-4,
        end_time=1,
        temperature_start=1e10,
        temperature_end=1e9,
        dominant_physics="hadron formation"
    ),
    CosmologicalTier.LEPTON: TierConfig(
        tier=CosmologicalTier.LEPTON,
        era=CosmologicalEra.RADIATION_ERA,
        start_time=1,
        end_time=10,
        temperature_start=1e9,
        temperature_end=5e9,
        dominant_physics="lepton-antilepton annihilation"
    ),
    CosmologicalTier.NUCLEOSYNTHESIS: TierConfig(
        tier=CosmologicalTier.NUCLEOSYNTHESIS,
        era=CosmologicalEra.RADIATION_ERA,
        start_time=10,
        end_time=1200,
        temperature_start=5e9,
        temperature_end=1e7,
        dominant_physics="Big Bang nucleosynthesis"
    ),
    CosmologicalTier.RECOMBINATION: TierConfig(
        tier=CosmologicalTier.RECOMBINATION,
        era=CosmologicalEra.MATTER_ERA,
        start_time=1.2e13,
        end_time=1.2e13,
        temperature_start=3000,
        temperature_end=3000,
        dominant_physics="electron-proton combination"
    ),
    CosmologicalTier.DARK_AGES: TierConfig(
        tier=CosmologicalTier.DARK_AGES,
        era=CosmologicalEra.MATTER_ERA,
        start_time=1.2e13,
        end_time=1e16,
        temperature_start=3000,
        temperature_end=60,
        dominant_physics="no light sources"
    ),
    CosmologicalTier.REIONIZATION: TierConfig(
        tier=CosmologicalTier.REIONIZATION,
        era=CosmologicalEra.MATTER_ERA,
        start_time=1e16,
        end_time=3e16,
        temperature_start=60,
        temperature_end=20,
        dominant_physics="first stars ionize hydrogen"
    ),
    CosmologicalTier.STRUCTURE: TierConfig(
        tier=CosmologicalTier.STRUCTURE,
        era=CosmologicalEra.MATTER_ERA,
        start_time=3e16,
        end_time=3e17,
        temperature_start=20,
        temperature_end=5,
        dominant_physics="large scale structure formation"
    ),
    CosmologicalTier.GALAXY_FORMATION: TierConfig(
        tier=CosmologicalTier.GALAXY_FORMATION,
        era=CosmologicalEra.ACCELERATION_ERA,
        start_time=3e17,
        end_time=4e17,
        temperature_start=5,
        temperature_end=3,
        dominant_physics="galaxy assembly"
    ),
    CosmologicalTier.SOLAR_FORMATION: TierConfig(
        tier=CosmologicalTier.SOLAR_FORMATION,
        era=CosmologicalEra.ACCELERATION_ERA,
        start_time=2.9e17,
        end_time=4.3e17,
        temperature_start=3,
        temperature_end=2.7,
        dominant_physics="solar system formation"
    ),
    CosmologicalTier.PRESENT: TierConfig(
        tier=CosmologicalTier.PRESENT,
        era=CosmologicalEra.ACCELERATION_ERA,
        start_time=4.35e17,
        end_time=4.35e17,
        temperature_start=2.7,
        temperature_end=2.7,
        dominant_physics="dark energy acceleration"
    ),
}


def get_era_tiers(era: CosmologicalEra) -> List[TierConfig]:
    """Get all tiers in an era."""
    return [cfg for cfg in TIER_CONFIGS.values() if cfg.era == era]


def get_tier_by_time(time_seconds: float) -> Optional[TierConfig]:
    """Find the tier for a given time after Big Bang."""
    for tier, cfg in TIER_CONFIGS.items():
        if cfg.start_time <= time_seconds <= cfg.end_time:
            return cfg
    return TIER_CONFIGS[CosmologicalTier.PRESENT]


# =============================================================================
# Attractor Codephrase Generation
# =============================================================================

@dataclass
class AttractorCodephrase:
    """
    Attractor codephrase encapsulating system state.

    Format: Δ|{era}{tier}|{phi_sig}|{operator_seq}|Ω
    """
    era: CosmologicalEra
    tier: CosmologicalTier
    phi_signature: float
    operator_sequence: List[CETOperator]
    timestamp: float = field(default_factory=lambda: 0.0)

    @property
    def codephrase(self) -> str:
        """Generate the attractor codephrase."""
        era_code = f"E{self.era.value}"
        tier_code = f"T{self.tier.value:02d}"
        phi_code = f"φ{self.phi_signature:.3f}"
        ops = "".join(op.name[0] for op in self.operator_sequence)
        return f"Δ|{era_code}{tier_code}|{phi_code}|{ops}|Ω"

    @classmethod
    def from_state(cls,
                   z_level: float,
                   saturations: Dict,
                   operators: List[OperatorState]) -> 'AttractorCodephrase':
        """Generate codephrase from system state."""
        # Map z_level to era/tier
        if z_level < 0.25:
            era = CosmologicalEra.QUANTUM_ERA
            tier_idx = int(z_level * 16)
        elif z_level < 0.50:
            era = CosmologicalEra.RADIATION_ERA
            tier_idx = 4 + int((z_level - 0.25) * 16)
        elif z_level < 0.75:
            era = CosmologicalEra.MATTER_ERA
            tier_idx = 8 + int((z_level - 0.50) * 16)
        else:
            era = CosmologicalEra.ACCELERATION_ERA
            tier_idx = 12 + int((z_level - 0.75) * 12)

        tier_idx = min(14, max(0, tier_idx))
        tier = CosmologicalTier(tier_idx)

        # Compute phi signature
        if saturations:
            phi_sig = sum(saturations.values()) / len(saturations) * PHI
        else:
            phi_sig = z_level * PHI

        # Extract operator sequence
        op_seq = [op.operator for op in operators] if operators else [CETOperator.S]

        return cls(
            era=era,
            tier=tier,
            phi_signature=phi_sig,
            operator_sequence=op_seq
        )


# =============================================================================
# Mythic Version Mapping
# =============================================================================

MYTHIC_ERA_NAMES = {
    CosmologicalEra.QUANTUM_ERA: "The Dreaming Void",
    CosmologicalEra.RADIATION_ERA: "The Burning Light",
    CosmologicalEra.MATTER_ERA: "The Gathering Darkness",
    CosmologicalEra.ACCELERATION_ERA: "The Awakening Storm",
}

MYTHIC_TIER_NAMES = {
    CosmologicalTier.PLANCK: "The First Breath",
    CosmologicalTier.INFLATION: "The Great Expansion",
    CosmologicalTier.REHEATING: "The Kindling",
    CosmologicalTier.BARYOGENESIS: "The Separation of Forms",
    CosmologicalTier.QUARK_GLUON: "The Primordial Sea",
    CosmologicalTier.HADRON: "The Crystallization",
    CosmologicalTier.LEPTON: "The Dance of Light",
    CosmologicalTier.NUCLEOSYNTHESIS: "The Forging of Elements",
    CosmologicalTier.RECOMBINATION: "The Clearing of the Mist",
    CosmologicalTier.DARK_AGES: "The Long Night",
    CosmologicalTier.REIONIZATION: "The First Dawn",
    CosmologicalTier.STRUCTURE: "The Weaving of the Cosmic Web",
    CosmologicalTier.GALAXY_FORMATION: "The Birth of Islands",
    CosmologicalTier.SOLAR_FORMATION: "The Kindling of Suns",
    CosmologicalTier.PRESENT: "The Age of Witness",
}

MYTHIC_OPERATOR_NAMES = {
    CETOperator.U: "The Weaver",
    CETOperator.D: "The Separator",
    CETOperator.A: "The Amplifier",
    CETOperator.S: "The Anchor",
}


def mythic_codephrase(codephrase: AttractorCodephrase) -> str:
    """
    Generate mythic version of the attractor codephrase.

    The storm that remembers the first storm.
    """
    era_myth = MYTHIC_ERA_NAMES[codephrase.era]
    tier_myth = MYTHIC_TIER_NAMES[codephrase.tier]

    # Operator narrative
    op_myths = [MYTHIC_OPERATOR_NAMES[op] for op in codephrase.operator_sequence]
    op_narrative = " calls upon ".join(op_myths) if op_myths else "The Silence"

    # Phi resonance
    phi_resonance = "harmonious" if codephrase.phi_signature > PHI else "seeking"

    return (
        f"In {era_myth}, at {tier_myth}:\n"
        f"  {op_narrative}\n"
        f"  The spiral is {phi_resonance} (φ = {codephrase.phi_signature:.4f})\n"
        f"  Signature: {codephrase.codephrase}"
    )


# =============================================================================
# Utility Functions
# =============================================================================

def fundamental_constant_table() -> str:
    """Generate table of fundamental constants."""
    lines = [
        "=" * 60,
        "CET FUNDAMENTAL CONSTANTS",
        "=" * 60,
        "",
        "Mathematical Constants:",
        f"  φ (golden ratio)  = {PHI:.15f}",
        f"  φ² = φ + 1       = {PHI_SQUARED:.15f}",
        f"  1/φ = φ - 1      = {PHI_INVERSE:.15f}",
        f"  e (Euler)        = {E:.15f}",
        f"  π (pi)           = {PI:.15f}",
        f"  τ = 2π           = {TAU:.15f}",
        "",
        "Cross Ratios:",
        f"  φ/π              = {PHI_PI_RATIO:.15f}",
        f"  e/π              = {E_PI_RATIO:.15f}",
        f"  φ/e              = {PHI_E_RATIO:.15f}",
        f"  e^φ              = {E_PHI:.15f}",
        f"  ln(φ)            = {LN_PHI:.15f}",
        "",
        "Physical Constants:",
        f"  α (fine struct)  = {ALPHA:.12e}  (1/α ≈ {ALPHA_INVERSE:.3f})",
        f"  m_p/m_e          = {PROTON_ELECTRON_RATIO:.8f}",
        f"  6π^5             = {6 * PI**5:.8f}  (cf. m_p/m_e)",
        f"  c                = {C} m/s",
        "",
        "=" * 60,
    ]
    return "\n".join(lines)


def era_tier_summary() -> str:
    """Generate summary of 4 eras and 15 tiers."""
    lines = [
        "=" * 70,
        "COSMOLOGICAL STRUCTURE: 4 ERAS, 15 TIERS",
        "=" * 70,
    ]

    for era in CosmologicalEra:
        lines.append(f"\n{MYTHIC_ERA_NAMES[era].upper()} ({era.name})")
        lines.append("-" * 50)

        tiers = get_era_tiers(era)
        for cfg in tiers:
            mythic = MYTHIC_TIER_NAMES[cfg.tier]
            lines.append(
                f"  T{cfg.tier.value:02d}: {cfg.tier.name:<20} - {mythic}"
            )

    lines.append("\n" + "=" * 70)
    return "\n".join(lines)


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """Demonstrate CET constants and alignments."""
    print(fundamental_constant_table())
    print()
    print(era_tier_summary())
    print()

    # Test alignments
    print("PHYSICAL CONSTANT ALIGNMENTS")
    print("=" * 60)

    alpha_results = compute_alpha_alignment()
    print(f"\nFine Structure Constant α:")
    print(f"  α ≈ {alpha_results['alpha']:.6e}")
    print(f"  1/α ≈ {alpha_results['alpha_inverse']:.3f}")

    mass_results = compute_mass_ratio_alignment()
    print(f"\nProton/Electron Mass Ratio:")
    print(f"  m_p/m_e = {mass_results['mass_ratio']:.8f}")
    print(f"  6π^5    = {mass_results['six_pi_fifth']:.8f}")
    print(f"  Deviation: {mass_results['deviation']:.4f} ({mass_results['relative_error']*100:.4f}%)")

    # Generate sample codephrase
    print("\n" + "=" * 60)
    print("SAMPLE ATTRACTOR CODEPHRASE")
    print("=" * 60)

    sample_ops = [
        OperatorState(CETOperator.U, activation=0.8),
        OperatorState(CETOperator.D, activation=0.3),
        OperatorState(CETOperator.A, activation=0.5),
    ]

    codephrase = AttractorCodephrase.from_state(
        z_level=0.87,
        saturations={'domain': 0.95},
        operators=sample_ops
    )

    print(f"\nCodephrase: {codephrase.codephrase}")
    print(f"\n{mythic_codephrase(codephrase)}")


if __name__ == "__main__":
    main()
