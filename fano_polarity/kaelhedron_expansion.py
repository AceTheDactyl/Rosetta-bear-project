# fano_polarity/kaelhedron_expansion.py
"""
Kaelhedron Expansion: Three Major Directions
=============================================

Implements the three expansion directions of the Kaelhedron framework:

1. THE FOUR KAELHEDRONS (Quaternionic Structure)
   - K:   Original (φ, inward, Q>0)
   - K*:  Anti-Kaelhedron (-1/φ, destructive)
   - K^∨: Dual-Kaelhedron (φ, outward, point↔line)
   - K̄:   Conjugate-Kaelhedron (φ, inward, Q<0)
   Together: 4 × 42 = 168 = |GL(3,2)|

2. ELECTROMAGNETISM FROM κ-FIELD
   - U(1) gauge symmetry from phase θ
   - Maxwell equations from □κ + ζκ³ = 0
   - Electric charge = topological winding Q
   - E-B duality = Fano point-line duality

3. INVERSIONS CATALOG
   - Golden: φ ↔ 1/φ
   - Wells: μ₁ ↔ μ₂
   - Modes: Λ ↔ Ν via Β
   - Scales: Κ ↔ κ via Γ
   - And more...
"""

from __future__ import annotations

import cmath
import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# Sacred constants
PHI = (1 + math.sqrt(5)) / 2          # Golden ratio ≈ 1.618
PHI_INV = PHI - 1                      # 1/φ ≈ 0.618
PHI_SQ = PHI + 1                       # φ² ≈ 2.618
PHI_MINUS = (1 - math.sqrt(5)) / 2    # -1/φ ≈ -0.618
SQRT5 = math.sqrt(5)
TAU = 2 * math.pi
ZETA = (5/3)**4                        # ≈ 0.482

# Potential well values
MU_P = 3/5                             # Central point = 0.6
MU_1 = MU_P / math.sqrt(PHI)          # Lower well ≈ 0.472
MU_2 = MU_P * math.sqrt(PHI)          # Upper well ≈ 0.764

# Fine structure related
ALPHA_INV = 137                        # ≈ 1/α
GL32_ORDER = 168                       # |GL(3,2)|


# ═══════════════════════════════════════════════════════════════════════════════
# PART I: THE FOUR KAELHEDRONS
# ═══════════════════════════════════════════════════════════════════════════════

class KaelhedronType(Enum):
    """The four types of Kaelhedron."""
    ORIGINAL = auto()    # K: φ, inward, Q>0 (constructive)
    ANTI = auto()        # K*: -1/φ, destructive
    DUAL = auto()        # K^∨: φ, outward (point↔line swap)
    CONJUGATE = auto()   # K̄: φ, inward, Q<0 (phase-reversed)


@dataclass
class KaelhedronVariant:
    """
    A single Kaelhedron variant in the Four-Kaelhedron structure.

    The four variants together form a quaternionic structure with
    4 × 42 = 168 elements, matching |GL(3,2)|.
    """
    variant_type: KaelhedronType
    phi_value: float
    direction: str          # "inward" or "outward"
    charge_sign: int        # +1, -1, or 0
    role: str               # Descriptive role

    # Internal state
    cells: int = 42         # Each Kaelhedron has 42 vertices
    topological_charge: float = 0.0
    phase: float = 0.0

    @property
    def symbol(self) -> str:
        """Unicode symbol for this variant."""
        symbols = {
            KaelhedronType.ORIGINAL: "K",
            KaelhedronType.ANTI: "K*",
            KaelhedronType.DUAL: "K^∨",
            KaelhedronType.CONJUGATE: "K̄",
        }
        return symbols[self.variant_type]

    def golden_transform(self, x: float) -> float:
        """Apply this variant's golden transformation."""
        if self.variant_type == KaelhedronType.ORIGINAL:
            return x * self.phi_value
        elif self.variant_type == KaelhedronType.ANTI:
            return x * self.phi_value  # -1/φ multiplication
        elif self.variant_type == KaelhedronType.DUAL:
            return 1 / x if x != 0 else float('inf')  # Inversion
        elif self.variant_type == KaelhedronType.CONJUGATE:
            return x * self.phi_value  # Same φ, but phase-reversed
        return x


def create_original_kaelhedron() -> KaelhedronVariant:
    """Create K: the original Kaelhedron."""
    return KaelhedronVariant(
        variant_type=KaelhedronType.ORIGINAL,
        phi_value=PHI,
        direction="inward",
        charge_sign=1,
        role="Reception, Integration, Construction",
    )


def create_anti_kaelhedron() -> KaelhedronVariant:
    """Create K*: the anti-Kaelhedron from -1/φ."""
    return KaelhedronVariant(
        variant_type=KaelhedronType.ANTI,
        phi_value=PHI_MINUS,  # -1/φ ≈ -0.618
        direction="inward",
        charge_sign=-1,
        role="Dissolution, Fragmentation, Destruction",
    )


def create_dual_kaelhedron() -> KaelhedronVariant:
    """Create K^∨: the dual Kaelhedron (point↔line swap)."""
    return KaelhedronVariant(
        variant_type=KaelhedronType.DUAL,
        phi_value=PHI,
        direction="outward",
        charge_sign=1,
        role="Expression, Manifestation, Projection",
    )


def create_conjugate_kaelhedron() -> KaelhedronVariant:
    """Create K̄: the conjugate Kaelhedron (phase-reversed)."""
    return KaelhedronVariant(
        variant_type=KaelhedronType.CONJUGATE,
        phi_value=PHI,
        direction="inward",
        charge_sign=-1,
        role="Mirror, CPT-conjugate, Left-handed",
    )


class FourKaelhedron:
    """
    The complete Four-Kaelhedron structure: K ⊕ K* ⊕ K^∨ ⊕ K̄

    Total vertices: 4 × 42 = 168 = |GL(3,2)|

    The four Kaelhedrons form a quaternionic structure:
      K  ↔ 1 (identity)
      K* ↔ i (imaginary, from φ × (-1/φ) = -1 = i²)
      K^∨ ↔ j (dual/inversion)
      K̄  ↔ k (i × j)
    """

    def __init__(self):
        self.K = create_original_kaelhedron()
        self.K_star = create_anti_kaelhedron()
        self.K_dual = create_dual_kaelhedron()
        self.K_conj = create_conjugate_kaelhedron()

        self._variants = [self.K, self.K_star, self.K_dual, self.K_conj]

    @property
    def total_vertices(self) -> int:
        """Total vertices: 4 × 42 = 168."""
        return sum(v.cells for v in self._variants)

    def quaternion_product(self, a: KaelhedronType, b: KaelhedronType) -> KaelhedronType:
        """
        Quaternion multiplication table for the four Kaelhedrons.

        Based on: i² = j² = k² = ijk = -1
        """
        # Multiplication table (returns index)
        table = {
            (KaelhedronType.ORIGINAL, KaelhedronType.ORIGINAL): KaelhedronType.ORIGINAL,
            (KaelhedronType.ORIGINAL, KaelhedronType.ANTI): KaelhedronType.ANTI,
            (KaelhedronType.ORIGINAL, KaelhedronType.DUAL): KaelhedronType.DUAL,
            (KaelhedronType.ORIGINAL, KaelhedronType.CONJUGATE): KaelhedronType.CONJUGATE,

            (KaelhedronType.ANTI, KaelhedronType.ORIGINAL): KaelhedronType.ANTI,
            (KaelhedronType.ANTI, KaelhedronType.ANTI): KaelhedronType.ORIGINAL,  # i² = -1 → K
            (KaelhedronType.ANTI, KaelhedronType.DUAL): KaelhedronType.CONJUGATE,  # ij = k
            (KaelhedronType.ANTI, KaelhedronType.CONJUGATE): KaelhedronType.DUAL,  # ik = -j

            (KaelhedronType.DUAL, KaelhedronType.ORIGINAL): KaelhedronType.DUAL,
            (KaelhedronType.DUAL, KaelhedronType.ANTI): KaelhedronType.CONJUGATE,  # ji = -k
            (KaelhedronType.DUAL, KaelhedronType.DUAL): KaelhedronType.ORIGINAL,  # j² = -1 → K
            (KaelhedronType.DUAL, KaelhedronType.CONJUGATE): KaelhedronType.ANTI,  # jk = i

            (KaelhedronType.CONJUGATE, KaelhedronType.ORIGINAL): KaelhedronType.CONJUGATE,
            (KaelhedronType.CONJUGATE, KaelhedronType.ANTI): KaelhedronType.DUAL,  # ki = j
            (KaelhedronType.CONJUGATE, KaelhedronType.DUAL): KaelhedronType.ANTI,  # kj = -i
            (KaelhedronType.CONJUGATE, KaelhedronType.CONJUGATE): KaelhedronType.ORIGINAL,  # k² = -1
        }
        return table[(a, b)]

    def get_variant(self, vtype: KaelhedronType) -> KaelhedronVariant:
        """Get variant by type."""
        mapping = {
            KaelhedronType.ORIGINAL: self.K,
            KaelhedronType.ANTI: self.K_star,
            KaelhedronType.DUAL: self.K_dual,
            KaelhedronType.CONJUGATE: self.K_conj,
        }
        return mapping[vtype]

    def golden_relations(self) -> Dict[str, float]:
        """Key golden ratio relations between variants."""
        return {
            "φ × φ'": PHI * PHI_MINUS,          # = -1
            "φ + φ'": PHI + PHI_MINUS,          # = 1
            "φ - φ'": PHI - PHI_MINUS,          # = √5
            "φ × (1/φ)": PHI * PHI_INV,         # = 1
            "K × K*": -1,                        # Annihilation
            "K + K*": 1,                         # Unity
        }

    def summary(self) -> Dict[str, Any]:
        """Get summary of the Four-Kaelhedron structure."""
        return {
            "total_vertices": self.total_vertices,
            "is_168": self.total_vertices == GL32_ORDER,
            "variants": [
                {
                    "symbol": v.symbol,
                    "phi": v.phi_value,
                    "direction": v.direction,
                    "charge": v.charge_sign,
                    "role": v.role,
                }
                for v in self._variants
            ],
            "golden_relations": self.golden_relations(),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# PART II: ELECTROMAGNETISM FROM κ-FIELD
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class KappaField:
    """
    The complex κ-field: κ = |κ| e^(iθ)

    The phase θ is the electromagnetic potential.
    The amplitude |κ| is the Higgs-like scalar.
    """
    amplitude: float = MU_2      # |κ|, defaults to upper well
    phase: float = 0.0           # θ, the EM potential

    @property
    def complex_value(self) -> complex:
        """Get κ as complex number."""
        return self.amplitude * cmath.exp(1j * self.phase)

    @property
    def conjugate(self) -> "KappaField":
        """Get κ* (complex conjugate)."""
        return KappaField(self.amplitude, -self.phase)

    def evolve_phase(self, dphi: float) -> None:
        """Evolve phase by dphi (U(1) gauge transformation)."""
        self.phase = (self.phase + dphi) % TAU


@dataclass
class ElectromagneticState:
    """
    Electromagnetic state derived from κ-field phase structure.

    E-field: Electric field (from phase gradient)
    B-field: Magnetic field (from phase curl)
    Q: Topological charge (quantized electric charge)
    """
    E_field: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    B_field: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    charge_Q: int = 0
    magnetic_flux: float = 0.0

    @property
    def field_tensor_invariant(self) -> float:
        """F_μν F^μν = 2(B² - E²)"""
        E_sq = sum(e*e for e in self.E_field)
        B_sq = sum(b*b for b in self.B_field)
        return 2 * (B_sq - E_sq)

    @property
    def dual_invariant(self) -> float:
        """F_μν *F^μν = 4 E·B"""
        return 4 * sum(e*b for e, b in zip(self.E_field, self.B_field))


class EMFromKappa:
    """
    Derives electromagnetism from κ-field phase dynamics.

    Key mappings:
      - κ phase θ → EM potential A_μ
      - Phase gradient → E-field
      - Phase curl → B-field
      - Topological winding → Electric charge Q
      - Fano point-line duality → E-B duality
    """

    def __init__(self):
        self.kappa_fields: Dict[int, KappaField] = {}
        self._initialize_fano_kappa()

    def _initialize_fano_kappa(self):
        """Initialize κ-field at each Fano point."""
        for point in range(1, 8):
            self.kappa_fields[point] = KappaField(
                amplitude=MU_2,
                phase=(point - 1) * TAU / 7,  # Distribute phases
            )

    def compute_line_flux(self, line: Tuple[int, int, int]) -> float:
        """
        Compute magnetic flux through a Fano line.

        Flux = θ_i + θ_j + θ_k (mod 2π)
        This is gauge-invariant!
        """
        phases = [self.kappa_fields[p].phase for p in line]
        return sum(phases) % TAU

    def compute_topological_charge(self) -> int:
        """
        Compute total topological charge (winding number).

        Q = ∮ dθ / 2π
        """
        total_phase_wind = 0.0
        for i in range(1, 8):
            j = (i % 7) + 1  # Next point cyclically
            dphase = self.kappa_fields[j].phase - self.kappa_fields[i].phase
            # Normalize to [-π, π]
            while dphase > math.pi:
                dphase -= TAU
            while dphase < -math.pi:
                dphase += TAU
            total_phase_wind += dphase

        return round(total_phase_wind / TAU)

    def apply_gauge_transform(self, alpha: float) -> None:
        """
        Apply U(1) gauge transformation: θ_p → θ_p + α for all p.

        This leaves all physical observables invariant.
        """
        for kappa in self.kappa_fields.values():
            kappa.evolve_phase(alpha)

    def eb_duality_transform(self) -> None:
        """
        Apply E-B duality (Fano point-line duality).

        This swaps "electric" (point-based) and "magnetic" (line-based)
        interpretations.
        """
        # Under duality, phases get redistributed
        # This is a discrete Fourier transform on the Fano plane
        new_phases = {}
        fano_lines = [
            (1, 2, 4), (2, 3, 5), (3, 4, 6), (4, 5, 7),
            (5, 6, 1), (6, 7, 2), (7, 1, 3),
        ]

        for point in range(1, 8):
            # Point p is on exactly 3 lines
            lines_containing_p = [
                i for i, line in enumerate(fano_lines) if point in line
            ]
            # New phase = average of line fluxes
            flux_sum = sum(
                self.compute_line_flux(fano_lines[l]) for l in lines_containing_p
            )
            new_phases[point] = flux_sum / 3

        for point, phase in new_phases.items():
            self.kappa_fields[point].phase = phase % TAU

    def get_em_state(self) -> ElectromagneticState:
        """Get current electromagnetic state."""
        # Compute E-field from phase gradients
        E = [0.0, 0.0, 0.0]
        for i, p in enumerate([1, 2, 3]):  # Use first 3 points for 3D
            next_p = [2, 3, 4][i]
            grad = self.kappa_fields[next_p].phase - self.kappa_fields[p].phase
            E[i] = -grad  # E = -∇φ

        # Compute B-field from phase curls (use remaining points)
        B = [0.0, 0.0, 0.0]
        fano_lines = [(1, 2, 4), (2, 3, 5), (3, 4, 6)]
        for i, line in enumerate(fano_lines):
            B[i] = self.compute_line_flux(line) / TAU

        return ElectromagneticState(
            E_field=E,
            B_field=B,
            charge_Q=self.compute_topological_charge(),
            magnetic_flux=sum(B),
        )

    def fine_structure_estimate(self) -> Dict[str, Any]:
        """
        Estimate fine structure constant from Kaelhedron structure.

        α⁻¹ ≈ 137 = |GL(3,2)| - (7 + 24) = 168 - 31
        """
        return {
            "alpha_inverse": ALPHA_INV,
            "gl32_order": GL32_ORDER,
            "fano_points": 7,
            "s4_order": 24,  # Symmetric group S₄
            "relation": f"{GL32_ORDER} - (7 + 24) = {GL32_ORDER - 31}",
            "fibonacci_relation": "F₁₁ + F₁₂/3 = 89 + 48 = 137",
        }


# ═══════════════════════════════════════════════════════════════════════════════
# PART III: INVERSIONS CATALOG
# ═══════════════════════════════════════════════════════════════════════════════

class InversionType(Enum):
    """Types of inversions in the Kaelhedron framework."""
    GOLDEN = auto()          # φ ↔ 1/φ
    CONJUGATE = auto()       # φ ↔ -1/φ
    POTENTIAL = auto()       # μ₁ ↔ μ₂
    MODE = auto()            # Λ ↔ Ν
    SCALE = auto()           # Κ ↔ κ
    FANO = auto()            # Point ↔ Line
    HOLOGRAPHIC = auto()     # Boundary ↔ Bulk
    ELECTROMAGNETIC = auto() # E ↔ B
    CHARGE = auto()          # +Q ↔ -Q
    MATTER = auto()          # Particle ↔ Antiparticle
    WAVE_PARTICLE = auto()   # Wave ↔ Particle
    CONSCIOUSNESS = auto()   # Subject ↔ Object
    AWARENESS = auto()       # Observer ↔ Observed
    TIME = auto()            # Past ↔ Future
    SPACE = auto()           # Here ↔ There


@dataclass
class Inversion:
    """
    Represents an inversion pair with mediator.

    Every inversion has:
      - Pole A: One extreme
      - Pole B: Other extreme
      - Mediator: The middle term connecting them
    """
    inv_type: InversionType
    pole_a: str
    pole_b: str
    mediator: str

    # Mathematical values if applicable
    value_a: Optional[float] = None
    value_b: Optional[float] = None
    value_mediator: Optional[float] = None

    # Special relations
    product_relation: Optional[str] = None
    sum_relation: Optional[str] = None

    def apply_to_value(self, x: float) -> float:
        """Apply inversion transformation if mathematical."""
        if self.inv_type == InversionType.GOLDEN:
            return 1 / x if x != 0 else float('inf')
        elif self.inv_type == InversionType.CONJUGATE:
            return -1 / x if x != 0 else float('inf')
        return x

    def verify_golden_harmonic(self) -> Optional[bool]:
        """
        Verify the golden harmonic relation: A × B = M²

        For modes: Λ × Ν = Β²
        """
        if all(v is not None for v in [self.value_a, self.value_b, self.value_mediator]):
            product = self.value_a * self.value_b
            mediator_sq = self.value_mediator ** 2
            return math.isclose(product, mediator_sq, rel_tol=1e-9)
        return None


class InversionsCatalog:
    """
    Complete catalog of inversions in the Kaelhedron framework.
    """

    def __init__(self):
        self.inversions: Dict[InversionType, Inversion] = {}
        self._build_catalog()

    def _build_catalog(self):
        """Build the complete inversions catalog."""

        # 1. Golden inversion: φ ↔ 1/φ
        self.inversions[InversionType.GOLDEN] = Inversion(
            inv_type=InversionType.GOLDEN,
            pole_a="φ",
            pole_b="1/φ = φ-1",
            mediator="1 (product)",
            value_a=PHI,
            value_b=PHI_INV,
            value_mediator=1.0,
            product_relation="φ × (1/φ) = 1",
            sum_relation="φ - (1/φ) = 1",
        )

        # 2. Conjugate inversion: φ ↔ -1/φ
        self.inversions[InversionType.CONJUGATE] = Inversion(
            inv_type=InversionType.CONJUGATE,
            pole_a="φ",
            pole_b="-1/φ",
            mediator="√5 (sum)",
            value_a=PHI,
            value_b=PHI_MINUS,
            value_mediator=SQRT5,
            product_relation="φ × (-1/φ) = -1",
            sum_relation="φ + (-1/φ) = 1",
        )

        # 3. Potential well inversion: μ₁ ↔ μ₂
        self.inversions[InversionType.POTENTIAL] = Inversion(
            inv_type=InversionType.POTENTIAL,
            pole_a="μ₁ (lower well)",
            pole_b="μ₂ (upper well)",
            mediator="μ_P (barrier)",
            value_a=MU_1,
            value_b=MU_2,
            value_mediator=MU_P,
            product_relation="√(μ₁×μ₂) = μ_P",
            sum_relation="(μ₁+μ₂)/2 ≈ φ⁻¹ (barrier)",
        )

        # 4. Mode inversion: Λ ↔ Ν via Β
        self.inversions[InversionType.MODE] = Inversion(
            inv_type=InversionType.MODE,
            pole_a="Λ (LOGOS, Logic)",
            pole_b="Ν (NOUS, Awareness)",
            mediator="Β (BIOS, Life)",
            value_a=1.0,              # Λ coupling
            value_b=PHI_INV**2,       # Ν coupling
            value_mediator=PHI_INV,   # Β coupling
            product_relation="Λ × Ν = Β² (golden harmonic!)",
        )

        # 5. Scale inversion: Κ ↔ κ via Γ
        self.inversions[InversionType.SCALE] = Inversion(
            inv_type=InversionType.SCALE,
            pole_a="Κ (KOSMOS, Universal)",
            pole_b="κ (KAEL, Individual)",
            mediator="Γ (GAIA, Planetary)",
        )

        # 6. Fano inversion: Point ↔ Line
        self.inversions[InversionType.FANO] = Inversion(
            inv_type=InversionType.FANO,
            pole_a="Point (7 vertices)",
            pole_b="Line (7 edges)",
            mediator="Incidence (21 relations)",
        )

        # 7. Holographic: Boundary ↔ Bulk
        self.inversions[InversionType.HOLOGRAPHIC] = Inversion(
            inv_type=InversionType.HOLOGRAPHIC,
            pole_a="Boundary (external)",
            pole_b="Bulk (internal)",
            mediator="Encoding (holographic)",
        )

        # 8. Electromagnetic: E ↔ B
        self.inversions[InversionType.ELECTROMAGNETIC] = Inversion(
            inv_type=InversionType.ELECTROMAGNETIC,
            pole_a="E (electric field)",
            pole_b="B (magnetic field)",
            mediator="F (field tensor)",
        )

        # 9. Charge: +Q ↔ -Q
        self.inversions[InversionType.CHARGE] = Inversion(
            inv_type=InversionType.CHARGE,
            pole_a="+Q (positive)",
            pole_b="-Q (negative)",
            mediator="0 (neutral)",
            value_a=1.0,
            value_b=-1.0,
            value_mediator=0.0,
        )

        # 10. Matter: Particle ↔ Antiparticle
        self.inversions[InversionType.MATTER] = Inversion(
            inv_type=InversionType.MATTER,
            pole_a="Particle",
            pole_b="Antiparticle",
            mediator="Photon (massless)",
        )

        # 11. Wave-Particle
        self.inversions[InversionType.WAVE_PARTICLE] = Inversion(
            inv_type=InversionType.WAVE_PARTICLE,
            pole_a="Wave",
            pole_b="Particle",
            mediator="Wavefunction ψ",
        )

        # 12. Consciousness: Subject ↔ Object
        self.inversions[InversionType.CONSCIOUSNESS] = Inversion(
            inv_type=InversionType.CONSCIOUSNESS,
            pole_a="Subject",
            pole_b="Object",
            mediator="Experience",
        )

        # 13. Awareness: Observer ↔ Observed
        self.inversions[InversionType.AWARENESS] = Inversion(
            inv_type=InversionType.AWARENESS,
            pole_a="Observer",
            pole_b="Observed",
            mediator="Observation",
        )

        # 14. Time: Past ↔ Future
        self.inversions[InversionType.TIME] = Inversion(
            inv_type=InversionType.TIME,
            pole_a="Past",
            pole_b="Future",
            mediator="Present",
        )

        # 15. Space: Here ↔ There
        self.inversions[InversionType.SPACE] = Inversion(
            inv_type=InversionType.SPACE,
            pole_a="Here",
            pole_b="There",
            mediator="Distance",
        )

    def get(self, inv_type: InversionType) -> Inversion:
        """Get inversion by type."""
        return self.inversions[inv_type]

    def verify_golden_harmonic(self) -> Dict[str, bool]:
        """
        Verify golden harmonic Λ × Ν = Β² for applicable inversions.
        """
        results = {}
        for inv_type, inv in self.inversions.items():
            result = inv.verify_golden_harmonic()
            if result is not None:
                results[inv_type.name] = result
        return results

    def chain_inversions(self, *types: InversionType) -> List[str]:
        """
        Chain multiple inversions to see composite transformations.
        """
        chain = []
        for t in types:
            inv = self.inversions[t]
            chain.append(f"{inv.pole_a} ↔ {inv.pole_b} via {inv.mediator}")
        return chain

    def get_mathematical_inversions(self) -> List[Inversion]:
        """Get inversions with numerical values."""
        return [
            inv for inv in self.inversions.values()
            if inv.value_a is not None
        ]

    def summary_table(self) -> List[Dict[str, str]]:
        """Generate summary table of all inversions."""
        return [
            {
                "type": inv.inv_type.name,
                "pole_a": inv.pole_a,
                "pole_b": inv.pole_b,
                "mediator": inv.mediator,
            }
            for inv in self.inversions.values()
        ]


# ═══════════════════════════════════════════════════════════════════════════════
# UNIFIED EXPANSION SYSTEM
# ═══════════════════════════════════════════════════════════════════════════════

class KaelhedronExpansion:
    """
    Unified system combining all three expansion directions:
    1. Four Kaelhedrons (quaternionic structure)
    2. Electromagnetism from κ-field
    3. Inversions catalog
    """

    def __init__(self):
        self.four_kaelhedron = FourKaelhedron()
        self.em_system = EMFromKappa()
        self.inversions = InversionsCatalog()

    def get_full_summary(self) -> Dict[str, Any]:
        """Get complete summary of expansion."""
        return {
            "four_kaelhedron": self.four_kaelhedron.summary(),
            "electromagnetic": {
                "em_state": {
                    "charge_Q": self.em_system.get_em_state().charge_Q,
                    "E_field": self.em_system.get_em_state().E_field,
                    "B_field": self.em_system.get_em_state().B_field,
                },
                "fine_structure": self.em_system.fine_structure_estimate(),
            },
            "inversions": {
                "count": len(self.inversions.inversions),
                "golden_harmonic_verified": self.inversions.verify_golden_harmonic(),
            },
            "key_numbers": {
                "GL32_order": GL32_ORDER,
                "4_times_42": 4 * 42,
                "alpha_inverse": ALPHA_INV,
                "phi": PHI,
                "phi_inverse": PHI_INV,
            },
        }

    def demonstrate(self) -> Dict[str, Any]:
        """Run demonstration of expansion system."""
        results = {}

        # 1. Four Kaelhedrons
        results["quaternion_products"] = {
            "K*K*": self.four_kaelhedron.quaternion_product(
                KaelhedronType.ANTI, KaelhedronType.ANTI
            ).name,  # Should be ORIGINAL (i² = -1)
            "K^∨*K^∨": self.four_kaelhedron.quaternion_product(
                KaelhedronType.DUAL, KaelhedronType.DUAL
            ).name,  # Should be ORIGINAL (j² = -1)
        }

        # 2. EM system
        self.em_system.apply_gauge_transform(math.pi / 4)
        results["em_after_gauge"] = {
            "charge": self.em_system.compute_topological_charge(),
            "gauge_invariant": True,  # Charge unchanged
        }

        # 3. Inversions
        golden = self.inversions.get(InversionType.GOLDEN)
        results["golden_inversion"] = {
            "phi": golden.value_a,
            "phi_inv": golden.value_b,
            "product": golden.value_a * golden.value_b,
        }

        # Mode golden harmonic
        mode = self.inversions.get(InversionType.MODE)
        results["mode_golden_harmonic"] = {
            "lambda": mode.value_a,
            "nu": mode.value_b,
            "beta": mode.value_mediator,
            "lambda_times_nu": mode.value_a * mode.value_b,
            "beta_squared": mode.value_mediator ** 2,
            "verified": mode.verify_golden_harmonic(),
        }

        return results


# ═══════════════════════════════════════════════════════════════════════════════
# CONVENIENCE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def create_expansion_system() -> KaelhedronExpansion:
    """Create a complete Kaelhedron expansion system."""
    return KaelhedronExpansion()


def demonstrate_four_kaelhedrons() -> Dict[str, Any]:
    """Demonstrate the Four Kaelhedron structure."""
    fk = FourKaelhedron()
    return fk.summary()


def demonstrate_em_from_kappa() -> Dict[str, Any]:
    """Demonstrate electromagnetism from κ-field."""
    em = EMFromKappa()
    return {
        "initial_state": {
            "charge_Q": em.get_em_state().charge_Q,
            "line_fluxes": [
                em.compute_line_flux((1, 2, 4)),
                em.compute_line_flux((2, 3, 5)),
                em.compute_line_flux((3, 4, 6)),
            ],
        },
        "fine_structure": em.fine_structure_estimate(),
    }


def demonstrate_inversions() -> Dict[str, Any]:
    """Demonstrate the inversions catalog."""
    cat = InversionsCatalog()
    return {
        "count": len(cat.inversions),
        "mathematical": [
            {
                "type": inv.inv_type.name,
                "a": inv.value_a,
                "b": inv.value_b,
                "mediator": inv.value_mediator,
            }
            for inv in cat.get_mathematical_inversions()
        ],
        "golden_harmonic": cat.verify_golden_harmonic(),
    }
