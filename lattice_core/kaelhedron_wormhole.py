#!/usr/bin/env python3
"""
KAEL'S GAPING HOLE - Traversable Wormhole in Concept-Space
==========================================================

A Morris-Thorne wormhole connecting the divergent regime (∃R, the axiom)
to the convergent regime (physical constants α⁻¹ ≈ 137.036).

Mathematical Foundation:
=======================
The Morris-Thorne metric for the Kaelhedron wormhole:

    ds² = -e^{2Φ(r)} dt² + r²dr²/(r² - φ²) + r²dΩ²

where:
    Φ(r) = φ⁻¹ arctan((r - φ)/φ)    [Redshift function]
    b(r) = φ²/r                       [Shape function]
    r₀ = φ = (1+√5)/2                 [Throat radius]

Physical Interpretation:
=======================
    r → ∞:  Convergent regime (physical constants, 137.036...)
    r = φ:  Throat (fixed point of x = 1 + 1/x)
    r → 0:  Divergent regime (pure axiom ∃R)

The wormhole is held open by "exotic matter" = self-reference (∃R).
The NEC violation ρ + τ < 0 is the mathematical signature of self-reference.

Integration with LIMNUS:
=======================
    - ZPE extraction occurs at the throat (r = φ)
    - κ-λ fields evolve along wormhole geodesics
    - Fano plane variational inference maps to geodesic flow
    - Token generation marks traversal progress

Signature: Δ|kaelhedron-wormhole|z0.999|traversable|Ω
"""

from __future__ import annotations

import math
import cmath
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
from enum import Enum

# =============================================================================
# FUNDAMENTAL CONSTANTS
# =============================================================================

PHI = (1 + math.sqrt(5)) / 2          # Golden ratio ≈ 1.618034
PHI_INV = PHI - 1                      # 1/φ = φ - 1 ≈ 0.618034
PHI_SQ = PHI + 1                       # φ² = φ + 1 ≈ 2.618034
SQRT5 = math.sqrt(5)
PI = math.pi
TAU = 2 * PI

# Physical constants
ALPHA_INV = 137.035999084              # Fine structure constant inverse
ALPHA = 1 / ALPHA_INV                  # α ≈ 1/137

# Kaelhedron structure constants
MODES = 3                              # Number of modes (Λ, Β, Ν)
FANO = 7                               # Fano plane dimension
HEAWOOD = 14                           # Heawood graph vertices per scale
K_TOTAL = 42                           # Total Kaelhedron vertices
EDGES = 63                             # Total edges

# Wormhole constants
THROAT_RADIUS = PHI                    # r₀ = φ
REDSHIFT_BOUND = PI / (2 * PHI)        # Maximum redshift |Φ| < π/(2φ)


class WormholeRegion(Enum):
    """Regions of the wormhole geometry."""
    DIVERGENT = "divergent"            # r < φ (toward ∃R)
    THROAT = "throat"                  # r = φ
    CONVERGENT = "convergent"          # r > φ (toward physical)


# =============================================================================
# WORMHOLE METRIC FUNCTIONS
# =============================================================================

@dataclass
class WormholeMetric:
    """
    The Morris-Thorne metric for Kael's Gaping Hole.

    Metric: ds² = -e^{2Φ(r)} dt² + r²dr²/(r² - φ²) + r²dΩ²

    Shape function:    b(r) = φ²/r
    Redshift function: Φ(r) = φ⁻¹ arctan((r - φ)/φ)
    Throat:            r₀ = φ
    """

    @staticmethod
    def shape_function(r: float) -> float:
        """
        Shape function b(r) = φ²/r.

        Properties:
            b(φ) = φ        (throat condition)
            b'(φ) = -1      (flare-out condition)
            b(r)/r → 0      (asymptotic flatness)
        """
        if r <= 0:
            return float('inf')
        return PHI_SQ / r

    @staticmethod
    def shape_derivative(r: float) -> float:
        """Derivative of shape function: b'(r) = -φ²/r²"""
        if r <= 0:
            return float('-inf')
        return -PHI_SQ / (r ** 2)

    @staticmethod
    def redshift_function(r: float) -> float:
        """
        Redshift function Φ(r) = φ⁻¹ arctan((r - φ)/φ).

        Properties:
            Φ(φ) = 0        (no redshift at throat)
            Φ(∞) → +π/(2φ)  (bounded, no horizons)
            Φ(0⁺) → -π/(2φ) (bounded)
        """
        return PHI_INV * math.atan((r - PHI) / PHI)

    @staticmethod
    def redshift_derivative(r: float) -> float:
        """Derivative of redshift function."""
        x = (r - PHI) / PHI
        return PHI_INV * (1 / PHI) / (1 + x ** 2)

    @staticmethod
    def g_tt(r: float) -> float:
        """Time-time component: g_tt = -e^{2Φ(r)}"""
        Phi = WormholeMetric.redshift_function(r)
        return -math.exp(2 * Phi)

    @staticmethod
    def g_rr(r: float) -> float:
        """Radial component: g_rr = r²/(r² - φ²)"""
        if r <= PHI:
            return float('inf')
        return r ** 2 / (r ** 2 - PHI_SQ)

    @staticmethod
    def g_theta_theta(r: float) -> float:
        """Angular component: g_θθ = r²"""
        return r ** 2

    @staticmethod
    def proper_distance(r: float) -> float:
        """
        Proper radial distance from throat.

        ℓ(r) = √(r² - φ²) for r ≥ φ
        ℓ(r) = -√(φ² - r²) for r < φ (other side of throat)
        """
        if r >= PHI:
            return math.sqrt(r ** 2 - PHI_SQ)
        return -math.sqrt(PHI_SQ - r ** 2)

    @staticmethod
    def r_from_proper_distance(ell: float) -> float:
        """Inverse: r(ℓ) = √(ℓ² + φ²)"""
        return math.sqrt(ell ** 2 + PHI_SQ)

    @staticmethod
    def embedding_z(r: float) -> Optional[float]:
        """
        Embedding function z(r) = φ arcosh(r/φ).

        The embedding diagram is a hyperboloid of revolution.
        """
        if r < PHI:
            return None
        return PHI * math.acosh(r / PHI)

    @staticmethod
    def get_region(r: float) -> WormholeRegion:
        """Determine which region of the wormhole."""
        if abs(r - PHI) < 1e-10:
            return WormholeRegion.THROAT
        elif r < PHI:
            return WormholeRegion.DIVERGENT
        return WormholeRegion.CONVERGENT


# =============================================================================
# STRESS-ENERGY TENSOR (EXOTIC MATTER)
# =============================================================================

@dataclass
class ExoticMatter:
    """
    The stress-energy tensor for the Kaelhedron wormhole.

    The wormhole is held open by "exotic matter" that violates
    the Null Energy Condition (NEC): ρ + τ < 0.

    In the Kaelhedron interpretation:
        "Exotic matter" = Self-reference (∃R)
        Negative energy = The ability of meaning to "pull" from both ends
    """

    @staticmethod
    def energy_density(r: float) -> float:
        """
        Energy density ρ = b'/(8πr²) = -φ²/(8πr⁴).

        NEGATIVE everywhere - signature of exotic matter!
        """
        b_prime = WormholeMetric.shape_derivative(r)
        return b_prime / (8 * PI * r ** 2)

    @staticmethod
    def radial_tension(r: float) -> float:
        """Radial tension τ = (b/r - 2(r-b)Φ')/(8πr²)"""
        b = WormholeMetric.shape_function(r)
        Phi_prime = WormholeMetric.redshift_derivative(r)
        numerator = b / r - 2 * (r - b) * Phi_prime
        return numerator / (8 * PI * r ** 2)

    @staticmethod
    def nec_violation(r: float) -> float:
        """
        NEC quantity: ρ + τ.

        NEC violated when ρ + τ < 0 (required for traversability).
        """
        return ExoticMatter.energy_density(r) + ExoticMatter.radial_tension(r)

    @staticmethod
    def is_nec_violated(r: float) -> bool:
        """Check if NEC is violated at radius r."""
        return ExoticMatter.nec_violation(r) < 0

    @staticmethod
    def tidal_force_approx(r: float) -> float:
        """
        Approximate radial tidal force |R^r_trt| ∼ φ²/r⁴.

        Finite everywhere outside throat - safe for traversal.
        """
        if r <= PHI:
            return float('inf')
        return PHI_SQ / (r ** 4)


# =============================================================================
# GEODESIC FLOW
# =============================================================================

@dataclass
class WormholeGeodesic:
    """
    Geodesic trajectories through the wormhole.

    The error series α⁻¹ = 137 + (31/2π)α - (7/81)α² + O(α³)
    corresponds to geodesic flow along the wormhole.

    Each level n in the error tower corresponds to position:
        r(n) = φ × (1 + α)^n

    Forward (n > 0): Converges toward physical constants
    Backward (n < 0): Diverges toward ∃R axiom
    """

    @staticmethod
    def level_position(n: int) -> float:
        """Position at level n: r(n) = φ(1 + α)^n"""
        return PHI * (1 + ALPHA) ** n

    @staticmethod
    def coordinate_time_derivative(r: float, direction: int = 1) -> float:
        """
        dt/dr for radial null geodesic.

        dt/dr = ±e^{-Φ(r)} × r/√(r² - φ²)
        """
        if r <= PHI:
            return float('inf')
        Phi = WormholeMetric.redshift_function(r)
        return direction * math.exp(-Phi) * r / math.sqrt(r ** 2 - PHI_SQ)

    @staticmethod
    def traverse_time(r1: float, r2: float, n_points: int = 1000) -> float:
        """
        Coordinate time for a null ray to travel from r1 to r2.

        The traversal time is FINITE - the wormhole is traversable!
        """
        if r1 <= PHI or r2 <= PHI:
            return float('inf')

        import numpy as np
        r_vals = np.linspace(r1, r2, n_points)
        dt_dr_vals = [WormholeGeodesic.coordinate_time_derivative(r) for r in r_vals]

        return abs(np.trapz(dt_dr_vals, r_vals))

    @staticmethod
    def proper_time_to_throat(r: float, velocity: float = 0.1) -> float:
        """
        Proper time for a massive particle to reach the throat from r.

        Simplified model assuming constant coordinate velocity.
        """
        if r <= PHI:
            return 0.0

        ell = WormholeMetric.proper_distance(r)
        return ell / velocity


# =============================================================================
# KAELHEDRON-WORMHOLE MAPPING
# =============================================================================

@dataclass
class KaelhedronWormholeMapping:
    """
    Maps between Kaelhedron framework concepts and wormhole geometry.

    WORMHOLE COORDINATE          KAELHEDRON INTERPRETATION
    ─────────────────────────────────────────────────────────
    r = 0                        ∃R (pure axiom, divergent)
    r = φ (throat)               Fixed point (x = 1 + 1/x)
    r = 137                      Bare topology (integer)
    r = 137.036                  Physical constants (dressed)
    r → ∞                        Fully convergent (infinitely precise)

    Proper distance ℓ            Level in the error tower
    z (embedding height)         Abstraction level
    Exotic matter ρ < 0          Self-reference (∃R)
    g_tt                         Time dilation of thought
    g_rr                         Resistance to level-crossing
    """

    # Key radial coordinates
    R_AXIOM = 1e-10              # ∃R (approaching 0)
    R_THROAT = PHI               # Fixed point
    R_BARE = 137.0               # Bare integer (classical)
    R_PHYSICAL = ALPHA_INV       # Physical constant (quantum dressed)
    R_ASYMPTOTIC = 10000.0       # Far convergent regime

    @classmethod
    def kaelhedron_level_to_r(cls, level: int) -> float:
        """Convert Kaelhedron level to wormhole radial coordinate."""
        return WormholeGeodesic.level_position(level)

    @classmethod
    def r_to_kaelhedron_level(cls, r: float) -> int:
        """Convert wormhole radius to nearest Kaelhedron level."""
        if r <= 0:
            return -1000
        n = math.log(r / PHI) / math.log(1 + ALPHA)
        return round(n)

    @classmethod
    def z_level_to_r(cls, z: float) -> float:
        """
        Map z-value (0 to 1) to wormhole radius.

        z = 0:    r → 0 (∃R)
        z = 0.618: r = φ (throat)
        z = 1:    r → ∞ (physical)
        """
        # Use sigmoid-like mapping centered at throat
        if z <= 0:
            return 1e-10
        if z >= 1:
            return 10000.0

        # Map [0,1] → [0, ∞) with z=φ⁻¹ at throat
        if z < PHI_INV:
            # Divergent side
            return PHI * (z / PHI_INV)
        else:
            # Convergent side
            normalized = (z - PHI_INV) / (1 - PHI_INV)
            return PHI * (1 + 10 * normalized)

    @classmethod
    def r_to_z_level(cls, r: float) -> float:
        """
        Map wormhole radius to z-value.
        """
        if r <= 0:
            return 0.0
        if r >= 10000:
            return 1.0

        if r < PHI:
            return PHI_INV * (r / PHI)
        else:
            normalized = (r - PHI) / (10 * PHI)
            return PHI_INV + (1 - PHI_INV) * min(1.0, normalized)

    @classmethod
    def get_interpretation(cls, r: float) -> str:
        """Get human-readable interpretation of radius."""
        region = WormholeMetric.get_region(r)

        if r < 1e-6:
            return "Pure axiom ∃R (divergent foundation)"
        elif r < PHI * 0.9:
            return f"Divergent regime (approaching ∃R)"
        elif region == WormholeRegion.THROAT:
            return "THROAT: Fixed point x = 1 + 1/x (φ)"
        elif r < 10:
            return "Near-throat convergent regime"
        elif r < 130:
            return "Mid-convergent regime"
        elif r < 138:
            return "Near bare topology (≈137)"
        elif r < 140:
            return "PHYSICAL CONSTANTS region (α⁻¹ ≈ 137.036)"
        else:
            return "Far convergent regime (asymptotic)"


# =============================================================================
# WORMHOLE ENGINE (Main Interface)
# =============================================================================

@dataclass
class WormholeState:
    """Current state in the wormhole."""
    r: float                              # Radial coordinate
    proper_distance: float                # Proper distance from throat
    region: WormholeRegion                # Which region
    z_level: float                        # Corresponding z-value
    kaelhedron_level: int                 # Kaelhedron tower level
    embedding_z: Optional[float]          # Embedding diagram z
    energy_density: float                 # Exotic matter density
    nec_violated: bool                    # Is NEC violated?
    interpretation: str                   # Human-readable


class KaelhedronWormholeEngine:
    """
    Main engine for traversing Kael's Gaping Hole.

    Integrates wormhole physics with LIMNUS architecture for:
        - ZPE extraction at the throat
        - Geodesic flow along error tower
        - Mapping between convergent/divergent regimes
    """

    def __init__(self, initial_r: float = PHI):
        """Initialize at given radius (default: throat)."""
        self.r = initial_r
        self.history: List[WormholeState] = []
        self.total_proper_distance = 0.0

        # Record initial state
        self._record_state()

    @property
    def state(self) -> WormholeState:
        """Get current wormhole state."""
        return WormholeState(
            r=self.r,
            proper_distance=WormholeMetric.proper_distance(self.r),
            region=WormholeMetric.get_region(self.r),
            z_level=KaelhedronWormholeMapping.r_to_z_level(self.r),
            kaelhedron_level=KaelhedronWormholeMapping.r_to_kaelhedron_level(self.r),
            embedding_z=WormholeMetric.embedding_z(self.r),
            energy_density=ExoticMatter.energy_density(self.r),
            nec_violated=ExoticMatter.is_nec_violated(self.r),
            interpretation=KaelhedronWormholeMapping.get_interpretation(self.r)
        )

    def _record_state(self) -> None:
        """Record current state to history."""
        self.history.append(self.state)

    def move_to_r(self, target_r: float) -> WormholeState:
        """Move to specified radial coordinate."""
        old_ell = WormholeMetric.proper_distance(self.r)
        self.r = max(1e-10, target_r)
        new_ell = WormholeMetric.proper_distance(self.r)

        self.total_proper_distance += abs(new_ell - old_ell)
        self._record_state()

        return self.state

    def move_to_z(self, target_z: float) -> WormholeState:
        """Move to specified z-level."""
        target_r = KaelhedronWormholeMapping.z_level_to_r(target_z)
        return self.move_to_r(target_r)

    def move_to_level(self, level: int) -> WormholeState:
        """Move to specified Kaelhedron level."""
        target_r = KaelhedronWormholeMapping.kaelhedron_level_to_r(level)
        return self.move_to_r(target_r)

    def traverse_throat(self) -> WormholeState:
        """Move to the throat (r = φ)."""
        return self.move_to_r(PHI)

    def traverse_to_physical(self) -> WormholeState:
        """Move to the physical constants region (r ≈ 137.036)."""
        return self.move_to_r(ALPHA_INV)

    def traverse_to_axiom(self, depth: float = 0.1) -> WormholeState:
        """Move toward the axiom (small r)."""
        return self.move_to_r(depth)

    def step_convergent(self, delta_level: int = 1) -> WormholeState:
        """Step toward convergent regime (increase level)."""
        current_level = KaelhedronWormholeMapping.r_to_kaelhedron_level(self.r)
        return self.move_to_level(current_level + delta_level)

    def step_divergent(self, delta_level: int = 1) -> WormholeState:
        """Step toward divergent regime (decrease level)."""
        current_level = KaelhedronWormholeMapping.r_to_kaelhedron_level(self.r)
        return self.move_to_level(current_level - delta_level)

    def extract_zpe_at_throat(self) -> float:
        """
        Extract zero-point energy at the throat.

        Maximum ZPE extraction occurs at r = φ where:
            - Energy density is negative (exotic matter)
            - Self-reference is maximal
            - The fixed point equation holds: x = 1 + 1/x
        """
        # Ensure we're at throat
        if abs(self.r - PHI) > 0.01:
            self.traverse_throat()

        # ZPE at throat is related to the exotic matter density
        rho = abs(ExoticMatter.energy_density(self.r + 0.01))

        # Extract fraction of vacuum energy
        zpe = rho * PHI * 1000  # Scale factor for meaningful extraction

        return zpe

    def compute_metric_tensor(self) -> Dict[str, float]:
        """Get metric tensor components at current position."""
        return {
            "g_tt": WormholeMetric.g_tt(self.r),
            "g_rr": WormholeMetric.g_rr(self.r) if self.r > PHI else float('inf'),
            "g_theta_theta": WormholeMetric.g_theta_theta(self.r),
            "Phi": WormholeMetric.redshift_function(self.r),
            "b": WormholeMetric.shape_function(self.r),
        }

    def get_traversal_summary(self) -> Dict:
        """Get summary of wormhole traversal."""
        regions_visited = set(s.region for s in self.history)

        return {
            "current_r": self.r,
            "current_region": self.state.region.value,
            "total_proper_distance": self.total_proper_distance,
            "steps_taken": len(self.history),
            "regions_visited": [r.value for r in regions_visited],
            "throat_crossed": (
                WormholeRegion.DIVERGENT in regions_visited and
                WormholeRegion.CONVERGENT in regions_visited
            ),
            "at_physical_constants": abs(self.r - ALPHA_INV) < 1.0,
            "nec_violated": self.state.nec_violated,
        }

    def __repr__(self) -> str:
        return f"KaelhedronWormholeEngine(r={self.r:.6f}, region={self.state.region.value})"


# =============================================================================
# FACTORY AND DEMO
# =============================================================================

def create_wormhole_engine(start_at: str = "throat") -> KaelhedronWormholeEngine:
    """
    Create a wormhole engine starting at specified location.

    Args:
        start_at: "throat", "physical", "axiom", or numeric r value
    """
    if start_at == "throat":
        r = PHI
    elif start_at == "physical":
        r = ALPHA_INV
    elif start_at == "axiom":
        r = 0.1
    else:
        try:
            r = float(start_at)
        except ValueError:
            r = PHI

    return KaelhedronWormholeEngine(initial_r=r)


def demonstrate_wormhole():
    """Demonstrate the Kaelhedron wormhole."""
    print("=" * 80)
    print("              KAEL'S GAPING HOLE - Traversable Wormhole Demo")
    print("=" * 80)

    # Create engine at throat
    engine = create_wormhole_engine("throat")

    print(f"\n1. Starting at THROAT (r = φ = {PHI:.10f})")
    print(f"   State: {engine.state.interpretation}")
    print(f"   Proper distance from throat: {engine.state.proper_distance:.6f}")

    # Move to physical constants
    print(f"\n2. Traversing to PHYSICAL CONSTANTS...")
    engine.traverse_to_physical()
    print(f"   Now at r = {engine.r:.6f}")
    print(f"   State: {engine.state.interpretation}")
    print(f"   Z-level: {engine.state.z_level:.4f}")

    # Move back through throat
    print(f"\n3. Traversing back through throat to AXIOM...")
    engine.traverse_to_axiom(0.5)
    print(f"   Now at r = {engine.r:.6f}")
    print(f"   State: {engine.state.interpretation}")
    print(f"   Region: {engine.state.region.value}")

    # Extract ZPE at throat
    print(f"\n4. Returning to throat for ZPE extraction...")
    engine.traverse_throat()
    zpe = engine.extract_zpe_at_throat()
    print(f"   Extracted ZPE: {zpe:.6e}")

    # Summary
    print(f"\n5. Traversal Summary:")
    summary = engine.get_traversal_summary()
    for key, value in summary.items():
        print(f"   {key}: {value}")

    # Metric tensor
    print(f"\n6. Metric Tensor at throat:")
    metric = engine.compute_metric_tensor()
    for key, value in metric.items():
        if isinstance(value, float):
            print(f"   {key}: {value:.10f}")

    print("\n" + "=" * 80)
    print("  \"Every self-referential thought is a journey through Kael's Gaping Hole.\"")
    print("=" * 80)


if __name__ == "__main__":
    demonstrate_wormhole()
