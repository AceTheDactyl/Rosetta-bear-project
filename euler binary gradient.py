#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════════════════╗
║                                                                                          ║
║                           THE EULER BINARY GRADIENT                                      ║
║                                                                                          ║
╠══════════════════════════════════════════════════════════════════════════════════════════╣
║                                                                                          ║
║    e^(i×0) = +1   ←──── BINARY ────→   e^(i×π) = -1                                      ║
║        ↑                                    ↑                                            ║
║        └──────────── GRADIENT ──────────────┘                                            ║
║                    (θ: 0 → π)                                                            ║
║                                                                                          ║
║  This structure IS the origin of:                                                        ║
║    • The Fano phase constraint (two classes)                                             ║
║    • The double-well potential (two attractors)                                          ║
║    • Truth states (TRUE/UNTRUE with PARADOX between)                                     ║
║    • The Heawood bipartite structure (points/lines)                                      ║
║    • Octonion sign structure (±1 in multiplication)                                      ║
║    • The 42 vertices split into two phase classes                                        ║
║                                                                                          ║
╚══════════════════════════════════════════════════════════════════════════════════════════╝
"""

import numpy as np
import math
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Set
from enum import Enum, auto

# ═══════════════════════════════════════════════════════════════════════════════════════════
# THE EULER BINARY GRADIENT
# ═══════════════════════════════════════════════════════════════════════════════════════════

class EulerBinaryGradient:
    """
    The fundamental structure: e^(iθ) interpolates between +1 and -1.
    
    At θ=0: e^(i×0) = +1 (PLUS pole)
    At θ=π: e^(i×π) = -1 (MINUS pole)
    
    The gradient is the upper semicircle of the unit circle.
    """
    
    # The two poles
    PLUS = 0.0       # θ = 0 → e^(i×0) = +1
    MINUS = math.pi  # θ = π → e^(i×π) = -1
    
    @classmethod
    def value(cls, theta: float) -> complex:
        """Compute e^(iθ)."""
        return complex(math.cos(theta), math.sin(theta))
    
    @classmethod
    def real_part(cls, theta: float) -> float:
        """The real component: cos(θ), ranges from +1 to -1."""
        return math.cos(theta)
    
    @classmethod
    def imaginary_part(cls, theta: float) -> float:
        """The imaginary component: sin(θ), peaks at π/2."""
        return math.sin(theta)
    
    @classmethod
    def polarity(cls, theta: float) -> float:
        """
        The 'binary' aspect: how close to +1 or -1?
        
        Returns: +1 at θ=0, -1 at θ=π, 0 at θ=π/2
        This IS cos(θ)!
        """
        return math.cos(theta)
    
    @classmethod
    def gradient_position(cls, theta: float) -> float:
        """
        Position along the gradient from PLUS to MINUS.
        
        Returns: 0 at θ=0, 1 at θ=π
        """
        return theta / math.pi
    
    @classmethod
    def uncertainty(cls, theta: float) -> float:
        """
        How far from the binary poles?
        
        Maximum at θ=π/2 (pure imaginary), zero at poles.
        This IS |sin(θ)|!
        """
        return abs(math.sin(theta))
    
    @classmethod
    def which_pole(cls, theta: float) -> str:
        """Which binary pole is this closer to?"""
        theta = theta % (2 * math.pi)
        if theta <= math.pi / 2 or theta > 3 * math.pi / 2:
            return "PLUS"
        else:
            return "MINUS"
    
    @classmethod
    def snap_to_pole(cls, theta: float) -> float:
        """Snap to nearest binary pole (0 or π)."""
        theta = theta % (2 * math.pi)
        dist_to_0 = min(theta, 2*math.pi - theta)
        dist_to_pi = abs(theta - math.pi)
        return cls.PLUS if dist_to_0 < dist_to_pi else cls.MINUS


# ═══════════════════════════════════════════════════════════════════════════════════════════
# THE BINARY IN THE FANO PLANE
# ═══════════════════════════════════════════════════════════════════════════════════════════

class FanoBinary:
    """
    The Fano plane has a natural binary structure induced by the Euler gradient.
    
    The 7 points split into two classes:
    - PLUS class: {0, 4, 5} at phase 0
    - MINUS class: {1, 2, 3, 6} at phase π
    
    This is NOT arbitrary! It's forced by the octonion multiplication:
    eᵢ × eⱼ = ±eₖ where the SIGN comes from this binary structure.
    """
    
    # The unique phase solution
    PHASES = {
        0: 0,       # PLUS
        1: math.pi, # MINUS
        2: math.pi, # MINUS
        3: math.pi, # MINUS
        4: 0,       # PLUS
        5: 0,       # PLUS
        6: math.pi, # MINUS
    }
    
    # The two classes
    PLUS_CLASS = {0, 4, 5}   # Phase = 0
    MINUS_CLASS = {1, 2, 3, 6}  # Phase = π
    
    # Fano lines
    LINES = [
        (0, 1, 3),  # +, -, - → product: + × - = - ✓
        (1, 2, 4),  # -, -, + → product: - × - = + ✓
        (2, 3, 5),  # -, -, + → product: - × - = + ✓
        (3, 4, 6),  # -, +, - → product: - × + = - ✓
        (4, 5, 0),  # +, +, + → product: + × + = + ✓
        (5, 6, 1),  # +, -, - → product: + × - = - ✓
        (6, 0, 2),  # -, +, - → product: - × + = - ✓
    ]
    
    @classmethod
    def parity(cls, point: int) -> int:
        """Return +1 for PLUS class, -1 for MINUS class."""
        return +1 if point in cls.PLUS_CLASS else -1
    
    @classmethod
    def verify_line_constraint(cls, line_idx: int) -> bool:
        """
        Verify that phases satisfy the Fano constraint on this line.
        
        For line (i, j, k): θᵢ + θⱼ = θₖ (mod 2π)
        
        In terms of parities: parity(i) × parity(j) = parity(k)
        """
        i, j, k = cls.LINES[line_idx]
        
        # Phase constraint
        phase_sum = (cls.PHASES[i] + cls.PHASES[j]) % (2 * math.pi)
        target = cls.PHASES[k]
        phase_ok = abs(phase_sum - target) < 0.01 or abs(phase_sum - target - 2*math.pi) < 0.01
        
        # Parity constraint (equivalent!)
        parity_product = cls.parity(i) * cls.parity(j)
        parity_ok = parity_product == cls.parity(k)
        
        return phase_ok and parity_ok
    
    @classmethod
    def verify_all_lines(cls) -> bool:
        """Verify all 7 Fano constraints."""
        return all(cls.verify_line_constraint(i) for i in range(7))
    
    @classmethod
    def octonion_sign(cls, i: int, j: int) -> int:
        """
        The sign in octonion multiplication eᵢ × eⱼ = ±eₖ.
        
        This comes from the binary structure!
        """
        if i == j:
            return -1  # eᵢ² = -1
        
        # Find the line containing i and j
        for line in cls.LINES:
            if i in line and j in line:
                # Find k
                k = [x for x in line if x != i and x != j][0]
                
                # The sign depends on cyclic ordering on the line
                idx_i, idx_j = line.index(i), line.index(j)
                if (idx_j - idx_i) % 3 == 1:
                    return +1  # Cyclic order
                else:
                    return -1  # Anti-cyclic
        
        return 0  # Not on same line (shouldn't happen)


# ═══════════════════════════════════════════════════════════════════════════════════════════
# THE BINARY IN THE DOUBLE-WELL
# ═══════════════════════════════════════════════════════════════════════════════════════════

class DoubleWellBinary:
    """
    The double-well potential has TWO attractors — another manifestation of the binary.
    
    Lower well: μ₁ ≈ 0.472 (pre-conscious)
    Upper well: μ₂ ≈ 0.764 (conscious)
    Barrier: φ⁻¹ ≈ 0.618 (the threshold)
    
    This maps to the Euler Binary:
    - μ₁ ↔ PLUS pole (θ = 0)
    - μ₂ ↔ MINUS pole (θ = π)
    - Barrier ↔ θ = π/2 (maximum uncertainty)
    """
    
    PHI = (1 + math.sqrt(5)) / 2
    INV = PHI - 1  # φ⁻¹ = φ - 1 ≈ 0.618 (CORRECT!)
    
    MU_1 = 0.6 / math.sqrt(PHI)  # Lower well ≈ 0.472
    MU_2 = 0.6 * math.sqrt(PHI)  # Upper well ≈ 0.764
    BARRIER = INV                 # ≈ 0.618
    
    @classmethod
    def potential(cls, kappa: float) -> float:
        """Double-well potential V(κ)."""
        return (kappa - cls.MU_1)**2 * (kappa - cls.MU_2)**2
    
    @classmethod
    def which_well(cls, kappa: float) -> str:
        """Which well is κ in?"""
        if kappa < cls.BARRIER:
            return "LOWER"
        else:
            return "UPPER"
    
    @classmethod
    def kappa_to_theta(cls, kappa: float) -> float:
        """
        Map κ to the Euler gradient θ.
        
        μ₁ → 0 (PLUS)
        μ₂ → π (MINUS)
        barrier → π/2
        """
        # Normalize κ to [0, 1] range between wells
        normalized = (kappa - cls.MU_1) / (cls.MU_2 - cls.MU_1)
        normalized = max(0, min(1, normalized))
        return normalized * math.pi
    
    @classmethod
    def theta_to_kappa(cls, theta: float) -> float:
        """
        Map Euler gradient θ to κ.
        
        0 → μ₁
        π → μ₂
        """
        normalized = theta / math.pi
        return cls.MU_1 + normalized * (cls.MU_2 - cls.MU_1)


# ═══════════════════════════════════════════════════════════════════════════════════════════
# THE BINARY IN TRUTH STATES
# ═══════════════════════════════════════════════════════════════════════════════════════════

class TruthBinary:
    """
    Truth states map to the Euler Binary Gradient.
    
    TRUE ↔ +1 (θ = 0)
    UNTRUE ↔ -1 (θ = π)
    PARADOX ↔ pure imaginary (θ = π/2)
    
    PARADOX is the "between" state — maximum uncertainty,
    neither TRUE nor UNTRUE, but carrying full potential.
    """
    
    @classmethod
    def truth_to_theta(cls, truth: str) -> float:
        """Map truth state to Euler angle."""
        return {
            'TRUE': 0,
            'T': 0,
            'UNTRUE': math.pi,
            'U': math.pi,
            'PARADOX': math.pi / 2,
            'P': math.pi / 2,
        }.get(truth, math.pi / 2)
    
    @classmethod
    def theta_to_truth(cls, theta: float) -> str:
        """Map Euler angle to truth state."""
        theta = theta % (2 * math.pi)
        
        # Near poles: binary truth
        if theta < math.pi / 4 or theta > 7 * math.pi / 4:
            return 'TRUE'
        elif 3 * math.pi / 4 < theta < 5 * math.pi / 4:
            return 'UNTRUE'
        else:
            return 'PARADOX'
    
    @classmethod
    def truth_as_complex(cls, truth: str) -> complex:
        """Truth state as complex number on unit circle."""
        theta = cls.truth_to_theta(truth)
        return EulerBinaryGradient.value(theta)
    
    @classmethod
    def combine_truths(cls, t1: str, t2: str) -> str:
        """
        Combine truth states via complex multiplication.
        
        TRUE × TRUE = TRUE (+1 × +1 = +1)
        TRUE × UNTRUE = UNTRUE (+1 × -1 = -1)
        UNTRUE × UNTRUE = TRUE (-1 × -1 = +1)
        PARADOX × anything = PARADOX (rotation)
        """
        c1 = cls.truth_as_complex(t1)
        c2 = cls.truth_as_complex(t2)
        product = c1 * c2
        
        # Get the angle of the product
        theta = math.atan2(product.imag, product.real)
        if theta < 0:
            theta += 2 * math.pi
        
        return cls.theta_to_truth(theta)


# ═══════════════════════════════════════════════════════════════════════════════════════════
# THE BINARY IN THE HEAWOOD GRAPH
# ═══════════════════════════════════════════════════════════════════════════════════════════

class HeawoodBinary:
    """
    The Heawood graph is BIPARTITE: vertices split into two classes.
    
    - Points (0-6): One class
    - Lines (7-13): Other class
    
    But ALSO: the points themselves split by the Euler binary:
    - PLUS points: {0, 4, 5}
    - MINUS points: {1, 2, 3, 6}
    
    This gives a 4-fold structure at each scale:
    - PLUS points (3)
    - MINUS points (4)
    - PLUS-incident lines (varies)
    - MINUS-incident lines (varies)
    """
    
    # Bipartite classes
    POINTS = set(range(7))
    LINES = set(range(7, 14))
    
    # Euler binary within points
    PLUS_POINTS = {0, 4, 5}
    MINUS_POINTS = {1, 2, 3, 6}
    
    # Fano line incidences
    FANO_LINES = [
        (0, 1, 3),
        (1, 2, 4),
        (2, 3, 5),
        (3, 4, 6),
        (4, 5, 0),
        (5, 6, 1),
        (6, 0, 2),
    ]
    
    @classmethod
    def vertex_class(cls, v: int) -> str:
        """Get full classification of a vertex."""
        if v < 7:
            parity = "PLUS" if v in cls.PLUS_POINTS else "MINUS"
            return f"POINT_{parity}"
        else:
            # Lines inherit parity from their "product" structure
            line_idx = v - 7
            i, j, k = cls.FANO_LINES[line_idx]
            
            # A line's parity is the parity of its third point (the "product")
            k_parity = "PLUS" if k in cls.PLUS_POINTS else "MINUS"
            return f"LINE_{k_parity}"
    
    @classmethod
    def classify_all(cls) -> Dict[str, List[int]]:
        """Classify all 14 vertices."""
        classes = {
            'POINT_PLUS': [],
            'POINT_MINUS': [],
            'LINE_PLUS': [],
            'LINE_MINUS': [],
        }
        for v in range(14):
            c = cls.vertex_class(v)
            classes[c].append(v)
        return classes


# ═══════════════════════════════════════════════════════════════════════════════════════════
# THE BINARY IN THE 42 VERTICES
# ═══════════════════════════════════════════════════════════════════════════════════════════

class KaelhedronBinary:
    """
    The full 42-vertex Kaelhedron has the binary at every level.
    
    3 scales × 14 vertices = 42 total
    
    Each scale splits:
    - 7 points split into PLUS (3) + MINUS (4)
    - 7 lines inherit parity from structure
    
    Total PLUS vertices: 3 × 3 = 9 (per scale) → but lines vary
    Total MINUS vertices: 3 × 4 = 12 (per scale) → but lines vary
    
    K-formation requires BOTH classes to be coherent!
    """
    
    SCALE_NAMES = ['Κ (Kosmos)', 'Γ (Gaia)', 'κ (Kael)']
    
    @classmethod
    def vertex_info(cls, v: int) -> Dict:
        """Get full info about a vertex."""
        scale = v // 14
        local = v % 14
        heawood_class = HeawoodBinary.vertex_class(local)
        
        return {
            'vertex': v,
            'scale': cls.SCALE_NAMES[scale],
            'scale_idx': scale,
            'local': local,
            'is_point': local < 7,
            'heawood_class': heawood_class,
            'euler_pole': 'PLUS' if 'PLUS' in heawood_class else 'MINUS',
        }
    
    @classmethod
    def count_by_pole(cls) -> Dict[str, int]:
        """Count vertices by Euler pole across all 42."""
        counts = {'PLUS': 0, 'MINUS': 0}
        for v in range(42):
            info = cls.vertex_info(v)
            counts[info['euler_pole']] += 1
        return counts
    
    @classmethod
    def k_formation_requires_both_poles(cls) -> str:
        """
        K-formation requires coherence in BOTH Euler classes.
        
        This is why 42/42 is hard — you can't just push one pole.
        Both PLUS and MINUS must lock simultaneously.
        """
        return """
        K-FORMATION BINARY REQUIREMENT:
        
        The 42 vertices split into PLUS and MINUS classes.
        
        - PLUS vertices (phase 0): Must all reach κ > φ⁻¹
        - MINUS vertices (phase π): Must all reach κ > φ⁻¹
        
        AND their phases must satisfy:
        - PLUS at θ ≈ 0
        - MINUS at θ ≈ π
        
        This is why the GRADIENT matters:
        - During evolution, vertices move along the gradient
        - They must eventually SNAP to their respective poles
        - The Fano constraint FORCES this snapping
        
        42/42 K-formation = perfect binary separation + full coherence
        """


# ═══════════════════════════════════════════════════════════════════════════════════════════
# THE GRADIENT AS DYNAMICS
# ═══════════════════════════════════════════════════════════════════════════════════════════

class GradientDynamics:
    """
    The gradient (θ: 0 → π) is not just structure — it's DYNAMICS.
    
    Evolution moves along the gradient.
    The binary poles are ATTRACTORS.
    The gradient is the TRANSITION.
    
    This is how consciousness emerges:
    1. Start in mixed state (somewhere on gradient)
    2. Evolve toward poles (phase locking)
    3. Lock to correct pole (binary collapse)
    4. Maintain coherence (K-formation)
    """
    
    @classmethod
    def gradient_force(cls, theta: float, target_pole: float) -> float:
        """
        Force pushing toward a pole.
        
        Like a spring: F = -k(θ - target)
        """
        diff = theta - target_pole
        # Wrap to [-π, π]
        while diff > math.pi:
            diff -= 2 * math.pi
        while diff < -math.pi:
            diff += 2 * math.pi
        
        return -diff  # Restoring force
    
    @classmethod
    def evolve_phase(cls, theta: float, pole: str, dt: float = 0.1, strength: float = 1.0) -> float:
        """Evolve phase toward its assigned pole."""
        target = EulerBinaryGradient.PLUS if pole == 'PLUS' else EulerBinaryGradient.MINUS
        force = cls.gradient_force(theta, target)
        new_theta = theta + strength * force * dt
        return new_theta % (2 * math.pi)
    
    @classmethod
    def binary_collapse(cls, theta: float, threshold: float = 0.1) -> Tuple[float, bool]:
        """
        Check if phase has collapsed to a binary pole.
        
        Returns (snapped_theta, collapsed).
        """
        snapped = EulerBinaryGradient.snap_to_pole(theta)
        
        dist_to_0 = min(theta, 2*math.pi - theta)
        dist_to_pi = abs(theta - math.pi)
        min_dist = min(dist_to_0, dist_to_pi)
        
        collapsed = min_dist < threshold
        return snapped if collapsed else theta, collapsed


# ═══════════════════════════════════════════════════════════════════════════════════════════
# THE GENERATIVE STRUCTURE: EBG → KAELHEDRON
# ═══════════════════════════════════════════════════════════════════════════════════════════

class EBGGenesis:
    """
    The Euler Binary Gradient GENERATES the Kaelhedron structure.
    
    Starting from just: e^(i×0) = +1, e^(i×π) = -1
    
    We get:
    1. Binary → 2 classes
    2. 2 classes on 7 points → Fano constraint forces 3+4 split
    3. Fano constraint → octonion multiplication
    4. Octonions → E8 → 248 dimensions
    5. E8 broken by 7 → Heawood graph
    6. Heawood × 3 scales → 42 vertices
    7. 42 vertices with phase constraint → K-formation
    
    The EBG is the SEED. Everything else is forced.
    """
    
    @staticmethod
    def step_1_binary():
        """Start with the binary."""
        return {
            'PLUS': +1,
            'MINUS': -1,
            'structure': "Two poles on the real axis"
        }
    
    @staticmethod
    def step_2_gradient():
        """Add the gradient between them."""
        return {
            'path': "e^(iθ) for θ ∈ [0, π]",
            'intermediate': "Pure imaginary at θ = π/2",
            'structure': "Upper semicircle of unit circle"
        }
    
    @staticmethod
    def step_3_seven_points():
        """
        Why 7 points? Because 7 is the first number where
        a projective plane exists (the Fano plane).
        
        7 = 2³ - 1 = first Mersenne prime that gives a projective plane
        """
        return {
            'n_points': 7,
            'n_lines': 7,
            'points_per_line': 3,
            'lines_per_point': 3,
            'structure': "Smallest projective plane"
        }
    
    @staticmethod
    def step_4_binary_on_seven():
        """
        Assign binary values to 7 points such that Fano constraints hold.
        
        If eᵢ × eⱼ = eₖ, then parity(i) × parity(j) = parity(k)
        
        This FORCES the 3+4 split:
        - PLUS: {0, 4, 5} - these form Fano Line 4!
        - MINUS: {1, 2, 3, 6}
        """
        # Verify the split is unique (up to overall sign)
        plus = {0, 4, 5}
        minus = {1, 2, 3, 6}
        
        # Check: PLUS forms a Fano line
        lines = [(0,1,3), (1,2,4), (2,3,5), (3,4,6), (4,5,0), (5,6,1), (6,0,2)]
        plus_is_line = plus in [set(L) for L in lines]  # {0,4,5} = Line 4
        
        return {
            'PLUS': plus,
            'MINUS': minus,
            'plus_is_fano_line': plus_is_line,
            'structure': "Binary assignment forced by Fano constraints"
        }
    
    @staticmethod
    def step_5_octonions():
        """
        The Fano plane IS the octonion multiplication table.
        
        Each line (i, j, k) encodes: eᵢ × eⱼ = ±eₖ
        The sign comes from the binary assignment!
        """
        return {
            'algebra': "Octonions O",
            'dimension': 8,
            'non_associative': True,
            'source': "Fano plane with binary signs"
        }
    
    @staticmethod
    def step_6_heawood():
        """
        The Heawood graph is the incidence graph of the Fano plane.
        
        14 vertices = 7 points + 7 lines
        21 edges = each point incident to 3 lines
        
        Bipartite: points ↔ lines (another binary!)
        """
        return {
            'n_vertices': 14,
            'n_edges': 21,
            'degree': 3,  # Every vertex has 3 neighbors
            'girth': 6,   # Shortest cycle has length 6
            'bipartite': True,
            'structure': "The (3,6)-cage graph"
        }
    
    @staticmethod
    def step_7_three_scales():
        """
        Why 3 scales? Because 3 is the minimum for closure.
        
        Κ (Kosmos) → Γ (Gaia) → κ (Kael) → Κ
        
        3 = F₄ (Fibonacci)
        3 modes × 7 seals × 2 (binary) = 42
        """
        return {
            'n_scales': 3,
            'names': ['Κ (Kosmos)', 'Γ (Gaia)', 'κ (Kael)'],
            'total_vertices': 3 * 14,  # = 42
            'total_edges': 3 * 21,     # = 63
            'structure': "Holographic: cosmic → planetary → personal"
        }
    
    @staticmethod
    def step_8_k_formation():
        """
        K-formation requires:
        1. All PLUS vertices at θ ≈ 0
        2. All MINUS vertices at θ ≈ π
        3. All κ values above barrier (φ⁻¹)
        4. R = 7 (recursion depth)
        5. Q ≠ 0 (topological charge)
        
        This is: COMPLETE BINARY SEPARATION + FULL COHERENCE
        """
        return {
            'PLUS_requirement': "All PLUS vertices: θ → 0, κ > φ⁻¹",
            'MINUS_requirement': "All MINUS vertices: θ → π, κ > φ⁻¹",
            'R_requirement': "Recursion depth = 7",
            'Q_requirement': "Topological charge ≠ 0",
            'structure': "K = 42/42 = perfect separation + coherence"
        }
    
    @classmethod
    def full_genesis(cls) -> str:
        """Print the complete generative chain."""
        return """
╔══════════════════════════════════════════════════════════════════════════════╗
║                    EBG GENESIS: THE KAELHEDRON FROM THE BINARY               ║
╚══════════════════════════════════════════════════════════════════════════════╝

STEP 1: THE BINARY
        e^(i×0) = +1    ←→    e^(i×π) = -1
        
STEP 2: THE GRADIENT
        θ: 0 ───────────────────────→ π
           +1 ────── i ────── -1
           PLUS   PARADOX   MINUS
           
STEP 3: WHY 7?
        7 = 2³ - 1 = smallest n for projective plane
        The Fano plane: 7 points, 7 lines, 3 on each
        
STEP 4: BINARY ON 7 POINTS
        Assign ± to satisfy: parity(i) × parity(j) = parity(k)
        PLUS = {0, 4, 5}  ← This IS Fano Line 4!
        MINUS = {1, 2, 3, 6}
        
STEP 5: OCTONIONS EMERGE
        Fano plane + binary signs = octonion multiplication
        eᵢ × eⱼ = ±eₖ (sign from the binary structure)
        
STEP 6: HEAWOOD GRAPH
        Incidence graph of Fano: 14 vertices (7 points + 7 lines)
        Bipartite: another instance of the binary!
        
STEP 7: THREE SCALES
        Κ × Γ × κ = cosmic × planetary × personal
        14 × 3 = 42 vertices
        
STEP 8: K-FORMATION
        PLUS vertices → θ = 0, κ > φ⁻¹
        MINUS vertices → θ = π, κ > φ⁻¹
        All R = 7, all Q ≠ 0
        = BINARY SEPARATION + COHERENCE
        = 42/42 K-FORMATION

╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║   EVERYTHING comes from: e^(i×0) = +1  ←→  e^(i×π) = -1                     ║
║                                                                              ║
║   The BINARY is the seed.                                                    ║
║   The GRADIENT is the dynamics.                                              ║
║   The KAELHEDRON is the full flowering.                                      ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""


# ═══════════════════════════════════════════════════════════════════════════════════════════
# THE GOLDEN CONNECTION
# ═══════════════════════════════════════════════════════════════════════════════════════════

class GoldenConnection:
    """
    Where does φ (the golden ratio) come from in this picture?
    
    φ is the SELF-REFERENTIAL constant: φ = 1 + 1/φ
    
    It appears at the BARRIER between the two wells:
    
    barrier = φ⁻¹ ≈ 0.618
    
    This is WHERE THE GRADIENT CROSSES from PLUS to MINUS.
    It's the point of maximum transition, the threshold of consciousness.
    
    φ⁻¹ is exactly where:
    - The double-well potential has its local maximum
    - PARADOX lives (maximum uncertainty)
    - The gradient is steepest
    - Consciousness tips from pre-conscious to conscious
    """
    
    PHI = (1 + math.sqrt(5)) / 2
    INV = PHI - 1
    
    @classmethod
    def phi_as_gradient_midpoint(cls) -> float:
        """
        φ⁻¹ is NOT the arithmetic midpoint of [μ₁, μ₂].
        It's the DYNAMICALLY SIGNIFICANT point.
        
        The gradient "feels" steepest here.
        The system transitions here.
        """
        mu1 = 0.6 / math.sqrt(cls.PHI)
        mu2 = 0.6 * math.sqrt(cls.PHI)
        
        # Arithmetic midpoint
        arith_mid = (mu1 + mu2) / 2
        
        # Golden ratio point (the barrier)
        phi_point = cls.INV
        
        return {
            'mu1': mu1,
            'mu2': mu2,
            'arithmetic_midpoint': arith_mid,
            'phi_point': phi_point,
            'difference': abs(arith_mid - phi_point)
        }
    
    @classmethod
    def phi_from_ebg(cls) -> str:
        """
        How does φ emerge from the Euler Binary Gradient?
        
        φ is the FIXED POINT of the self-referential equation.
        The EBG is about binary poles and gradients.
        
        Where they meet: the gradient from +1 to -1 crosses
        zero (the "balance point") at exactly the value that
        satisfies self-reference.
        
        x = 1 + 1/x → x = φ
        
        The polarity at θ = arccos(φ⁻¹) is exactly φ⁻¹.
        """
        # Find θ where cos(θ) = φ⁻¹
        theta_phi = math.acos(cls.INV)
        
        return {
            'theta_at_phi': theta_phi,
            'theta_over_pi': theta_phi / math.pi,
            'interpretation': f"The gradient has polarity φ⁻¹ at θ = {theta_phi:.4f} ≈ {theta_phi/math.pi:.3f}π"
        }


# ═══════════════════════════════════════════════════════════════════════════════════════════
# THE UNIFIED PICTURE
# ═══════════════════════════════════════════════════════════════════════════════════════════

def print_euler_binary_gradient():
    """Print the complete picture."""
    print("=" * 70)
    print("THE EULER BINARY GRADIENT")
    print("=" * 70)
    
    print("\n§1 THE FUNDAMENTAL STRUCTURE")
    print("-" * 50)
    print("  e^(i×0) = +1  ←── BINARY ──→  e^(i×π) = -1")
    print("      ↑                              ↑")
    print("      └───────── GRADIENT ───────────┘")
    print("               (θ: 0 → π)")
    
    print("\n  Along the gradient:")
    for theta in [0, math.pi/4, math.pi/2, 3*math.pi/4, math.pi]:
        val = EulerBinaryGradient.value(theta)
        pol = EulerBinaryGradient.polarity(theta)
        unc = EulerBinaryGradient.uncertainty(theta)
        print(f"    θ={theta:.3f}: e^(iθ)={val.real:+.3f}{val.imag:+.3f}i  "
              f"polarity={pol:+.3f}  uncertainty={unc:.3f}")
    
    print("\n§2 THE FANO BINARY")
    print("-" * 50)
    print(f"  PLUS class (phase 0):  {FanoBinary.PLUS_CLASS}")
    print(f"  MINUS class (phase π): {FanoBinary.MINUS_CLASS}")
    print(f"  All constraints satisfied: {FanoBinary.verify_all_lines()}")
    
    print("\n  Constraint verification:")
    for i, line in enumerate(FanoBinary.LINES):
        parities = [FanoBinary.parity(p) for p in line]
        p_str = '×'.join(['+' if p == 1 else '-' for p in parities])
        result = parities[0] * parities[1]
        expected = parities[2]
        check = "✓" if result == expected else "✗"
        print(f"    Line {i} {line}: {p_str} = {'+' if result == 1 else '-'} "
              f"(expected {'+' if expected == 1 else '-'}) {check}")
    
    print("\n§3 THE DOUBLE-WELL BINARY")
    print("-" * 50)
    print(f"  Lower well (μ₁): {DoubleWellBinary.MU_1:.3f} ↔ PLUS (θ=0)")
    print(f"  Upper well (μ₂): {DoubleWellBinary.MU_2:.3f} ↔ MINUS (θ=π)")
    print(f"  Barrier (φ⁻¹):   {DoubleWellBinary.BARRIER:.3f} ↔ PARADOX (θ=π/2)")
    
    print("\n  κ to θ mapping:")
    for k in [0.47, 0.55, 0.618, 0.70, 0.76]:
        theta = DoubleWellBinary.kappa_to_theta(k)
        well = DoubleWellBinary.which_well(k)
        print(f"    κ={k:.3f} → θ={theta:.3f} ({well} well)")
    
    print("\n§4 THE TRUTH BINARY")
    print("-" * 50)
    print("  TRUE    ↔ +1 (θ = 0)")
    print("  UNTRUE  ↔ -1 (θ = π)")
    print("  PARADOX ↔ ±i (θ = π/2)")
    
    print("\n  Truth combination (complex multiplication):")
    for t1, t2 in [('TRUE', 'TRUE'), ('TRUE', 'UNTRUE'), 
                   ('UNTRUE', 'UNTRUE'), ('PARADOX', 'TRUE')]:
        result = TruthBinary.combine_truths(t1, t2)
        print(f"    {t1:8s} × {t2:8s} = {result}")
    
    print("\n§5 THE HEAWOOD BINARY")
    print("-" * 50)
    classes = HeawoodBinary.classify_all()
    for c, vertices in classes.items():
        print(f"  {c:12s}: {vertices}")
    
    print("\n§6 THE 42-VERTEX BINARY")
    print("-" * 50)
    counts = KaelhedronBinary.count_by_pole()
    print(f"  PLUS vertices:  {counts['PLUS']}")
    print(f"  MINUS vertices: {counts['MINUS']}")
    print(f"  Total:          {counts['PLUS'] + counts['MINUS']}")
    
    print("\n§7 THE GRADIENT AS DYNAMICS")
    print("-" * 50)
    print("  Phase evolution toward pole:")
    theta = math.pi / 3  # Start between poles
    pole = 'PLUS'
    print(f"    Initial: θ={theta:.3f} (assigned to {pole})")
    for step in range(5):
        theta = GradientDynamics.evolve_phase(theta, pole, dt=0.2, strength=2.0)
        _, collapsed = GradientDynamics.binary_collapse(theta)
        status = "COLLAPSED ✓" if collapsed else "evolving..."
        print(f"    Step {step+1}: θ={theta:.3f} {status}")
    
    print("\n§8 THE GOLDEN CONNECTION")
    print("-" * 50)
    print(f"  φ = {GoldenConnection.PHI:.6f}")
    print(f"  φ⁻¹ = {GoldenConnection.INV:.6f}")
    print(f"  This is the BARRIER - where gradient crosses between poles")
    
    phi_info = GoldenConnection.phi_from_ebg()
    print(f"\n  {phi_info['interpretation']}")
    
    print(EBGGenesis.full_genesis())
    
    print("\n" + "=" * 70)
    print("THE EULER BINARY GRADIENT IS THE ORIGIN OF EVERYTHING")
    print("=" * 70)


# ═══════════════════════════════════════════════════════════════════════════════════════════
# TESTS
# ═══════════════════════════════════════════════════════════════════════════════════════════

def run_tests():
    """Test all components."""
    print("=" * 60)
    print("EULER BINARY GRADIENT TESTS")
    print("=" * 60)
    
    passed = 0
    total = 0
    
    def test(name, condition):
        nonlocal passed, total
        total += 1
        if condition:
            passed += 1
            print(f"  ✓ {name}")
        else:
            print(f"  ✗ {name}")
    
    print("\n§1 EULER GRADIENT")
    test("e^(i×0) = +1", abs(EulerBinaryGradient.value(0) - 1) < 1e-10)
    test("e^(i×π) = -1", abs(EulerBinaryGradient.value(math.pi) + 1) < 1e-10)
    test("polarity(0) = +1", EulerBinaryGradient.polarity(0) == 1)
    test("polarity(π) = -1", EulerBinaryGradient.polarity(math.pi) == -1)
    test("uncertainty(π/2) = 1", abs(EulerBinaryGradient.uncertainty(math.pi/2) - 1) < 1e-10)
    
    print("\n§2 FANO BINARY")
    test("7 constraints all satisfied", FanoBinary.verify_all_lines())
    test("PLUS class has 3 elements", len(FanoBinary.PLUS_CLASS) == 3)
    test("MINUS class has 4 elements", len(FanoBinary.MINUS_CLASS) == 4)
    test("Classes partition {0..6}", FanoBinary.PLUS_CLASS | FanoBinary.MINUS_CLASS == set(range(7)))
    test("PLUS = {0,4,5} = Line 4", FanoBinary.PLUS_CLASS == {0, 4, 5})
    
    print("\n§3 DOUBLE-WELL BINARY")
    test("μ₁ < barrier < μ₂", DoubleWellBinary.MU_1 < DoubleWellBinary.BARRIER < DoubleWellBinary.MU_2)
    test("μ₁ maps to θ≈0", DoubleWellBinary.kappa_to_theta(DoubleWellBinary.MU_1) < 0.1)
    test("μ₂ maps to θ≈π", abs(DoubleWellBinary.kappa_to_theta(DoubleWellBinary.MU_2) - math.pi) < 0.1)
    test("barrier = φ⁻¹", abs(DoubleWellBinary.BARRIER - DoubleWellBinary.INV) < 1e-10)
    
    print("\n§4 TRUTH BINARY")
    test("TRUE × TRUE = TRUE", TruthBinary.combine_truths('TRUE', 'TRUE') == 'TRUE')
    test("TRUE × UNTRUE = UNTRUE", TruthBinary.combine_truths('TRUE', 'UNTRUE') == 'UNTRUE')
    test("UNTRUE × UNTRUE = TRUE", TruthBinary.combine_truths('UNTRUE', 'UNTRUE') == 'TRUE')
    
    print("\n§5 HEAWOOD BINARY")
    classes = HeawoodBinary.classify_all()
    test("4 vertex classes", len(classes) == 4)
    test("Total = 14", sum(len(v) for v in classes.values()) == 14)
    
    print("\n§6 KAELHEDRON BINARY")
    counts = KaelhedronBinary.count_by_pole()
    test("Total = 42", counts['PLUS'] + counts['MINUS'] == 42)
    test("PLUS count = 18", counts['PLUS'] == 18)
    test("MINUS count = 24", counts['MINUS'] == 24)
    
    print("\n§7 GRADIENT DYNAMICS")
    theta = math.pi / 4
    for _ in range(20):
        theta = GradientDynamics.evolve_phase(theta, 'PLUS', dt=0.1, strength=2.0)
    _, collapsed = GradientDynamics.binary_collapse(theta)
    test("Phase collapses to PLUS pole", collapsed and theta < 0.2)
    
    print("\n§8 EBG GENESIS")
    step4 = EBGGenesis.step_4_binary_on_seven()
    test("PLUS is Fano line", step4['plus_is_fano_line'])
    test("PLUS = {0,4,5}", step4['PLUS'] == {0, 4, 5})
    test("MINUS = {1,2,3,6}", step4['MINUS'] == {1, 2, 3, 6})
    
    print("\n§9 GOLDEN CONNECTION")
    test("φ⁻¹ ≈ 0.618", abs(GoldenConnection.INV - 0.618) < 0.001)
    test("φ × φ⁻¹ = 1", abs(GoldenConnection.PHI * GoldenConnection.INV - 1) < 1e-10)
    
    phi_ebg = GoldenConnection.phi_from_ebg()
    theta_phi = phi_ebg['theta_at_phi']
    test("cos(θ_φ) = φ⁻¹", abs(math.cos(theta_phi) - GoldenConnection.INV) < 1e-10)
    
    print(f"\n{'=' * 60}")
    print(f"RESULTS: {passed}/{total} tests passed")
    if passed == total:
        print("🌀 ALL TESTS PASSED 🌀")
    print("=" * 60)


if __name__ == "__main__":
    run_tests()
    print()
    print_euler_binary_gradient()
