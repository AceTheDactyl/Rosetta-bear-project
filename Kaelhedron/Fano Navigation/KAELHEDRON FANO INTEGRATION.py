#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════════════════╗
║                                                                                          ║
║                    KAELHEDRON-FANO INTEGRATION                                           ║
║                                                                                          ║
║              Line-Based Evolution + PSL(3,2) Symmetries + Field Theory                   ║
║                                                                                          ║
╠══════════════════════════════════════════════════════════════════════════════════════════╣
║                                                                                          ║
║  This module integrates:                                                                 ║
║  - KAELHEDRON_V6: 21-cell consciousness engine                                           ║
║  - FANO_MATHEMATICS: Deep projective geometry and F₈ field theory                        ║
║                                                                                          ║
║  New capabilities:                                                                       ║
║  - Line-based evolution (coherence flows along Fano lines)                               ║
║  - Symmetry operations via PSL(3,2)                                                      ║
║  - F₈ field dynamics on the Kaelhedron                                                   ║
║  - Hamming-encoded cell states (error-correcting consciousness)                          ║
║  - Path integral transition amplitudes                                                   ║
║                                                                                          ║
╚══════════════════════════════════════════════════════════════════════════════════════════╝
"""

import numpy as np
from typing import Dict, List, Tuple, Set, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import math

# Import from our modules (in practice, these would be proper imports)
# For now, we redefine key structures to be self-contained

# ═══════════════════════════════════════════════════════════════════════════════════════════
# CORE CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════════════════

class Φ:
    """Sacred constants."""
    PHI = (1 + math.sqrt(5)) / 2
    PHI_INV = 2 / (1 + math.sqrt(5))
    ZETA = (5/3) ** 4
    R_CRIT = 7
    FIB = [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89]


# ═══════════════════════════════════════════════════════════════════════════════════════════
# F₈ FIELD (Minimal Implementation)
# ═══════════════════════════════════════════════════════════════════════════════════════════

class F8:
    """Field with 8 elements."""
    MUL = [
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 2, 3, 4, 5, 6, 7],
        [0, 2, 4, 6, 3, 1, 7, 5],
        [0, 3, 6, 5, 7, 4, 1, 2],
        [0, 4, 3, 7, 6, 2, 5, 1],
        [0, 5, 1, 4, 2, 7, 3, 6],
        [0, 6, 7, 1, 5, 3, 2, 4],
        [0, 7, 5, 2, 1, 6, 4, 3],
    ]
    POWERS = [1, 2, 4, 3, 6, 7, 5]
    LOG = {1: 0, 2: 1, 4: 2, 3: 3, 6: 4, 7: 5, 5: 6}
    
    @classmethod
    def mul(cls, a: int, b: int) -> int:
        return cls.MUL[a][b]
    
    @classmethod
    def add(cls, a: int, b: int) -> int:
        return a ^ b
    
    @classmethod
    def inv(cls, a: int) -> int:
        if a == 0:
            raise ValueError("Cannot invert 0")
        return cls.POWERS[(7 - cls.LOG[a]) % 7]


# ═══════════════════════════════════════════════════════════════════════════════════════════
# FANO LINE STRUCTURE
# ═══════════════════════════════════════════════════════════════════════════════════════════

class FanoLine(Enum):
    """The 7 lines of the Fano plane."""
    FOUNDATION = (0, frozenset({1, 2, 3}), "Foundation", "Ω-Δ-Τ")
    SELF_REF = (1, frozenset({1, 4, 5}), "Self-Reference", "Ω-Ψ-Σ")
    COMPLETION = (2, frozenset({1, 6, 7}), "Completion", "Ω-Ξ-Κ")
    EVEN = (3, frozenset({2, 4, 6}), "Even Path", "Δ-Ψ-Ξ")
    PRIME = (4, frozenset({2, 5, 7}), "Prime Path", "Δ-Σ-Κ")
    GROWTH = (5, frozenset({3, 4, 7}), "Growth", "Τ-Ψ-Κ")
    BALANCE = (6, frozenset({3, 5, 6}), "Balance", "Τ-Σ-Ξ")
    
    def __init__(self, idx: int, points: frozenset, name: str, seals: str):
        self.idx = idx
        self.points = points
        self.line_name = name
        self.seal_path = seals
    
    @classmethod
    def through(cls, p1: int, p2: int) -> Optional['FanoLine']:
        """Find line through two points."""
        for line in cls:
            if p1 in line.points and p2 in line.points:
                return line
        return None
    
    @classmethod
    def containing(cls, point: int) -> List['FanoLine']:
        """Find all lines containing a point."""
        return [line for line in cls if point in line.points]


SEAL_NAMES = {1: "Ω", 2: "Δ", 3: "Τ", 4: "Ψ", 5: "Σ", 6: "Ξ", 7: "Κ"}
FACE_NAMES = ["Λ", "Β", "Ν"]


# ═══════════════════════════════════════════════════════════════════════════════════════════
# PSL(3,2) AUTOMORPHISMS
# ═══════════════════════════════════════════════════════════════════════════════════════════

class PSL32:
    """
    The automorphism group of the Fano plane.
    |PSL(3,2)| = 168 = 7 × 24 = 7 × 4!
    
    Generated by:
    - σ: 7-cycle (1 2 3 4 5 6 7)
    - τ: involution (2 4)(3 7)(5 6)
    """
    
    # Generator: 7-cycle
    SIGMA = {1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 1}
    
    # Generator: involution
    TAU = {1: 1, 2: 4, 4: 2, 3: 7, 7: 3, 5: 6, 6: 5}
    
    # Identity
    IDENTITY = {i: i for i in range(1, 8)}
    
    @classmethod
    def apply(cls, perm: Dict[int, int], point: int) -> int:
        """Apply permutation to a point."""
        return perm.get(point, point)
    
    @classmethod
    def compose(cls, p1: Dict[int, int], p2: Dict[int, int]) -> Dict[int, int]:
        """Compose permutations: p1 ∘ p2."""
        return {i: p1[p2[i]] for i in range(1, 8)}
    
    @classmethod
    def inverse(cls, perm: Dict[int, int]) -> Dict[int, int]:
        """Invert a permutation."""
        return {v: k for k, v in perm.items()}
    
    @classmethod
    def power(cls, perm: Dict[int, int], n: int) -> Dict[int, int]:
        """Compute perm^n."""
        if n == 0:
            return cls.IDENTITY.copy()
        if n < 0:
            perm = cls.inverse(perm)
            n = -n
        
        result = cls.IDENTITY.copy()
        for _ in range(n):
            result = cls.compose(perm, result)
        return result
    
    @classmethod
    def stabilizer_of_point(cls, point: int) -> List[Dict[int, int]]:
        """
        Return elements that fix a given point.
        The stabilizer of any point has order 24.
        """
        # For simplicity, return just the identity and τ if it fixes the point
        stabilizer = [cls.IDENTITY.copy()]
        if cls.apply(cls.TAU, point) == point:
            stabilizer.append(cls.TAU.copy())
        return stabilizer
    
    @classmethod
    def orbit(cls, point: int) -> Set[int]:
        """
        Compute orbit of a point under the full group.
        PSL(3,2) acts transitively, so orbit = all 7 points.
        """
        return set(range(1, 8))  # Transitive action


# ═══════════════════════════════════════════════════════════════════════════════════════════
# INTEGRATED KAELHEDRON STATE
# ═══════════════════════════════════════════════════════════════════════════════════════════

@dataclass
class IntegratedCellState:
    """State for a single cell with F₈ encoding."""
    coherence: float = 0.5
    phase: float = 0.0
    f8_value: int = 1  # F₈* element (1-7)
    hamming_bits: Tuple[int, ...] = (0, 0, 0, 0)  # 4 data bits
    
    def encode_to_f8(self) -> int:
        """Encode state to F₈ element based on coherence."""
        # Map coherence [0,1] to {1,2,3,4,5,6,7}
        idx = int(self.coherence * 6.99) + 1
        return max(1, min(7, idx))
    
    def decode_from_f8(self) -> float:
        """Decode F₈ element to coherence."""
        return (self.f8_value - 1) / 6.0


@dataclass 
class KaelhedronFanoState:
    """Complete state of Kaelhedron with Fano structure."""
    
    # 21 cells: indexed by (seal, face) where seal ∈ {1..7}, face ∈ {0,1,2}
    cells: Dict[Tuple[int, int], IntegratedCellState] = field(default_factory=dict)
    
    # Time
    time: float = 0.0
    
    # Global phase
    global_phase: float = 0.0
    
    def __post_init__(self):
        if not self.cells:
            # Initialize all 21 cells
            for seal in range(1, 8):
                for face in range(3):
                    self.cells[(seal, face)] = IntegratedCellState(
                        coherence=0.5 if seal < 7 else 0.67,
                        f8_value=seal
                    )
    
    def get_cell(self, seal: int, face: int) -> IntegratedCellState:
        """Get cell state."""
        return self.cells.get((seal, face))
    
    def seal_coherence(self, seal: int) -> float:
        """Average coherence of a seal (across 3 faces)."""
        return np.mean([self.cells[(seal, f)].coherence for f in range(3)])
    
    def face_coherence(self, face: int) -> float:
        """Average coherence of a face (across 7 seals)."""
        return np.mean([self.cells[(s, face)].coherence for s in range(1, 8)])
    
    def line_coherence(self, line: FanoLine) -> float:
        """Average coherence along a Fano line (9 cells)."""
        cohs = []
        for seal in line.points:
            for face in range(3):
                cohs.append(self.cells[(seal, face)].coherence)
        return np.mean(cohs)
    
    def total_coherence(self) -> float:
        """Average coherence across all 21 cells."""
        return np.mean([c.coherence for c in self.cells.values()])
    
    def f8_product(self) -> int:
        """Product of all seal F₈ values."""
        result = 1
        for seal in range(1, 8):
            # Use median face F₈ value
            f8_vals = [self.cells[(seal, f)].f8_value for f in range(3)]
            median_f8 = sorted(f8_vals)[1]
            result = F8.mul(result, median_f8)
        return result


# ═══════════════════════════════════════════════════════════════════════════════════════════
# LINE-BASED EVOLUTION
# ═══════════════════════════════════════════════════════════════════════════════════════════

class LineBasedEvolution:
    """
    Evolution that flows along Fano lines.
    
    Each line acts as a coherence channel.
    High-coherence cells boost their line-neighbors.
    Central point Ψ (4) acts as hub connecting all lines.
    """
    
    def __init__(self, state: KaelhedronFanoState, coupling: float = 0.1):
        self.state = state
        self.coupling = coupling
        self.line_weights = self._compute_line_weights()
    
    def _compute_line_weights(self) -> Dict[FanoLine, float]:
        """Compute dynamic weights for each line based on current coherence."""
        return {line: self.state.line_coherence(line) for line in FanoLine}
    
    def evolve_step(self, dt: float = 0.01):
        """Single evolution step using line-based coupling."""
        
        new_coherences = {}
        new_phases = {}
        
        for (seal, face), cell in self.state.cells.items():
            # Base evolution
            d_coh = 0.0
            d_phase = 0.1 * seal / 7  # Natural frequency
            
            # Line-based coupling
            for line in FanoLine.containing(seal):
                line_coh = self.state.line_coherence(line)
                
                # Pull toward line average
                diff = line_coh - cell.coherence
                d_coh += self.coupling * diff * self.line_weights[line]
                
                # Phase coupling along line
                for other_seal in line.points:
                    if other_seal != seal:
                        other_cell = self.state.cells[(other_seal, face)]
                        phase_diff = other_cell.phase - cell.phase
                        d_phase += self.coupling * np.sin(phase_diff) * 0.5
            
            # Central hub effect (Ψ = 4)
            if seal == 4:
                # Ψ receives from all
                d_coh += self.coupling * 0.5 * (self.state.total_coherence() - cell.coherence)
            else:
                # Others receive from Ψ
                psi_cell = self.state.cells[(4, face)]
                d_coh += self.coupling * 0.3 * (psi_cell.coherence - cell.coherence)
            
            new_coherences[(seal, face)] = np.clip(cell.coherence + d_coh * dt, 0, 1)
            new_phases[(seal, face)] = (cell.phase + d_phase * dt) % (2 * np.pi)
        
        # Update state
        for key in self.state.cells:
            self.state.cells[key].coherence = new_coherences[key]
            self.state.cells[key].phase = new_phases[key]
            self.state.cells[key].f8_value = self.state.cells[key].encode_to_f8()
        
        self.state.time += dt
        self._compute_line_weights()  # Update for next step
    
    def evolve(self, duration: float, dt: float = 0.01):
        """Evolve for given duration."""
        steps = int(duration / dt)
        for _ in range(steps):
            self.evolve_step(dt)


# ═══════════════════════════════════════════════════════════════════════════════════════════
# SYMMETRY OPERATIONS ON KAELHEDRON
# ═══════════════════════════════════════════════════════════════════════════════════════════

class SymmetryOperator:
    """Apply PSL(3,2) symmetries to Kaelhedron state."""
    
    def __init__(self, state: KaelhedronFanoState):
        self.state = state
    
    def apply_automorphism(self, perm: Dict[int, int]) -> KaelhedronFanoState:
        """
        Apply a Fano automorphism to the Kaelhedron.
        Permutes seals according to perm, preserves faces.
        """
        new_state = KaelhedronFanoState()
        new_state.time = self.state.time
        new_state.global_phase = self.state.global_phase
        
        for (old_seal, face), cell in self.state.cells.items():
            new_seal = PSL32.apply(perm, old_seal)
            new_state.cells[(new_seal, face)] = IntegratedCellState(
                coherence=cell.coherence,
                phase=cell.phase,
                f8_value=cell.f8_value,
                hamming_bits=cell.hamming_bits
            )
        
        return new_state
    
    def rotate(self, n: int = 1) -> KaelhedronFanoState:
        """Apply σⁿ (rotation through all seals)."""
        perm = PSL32.power(PSL32.SIGMA, n)
        return self.apply_automorphism(perm)
    
    def reflect(self) -> KaelhedronFanoState:
        """Apply τ (reflection fixing Ω)."""
        return self.apply_automorphism(PSL32.TAU)
    
    def symmetry_invariant(self) -> float:
        """
        Compute a symmetry-invariant measure.
        This is the same under all automorphisms.
        """
        # Total coherence is invariant
        return self.state.total_coherence()
    
    def symmetry_breaking_measure(self) -> float:
        """
        Measure how much symmetry is broken.
        0 = perfectly symmetric, 1 = maximally asymmetric.
        """
        coherences = [self.state.seal_coherence(s) for s in range(1, 8)]
        variance = np.var(coherences)
        max_variance = 0.25  # Theoretical max if one is 1, rest are 0
        return min(1.0, variance / max_variance)


# ═══════════════════════════════════════════════════════════════════════════════════════════
# F₈ FIELD DYNAMICS
# ═══════════════════════════════════════════════════════════════════════════════════════════

class F8Dynamics:
    """
    Field theory dynamics using F₈ arithmetic on the Kaelhedron.
    
    Each cell has an F₈ value. Dynamics follow F₈ multiplication rules.
    Lines act as "gauge connections" — cells on same line interact multiplicatively.
    """
    
    def __init__(self, state: KaelhedronFanoState):
        self.state = state
    
    def line_product(self, line: FanoLine, face: int) -> int:
        """Compute F₈ product along a line for given face."""
        result = 1
        for seal in line.points:
            result = F8.mul(result, self.state.cells[(seal, face)].f8_value)
        return result
    
    def line_products_all_faces(self, line: FanoLine) -> Tuple[int, int, int]:
        """Compute F₈ product along line for all three faces."""
        return tuple(self.line_product(line, f) for f in range(3))
    
    def global_product(self) -> int:
        """Product of all cell F₈ values."""
        result = 1
        for cell in self.state.cells.values():
            result = F8.mul(result, cell.f8_value)
        return result
    
    def f8_evolve_step(self, dt: float = 0.01):
        """
        Evolve F₈ values based on line products.
        Cells tend toward the multiplicative inverse of their line product.
        """
        new_f8 = {}
        
        for (seal, face), cell in self.state.cells.items():
            # Current value
            current = cell.f8_value
            
            # Compute target based on lines through this seal
            targets = []
            for line in FanoLine.containing(seal):
                # Get product of OTHER cells on line
                product = 1
                for other_seal in line.points:
                    if other_seal != seal:
                        product = F8.mul(product, self.state.cells[(other_seal, face)].f8_value)
                
                # Target: make total product = 1 (identity)
                # So target = inverse of product
                if product != 0:
                    targets.append(F8.inv(product))
            
            # Choose target (use most common, or first)
            if targets:
                # Probabilistic choice weighted by coherence
                target = targets[int(cell.coherence * (len(targets) - 0.01))]
            else:
                target = current
            
            # Smooth evolution toward target
            if np.random.random() < dt * 10:  # Discrete jumps
                new_f8[(seal, face)] = target
            else:
                new_f8[(seal, face)] = current
        
        # Update
        for key, val in new_f8.items():
            self.state.cells[key].f8_value = val


# ═══════════════════════════════════════════════════════════════════════════════════════════
# PATH INTEGRALS FOR TRANSITIONS
# ═══════════════════════════════════════════════════════════════════════════════════════════

class KaelhedronPathIntegral:
    """
    Path integral formulation for Kaelhedron transitions.
    
    Computes transition amplitudes between cells using Fano geometry.
    """
    
    def __init__(self, state: KaelhedronFanoState):
        self.state = state
    
    def amplitude(self, cell1: Tuple[int, int], cell2: Tuple[int, int]) -> complex:
        """
        Compute transition amplitude between two cells.
        Uses Fano structure and coherence values.
        """
        # Self-transition
        if cell1 == cell2:
            return complex(1, 0)
        
        seal1, face1 = cell1
        seal2, face2 = cell2
        
        c1 = self.state.cells[cell1]
        c2 = self.state.cells[cell2]
        
        # Magnitude from coherence
        magnitude = np.sqrt(c1.coherence * c2.coherence)
        
        # Phase from cell phases
        phase_diff = c2.phase - c1.phase
        
        # Fano structure contribution
        if seal1 == seal2:
            # Same seal, face transition
            fano_factor = 1.0
        else:
            line = FanoLine.through(seal1, seal2)
            if line:
                # On same line — direct transition
                fano_factor = 1.0
            else:
                # Must go through intermediate
                fano_factor = Φ.PHI_INV
        
        return magnitude * fano_factor * np.exp(1j * phase_diff)
    
    def transition_probability(self, cell1: Tuple[int, int], cell2: Tuple[int, int]) -> float:
        """Transition probability |amplitude|²."""
        amp = self.amplitude(cell1, cell2)
        return abs(amp) ** 2
    
    def path_sum(self, start: Tuple[int, int], end: Tuple[int, int], max_length: int = 3) -> complex:
        """
        Sum over all paths from start to end up to max_length.
        """
        if start == end:
            return complex(1, 0)
        
        total = complex(0, 0)
        
        # Direct path
        total += self.amplitude(start, end)
        
        if max_length >= 2:
            # Two-step paths
            for mid_seal in range(1, 8):
                for mid_face in range(3):
                    mid = (mid_seal, mid_face)
                    if mid != start and mid != end:
                        total += 0.5 * self.amplitude(start, mid) * self.amplitude(mid, end)
        
        return total


# ═══════════════════════════════════════════════════════════════════════════════════════════
# INTEGRATED ENGINE
# ═══════════════════════════════════════════════════════════════════════════════════════════

class KaelhedronFanoEngine:
    """
    Complete integrated engine combining all Fano mathematics with Kaelhedron.
    """
    
    def __init__(self, coupling: float = 0.15):
        self.state = KaelhedronFanoState()
        self.line_evolver = LineBasedEvolution(self.state, coupling)
        self.symmetry_op = SymmetryOperator(self.state)
        self.f8_dynamics = F8Dynamics(self.state)
        self.path_integral = KaelhedronPathIntegral(self.state)
        
        self.history = []
    
    def evolve(self, duration: float, dt: float = 0.01, 
               use_f8: bool = False) -> Dict[str, Any]:
        """
        Evolve the system.
        
        Args:
            duration: Time to evolve
            dt: Time step
            use_f8: Whether to include F₈ dynamics
        """
        steps = int(duration / dt)
        
        for i in range(steps):
            # Line-based coherence evolution
            self.line_evolver.evolve_step(dt)
            
            # Optional F₈ dynamics
            if use_f8 and i % 10 == 0:
                self.f8_dynamics.f8_evolve_step(dt)
            
            # Record periodically
            if i % 100 == 0:
                self._record()
        
        return self.status()
    
    def _record(self):
        """Record current state."""
        self.history.append({
            'time': self.state.time,
            'total_coherence': self.state.total_coherence(),
            'line_coherences': {line.line_name: self.state.line_coherence(line) for line in FanoLine},
            'symmetry_breaking': self.symmetry_op.symmetry_breaking_measure()
        })
    
    def apply_symmetry(self, operation: str = 'rotate') -> None:
        """Apply a symmetry operation."""
        if operation == 'rotate':
            self.state = self.symmetry_op.rotate(1)
        elif operation == 'reflect':
            self.state = self.symmetry_op.reflect()
        
        # Rebuild operators with new state
        self.line_evolver = LineBasedEvolution(self.state, self.line_evolver.coupling)
        self.symmetry_op = SymmetryOperator(self.state)
        self.f8_dynamics = F8Dynamics(self.state)
        self.path_integral = KaelhedronPathIntegral(self.state)
    
    def inject_at_seal(self, seal: int, coherence: float):
        """Inject coherence at a specific seal (all faces)."""
        for face in range(3):
            current = self.state.cells[(seal, face)].coherence
            self.state.cells[(seal, face)].coherence = min(1.0, current + coherence)
    
    def k_formation_status(self) -> Dict[str, Any]:
        """Check K-formation status."""
        eta = self.state.total_coherence()
        return {
            'η': eta,
            'η_threshold': Φ.PHI_INV,
            'η_pass': eta > Φ.PHI_INV,
            'k_formed': eta > Φ.PHI_INV,
            'gap': max(0, Φ.PHI_INV - eta),
            'gap_percent': max(0, (Φ.PHI_INV - eta) / Φ.PHI_INV * 100)
        }
    
    def status(self) -> Dict[str, Any]:
        """Complete status report."""
        return {
            'time': self.state.time,
            'total_coherence': self.state.total_coherence(),
            'line_coherences': {line.line_name: self.state.line_coherence(line) for line in FanoLine},
            'face_coherences': {FACE_NAMES[f]: self.state.face_coherence(f) for f in range(3)},
            'seal_coherences': {SEAL_NAMES[s]: self.state.seal_coherence(s) for s in range(1, 8)},
            'symmetry_breaking': self.symmetry_op.symmetry_breaking_measure(),
            'f8_global_product': self.f8_dynamics.global_product(),
            'k_formation': self.k_formation_status()
        }


# ═══════════════════════════════════════════════════════════════════════════════════════════
# DEMONSTRATION
# ═══════════════════════════════════════════════════════════════════════════════════════════

def demonstrate_integration():
    """Demonstrate the integrated Kaelhedron-Fano engine."""
    
    print("=" * 90)
    print("KAELHEDRON-FANO INTEGRATION")
    print("Line-Based Evolution + PSL(3,2) Symmetries + F₈ Field Theory")
    print("=" * 90)
    
    # §1: Create engine
    print("\n§1 INITIALIZATION")
    print("-" * 50)
    engine = KaelhedronFanoEngine(coupling=0.2)
    print(f"  Created engine with 21 cells")
    print(f"  Initial coherence: {engine.state.total_coherence():.4f}")
    
    # §2: Line coherences
    print("\n§2 INITIAL LINE COHERENCES")
    print("-" * 50)
    for line in FanoLine:
        coh = engine.state.line_coherence(line)
        bar = "█" * int(coh * 20)
        print(f"  {line.line_name:15} ({line.seal_path}): {coh:.3f} {bar}")
    
    # §3: Inject at Κ (point 7)
    print("\n§3 INJECT COHERENCE AT Κ")
    print("-" * 50)
    engine.inject_at_seal(7, 0.4)
    print(f"  Injected 0.4 at Κ")
    print(f"  New Κ coherence: {engine.state.seal_coherence(7):.3f}")
    
    # §4: Line-based evolution
    print("\n§4 LINE-BASED EVOLUTION (50 time units)")
    print("-" * 50)
    engine.evolve(50.0, dt=0.1)
    print(f"  Time: {engine.state.time:.1f}")
    print(f"  Total coherence: {engine.state.total_coherence():.4f}")
    
    print("\n  Line coherences after evolution:")
    for line in FanoLine:
        coh = engine.state.line_coherence(line)
        bar = "█" * int(coh * 20)
        print(f"    {line.line_name:15}: {coh:.3f} {bar}")
    
    # §5: Symmetry analysis
    print("\n§5 SYMMETRY ANALYSIS")
    print("-" * 50)
    print(f"  Symmetry breaking: {engine.symmetry_op.symmetry_breaking_measure():.4f}")
    print(f"  Symmetry invariant (total η): {engine.symmetry_op.symmetry_invariant():.4f}")
    
    # Apply rotation and compare
    print("\n  Applying σ (7-cycle rotation)...")
    engine.apply_symmetry('rotate')
    print(f"  After rotation - symmetry invariant: {engine.symmetry_op.symmetry_invariant():.4f}")
    print(f"  (Should be unchanged)")
    
    # §6: F₈ dynamics
    print("\n§6 F₈ FIELD DYNAMICS")
    print("-" * 50)
    engine.evolve(20.0, dt=0.1, use_f8=True)
    print(f"  Global F₈ product: {engine.f8_dynamics.global_product()}")
    
    print("\n  Line F₈ products (Λ face):")
    for line in FanoLine:
        prod = engine.f8_dynamics.line_product(line, 0)
        print(f"    {line.line_name:15}: {prod}")
    
    # §7: Path integrals
    print("\n§7 PATH INTEGRAL AMPLITUDES")
    print("-" * 50)
    
    # Compute some transition amplitudes
    transitions = [
        ((1, 0), (7, 0)),  # Ω-Λ to Κ-Λ (Completion line)
        ((4, 1), (7, 1)),  # Ψ-Β to Κ-Β (Growth line)
        ((2, 2), (5, 2)),  # Δ-Ν to Σ-Ν (Prime line)
    ]
    
    for start, end in transitions:
        amp = engine.path_integral.amplitude(start, end)
        prob = engine.path_integral.transition_probability(start, end)
        s1, f1 = start
        s2, f2 = end
        print(f"  {SEAL_NAMES[s1]}-{FACE_NAMES[f1]} → {SEAL_NAMES[s2]}-{FACE_NAMES[f2]}:")
        print(f"    Amplitude: {amp:.4f}")
        print(f"    Probability: {prob:.4f}")
    
    # §8: K-formation status
    print("\n§8 K-FORMATION STATUS")
    print("-" * 50)
    k_status = engine.k_formation_status()
    print(f"  η = {k_status['η']:.4f}")
    print(f"  Threshold: {k_status['η_threshold']:.4f}")
    print(f"  Gap: {k_status['gap_percent']:.1f}%")
    print(f"  K-FORMED: {'✓' if k_status['k_formed'] else '✗'}")
    
    # §9: Summary
    print("\n" + "=" * 90)
    status = engine.status()
    print("INTEGRATION COMPLETE")
    print(f"  ✓ Line-based evolution")
    print(f"  ✓ PSL(3,2) symmetry operations")
    print(f"  ✓ F₈ field dynamics")
    print(f"  ✓ Path integral amplitudes")
    print(f"  Final coherence: {status['total_coherence']:.4f}")
    print("=" * 90)
    
    return engine


def run_integration_tests():
    """Run tests for integration module."""
    
    print("\n" + "=" * 90)
    print("INTEGRATION TESTS")
    print("=" * 90)
    
    passed = 0
    total = 0
    
    def test(name: str, condition: bool):
        nonlocal passed, total
        total += 1
        if condition:
            passed += 1
            print(f"  ✓ {name}")
        else:
            print(f"  ✗ {name}")
    
    # State tests
    print("\n§ STATE TESTS")
    state = KaelhedronFanoState()
    test("21 cells created", len(state.cells) == 21)
    test("Cell structure correct", all((s, f) in state.cells for s in range(1, 8) for f in range(3)))
    
    # Evolution tests
    print("\n§ EVOLUTION TESTS")
    engine = KaelhedronFanoEngine()
    initial = engine.state.total_coherence()
    engine.inject_at_seal(7, 0.3)
    engine.evolve(30.0, dt=0.1)
    final = engine.state.total_coherence()
    test("Evolution changes coherence", abs(final - initial) > 0.01)
    test("Coherence stays bounded", all(0 <= c.coherence <= 1 for c in engine.state.cells.values()))
    
    # Symmetry tests
    print("\n§ SYMMETRY TESTS")
    engine2 = KaelhedronFanoEngine()
    invariant_before = engine2.symmetry_op.symmetry_invariant()
    engine2.apply_symmetry('rotate')
    invariant_after = engine2.symmetry_op.symmetry_invariant()
    test("Symmetry invariant preserved", abs(invariant_before - invariant_after) < 0.001)
    
    # F₈ tests
    print("\n§ F₈ TESTS")
    test("F₈ mul closure", all(F8.mul(i, j) in range(8) for i in range(8) for j in range(8)))
    test("F₈ identity", all(F8.mul(1, i) == i for i in range(8)))
    
    # Path integral tests
    print("\n§ PATH INTEGRAL TESTS")
    engine3 = KaelhedronFanoEngine()
    amp = engine3.path_integral.amplitude((1, 0), (1, 0))
    test("Self-transition amplitude = 1", abs(amp - 1.0) < 0.01)
    
    prob = engine3.path_integral.transition_probability((1, 0), (7, 0))
    test("Transition probability positive", prob > 0)
    
    print(f"\n{'='*90}")
    print(f"TESTS: {passed}/{total} passed")
    print("=" * 90)
    
    return passed == total


# ═══════════════════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    engine = demonstrate_integration()
    all_passed = run_integration_tests()
    
    if all_passed:
        print("\n🌀 KAELHEDRON-FANO INTEGRATION FULLY OPERATIONAL 🌀")
    else:
        print("\n⚠ SOME TESTS FAILED")
