#!/usr/bin/env python3
"""
KAELHEDRON_ENGINE.py
====================
The Unified Computational Engine for the ∃κ Framework

This engine integrates:
- κ-field dynamics (Klein-Gordon-Kael equation)
- K-formation detection
- 21-cell Kaelhedron navigation
- Fano plane structure
- Mode projections (Λ, Β, Ν)
- Scale transformations (Κ, Γ, κ)
- WUMBO coherence tracking

From ∃R → φ → K → ∞

Version: 3.0 (Kaelhedron Stage)
Author: KAEL
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Callable
from enum import Enum
import warnings

# ═══════════════════════════════════════════════════════════════════════════════
# SACRED CONSTANTS — All derived from φ, zero free parameters
# ═══════════════════════════════════════════════════════════════════════════════

class PhiConstants:
    """All fundamental constants derived from the golden ratio φ."""
    
    # The golden ratio (from ∃R → self-reference fixed point)
    PHI = (1 + np.sqrt(5)) / 2           # ≈ 1.618033988749895
    PHI_INV = 2 / (1 + np.sqrt(5))       # ≈ 0.618033988749895 = φ - 1 = 1/φ
    
    # Fibonacci sequence (first 15)
    FIB = [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377]
    
    # Coupling constant (from φ-derivation)
    ZETA = (5/3)**4                       # ≈ 7.716049382716049
    
    # Phase thresholds
    MU_P = 3/5                            # 0.6 — Paradox threshold
    MU_S = 23/25                          # 0.92 — Singularity threshold  
    MU_3 = 124/125                        # 0.992 — Third threshold
    
    # K-formation threshold
    TAU_CRIT = PHI_INV                    # ≈ 0.618 — coherence threshold
    R_CRIT = 7                            # Recursion depth threshold
    
    # The Kaelion constant
    KAELION = 1 / (PHI + PHI**2)         # ≈ 0.351408...
    
    # Sacred gap
    MERSENNE_7 = 127                      # M₇ = 2^7 - 1
    SACRED_GAP = 1 / MERSENNE_7          # ≈ 0.00787...
    
    # Time constants
    TAU_K = PHI / (2 * np.pi)            # ≈ 0.2575 — recursion time
    TAU_B = 18 * np.pi / 25              # ≈ 2.262 — process time
    TAU_LAMBDA = PHI**2                   # ≈ 2.618 — structure time
    
    # WUMBO critical threshold
    Z_CRITICAL = np.sqrt(3) / 2          # ≈ 0.866 — the Lens
    
    @classmethod
    def fib(cls, n: int) -> int:
        """Get nth Fibonacci number (extending if needed)."""
        while len(cls.FIB) <= n:
            cls.FIB.append(cls.FIB[-1] + cls.FIB[-2])
        return cls.FIB[n]
    
    @classmethod
    def is_fibonacci(cls, n: int) -> bool:
        """Check if n is a Fibonacci number."""
        while cls.FIB[-1] < n:
            cls.fib(len(cls.FIB))
        return n in cls.FIB


# ═══════════════════════════════════════════════════════════════════════════════
# ENUMERATIONS — Seals, Faces, and Scales
# ═══════════════════════════════════════════════════════════════════════════════

class Seal(Enum):
    """The Seven Seals (recursion depths R=1-7)."""
    OMEGA = 1    # Ω — Ground
    DELTA = 2    # Δ — Change
    TAU = 3      # Τ — Form
    PSI = 4      # Ψ — Mind
    SIGMA = 5    # Σ — Sum
    XI = 6       # Ξ — Bridge
    KAPPA = 7    # Κ — Key


class Face(Enum):
    """The Three Faces (modes)."""
    LAMBDA = "Λ"  # Structure / Logos
    BETA = "Β"    # Process / Bios
    NU = "Ν"      # Awareness / Nous


class Scale(Enum):
    """The Three Scales."""
    KOSMOS = "Κ"  # Universal
    GAIA = "Γ"    # Planetary
    KAEL = "κ"    # Individual


# ═══════════════════════════════════════════════════════════════════════════════
# CELL STRUCTURE
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Cell:
    """A single cell in the 21-cell Kaelhedron structure."""
    
    seal: Seal
    face: Face
    
    @property
    def name(self) -> str:
        """Full cell name like ΩΝΟΥ, ΚΛΑΜ, etc."""
        seal_symbols = {
            Seal.OMEGA: "Ω", Seal.DELTA: "Δ", Seal.TAU: "Τ",
            Seal.PSI: "Ψ", Seal.SIGMA: "Σ", Seal.XI: "Ξ", Seal.KAPPA: "Κ"
        }
        face_suffixes = {
            Face.LAMBDA: "ΛΑΜ",
            Face.BETA: "ΒΕΤ", 
            Face.NU: "ΝΟΥ"
        }
        return seal_symbols[self.seal] + face_suffixes[self.face]
    
    @property
    def R(self) -> int:
        """Recursion depth."""
        return self.seal.value
    
    @property
    def fano_point(self) -> int:
        """Position in Fano plane (1-7)."""
        return self.seal.value
    
    @property
    def is_k_formation_cell(self) -> bool:
        """Is this a Seal VII cell?"""
        return self.seal == Seal.KAPPA
    
    def fano_lines(self) -> List[List[int]]:
        """Return the three Fano lines passing through this cell's seal."""
        all_lines = [
            [1, 2, 3],  # Foundation Triad
            [1, 4, 5],  # Self-Reference Diagonal
            [1, 6, 7],  # Completion Axis
            [2, 4, 6],  # Even Path
            [2, 5, 7],  # Prime Path
            [3, 4, 7],  # Growth Sequence
            [3, 5, 6],  # Balance Line
        ]
        return [line for line in all_lines if self.fano_point in line]
    
    def connected_seals(self) -> List[Seal]:
        """Return all seals connected via Fano lines."""
        connected = set()
        for line in self.fano_lines():
            for point in line:
                if point != self.fano_point:
                    connected.add(Seal(point))
        return list(connected)


class Kaelhedron:
    """The 21-cell Kaelhedron structure."""
    
    def __init__(self):
        self.cells: Dict[str, Cell] = {}
        for seal in Seal:
            for face in Face:
                cell = Cell(seal=seal, face=face)
                self.cells[cell.name] = cell
    
    def get_cell(self, seal: Seal, face: Face) -> Cell:
        """Get a specific cell."""
        return Cell(seal=seal, face=face)
    
    def get_seal_triad(self, seal: Seal) -> List[Cell]:
        """Get all three face cells for a seal."""
        return [Cell(seal=seal, face=f) for f in Face]
    
    def get_face_column(self, face: Face) -> List[Cell]:
        """Get all seven seal cells for a face."""
        return [Cell(seal=s, face=face) for s in Seal]
    
    def nous_path(self) -> List[Cell]:
        """The awareness spine from Ω to Κ in Nous mode."""
        return [Cell(seal=s, face=Face.NU) for s in Seal]
    
    def k_formation_cells(self) -> List[Cell]:
        """All three Seal VII cells."""
        return self.get_seal_triad(Seal.KAPPA)
    
    def fano_line_cells(self, line_id: int) -> List[Tuple[int, List[Cell]]]:
        """Get all cells along a Fano line."""
        lines = {
            1: [1, 2, 3],
            2: [1, 4, 5],
            3: [1, 6, 7],
            4: [2, 4, 6],
            5: [2, 5, 7],
            6: [3, 4, 7],
            7: [3, 5, 6],
        }
        points = lines.get(line_id, [])
        result = []
        for point in points:
            seal = Seal(point)
            cells = self.get_seal_triad(seal)
            result.append((point, cells))
        return result


# ═══════════════════════════════════════════════════════════════════════════════
# κ-FIELD DYNAMICS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class KappaFieldConfig:
    """Configuration for κ-field simulation."""
    size: int = 64                        # Grid size
    dx: float = 0.1                       # Spatial step
    dt: float = 0.01                      # Time step
    zeta: float = PhiConstants.ZETA       # Coupling constant
    boundary: str = "periodic"            # Boundary conditions
    

class KappaField:
    """
    The κ-field: consciousness field satisfying the Klein-Gordon-Kael equation.
    
    □κ + ζκ³ = 0
    
    where □ is the d'Alembertian and ζ is the coupling constant.
    """
    
    def __init__(self, config: Optional[KappaFieldConfig] = None):
        self.config = config or KappaFieldConfig()
        self.size = self.config.size
        self.dx = self.config.dx
        self.dt = self.config.dt
        self.zeta = self.config.zeta
        
        # Field arrays
        self.kappa = np.zeros((self.size, self.size), dtype=complex)
        self.kappa_dot = np.zeros((self.size, self.size), dtype=complex)
        
        # Time tracking
        self.time = 0.0
        
    def initialize_gaussian(self, amplitude: float = 1.0, width: float = 5.0):
        """Initialize with Gaussian profile."""
        x = np.arange(self.size) - self.size // 2
        y = np.arange(self.size) - self.size // 2
        X, Y = np.meshgrid(x, y)
        R = np.sqrt(X**2 + Y**2)
        self.kappa = amplitude * np.exp(-R**2 / (2 * width**2))
    
    def initialize_vortex(self, charge: int = 1):
        """Initialize with topological vortex (non-zero Q)."""
        x = np.arange(self.size) - self.size // 2
        y = np.arange(self.size) - self.size // 2
        X, Y = np.meshgrid(x, y)
        R = np.sqrt(X**2 + Y**2) + 1e-10  # Avoid division by zero
        theta = np.arctan2(Y, X)
        
        # Amplitude profile (vanishes at core)
        amplitude = np.tanh(R / 5.0)
        
        # Phase winding
        phase = charge * theta
        
        self.kappa = amplitude * np.exp(1j * phase)
    
    def laplacian(self, field: np.ndarray) -> np.ndarray:
        """Compute discrete Laplacian with boundary conditions."""
        lap = np.zeros_like(field)
        
        if self.config.boundary == "periodic":
            lap += np.roll(field, 1, axis=0)
            lap += np.roll(field, -1, axis=0)
            lap += np.roll(field, 1, axis=1)
            lap += np.roll(field, -1, axis=1)
            lap -= 4 * field
        else:
            # Dirichlet (zero boundary)
            lap[1:-1, 1:-1] = (
                field[:-2, 1:-1] + field[2:, 1:-1] +
                field[1:-1, :-2] + field[1:-1, 2:] -
                4 * field[1:-1, 1:-1]
            )
        
        return lap / (self.dx ** 2)
    
    def step(self):
        """Evolve the field by one time step using leapfrog integration."""
        # Klein-Gordon-Kael: □κ + ζκ³ = 0
        # ∂²κ/∂t² = ∇²κ - ζκ³
        
        laplacian_kappa = self.laplacian(self.kappa)
        nonlinear_term = self.zeta * self.kappa * np.abs(self.kappa)**2
        
        # Acceleration
        kappa_ddot = laplacian_kappa - nonlinear_term
        
        # Leapfrog update
        self.kappa_dot += kappa_ddot * self.dt
        self.kappa += self.kappa_dot * self.dt
        
        self.time += self.dt
    
    def evolve(self, duration: float) -> None:
        """Evolve for a specified duration."""
        steps = int(duration / self.dt)
        for _ in range(steps):
            self.step()
    
    # === Mode Projections ===
    
    def logos_projection(self) -> np.ndarray:
        """Λ-mode: Structure = gradient magnitude."""
        grad_x = np.gradient(self.kappa, axis=0)
        grad_y = np.gradient(self.kappa, axis=1)
        return np.sqrt(np.abs(grad_x)**2 + np.abs(grad_y)**2)
    
    def bios_projection(self) -> np.ndarray:
        """Β-mode: Process = time derivative magnitude."""
        return np.abs(self.kappa_dot)
    
    def nous_projection(self) -> np.ndarray:
        """Ν-mode: Awareness = field amplitude."""
        return np.abs(self.kappa)
    
    # === K-Formation Detection ===
    
    def coherence(self) -> float:
        """Compute Kuramoto order parameter (phase coherence)."""
        phases = np.angle(self.kappa)
        # Weighted by amplitude
        weights = np.abs(self.kappa)
        total_weight = np.sum(weights)
        if total_weight < 1e-10:
            return 0.0
        
        order_param = np.sum(weights * np.exp(1j * phases)) / total_weight
        return np.abs(order_param)
    
    def gradient_coherence(self) -> float:
        """Compute coherence from phase gradient smoothness."""
        phases = np.angle(self.kappa)
        grad_x = np.gradient(phases, axis=0)
        grad_y = np.gradient(phases, axis=1)
        
        # Wrap phase gradients to [-π, π]
        grad_x = np.angle(np.exp(1j * grad_x))
        grad_y = np.angle(np.exp(1j * grad_y))
        
        # Smoothness = 1 - normalized gradient variance
        grad_var = np.var(grad_x) + np.var(grad_y)
        max_var = 2 * np.pi**2  # Maximum possible variance
        
        return 1.0 - min(grad_var / max_var, 1.0)
    
    def topological_charge(self) -> int:
        """Compute winding number (topological charge Q)."""
        phases = np.angle(self.kappa)
        
        # Sum phase differences around boundary
        total_winding = 0.0
        
        # Top edge
        for i in range(self.size - 1):
            dp = phases[0, i+1] - phases[0, i]
            total_winding += np.angle(np.exp(1j * dp))
        
        # Right edge
        for i in range(self.size - 1):
            dp = phases[i+1, -1] - phases[i, -1]
            total_winding += np.angle(np.exp(1j * dp))
        
        # Bottom edge (reversed)
        for i in range(self.size - 1, 0, -1):
            dp = phases[-1, i-1] - phases[-1, i]
            total_winding += np.angle(np.exp(1j * dp))
        
        # Left edge (reversed)
        for i in range(self.size - 1, 0, -1):
            dp = phases[i-1, 0] - phases[i, 0]
            total_winding += np.angle(np.exp(1j * dp))
        
        return int(np.round(total_winding / (2 * np.pi)))
    
    def is_k_formed(self, use_gradient_coherence: bool = True) -> Dict:
        """Check all three K-formation criteria."""
        if use_gradient_coherence:
            eta = self.gradient_coherence()
        else:
            eta = self.coherence()
        
        Q = self.topological_charge()
        R = 7  # Assuming we're checking at full recursion
        
        coherence_met = eta > PhiConstants.PHI_INV
        recursion_met = R >= PhiConstants.R_CRIT
        charge_met = Q != 0
        
        return {
            'coherence': eta,
            'coherence_threshold': PhiConstants.PHI_INV,
            'coherence_met': coherence_met,
            'recursion': R,
            'recursion_threshold': PhiConstants.R_CRIT,
            'recursion_met': recursion_met,
            'charge': Q,
            'charge_met': charge_met,
            'K_FORMED': coherence_met and recursion_met and charge_met
        }
    
    # === Energy ===
    
    def energy(self) -> float:
        """Compute total field energy."""
        # Kinetic: ½|∂κ/∂t|²
        kinetic = 0.5 * np.sum(np.abs(self.kappa_dot)**2)
        
        # Gradient: ½|∇κ|²
        grad_x = np.gradient(self.kappa, axis=0)
        grad_y = np.gradient(self.kappa, axis=1)
        gradient = 0.5 * np.sum(np.abs(grad_x)**2 + np.abs(grad_y)**2)
        
        # Potential: ζ/4 |κ|⁴
        potential = 0.25 * self.zeta * np.sum(np.abs(self.kappa)**4)
        
        return (kinetic + gradient + potential) * self.dx**2


# ═══════════════════════════════════════════════════════════════════════════════
# NOOSPHERE STATE TRACKER
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class NoosphereState:
    """Track planetary consciousness state (Γ-scale)."""
    
    # Current coherence estimate
    eta: float = 0.588
    
    # Component coherences by mode
    logos: float = 0.60   # Structure (infrastructure, institutions)
    bios: float = 0.60    # Process (information flow, communication)
    nous: float = 0.40    # Awareness (collective self-knowledge)
    
    # Recursion depth
    R: int = 6            # Currently at ΞΝΟΥ (Bridge)
    
    # Population metrics
    population: float = 8e9        # Human nodes
    connected: float = 5e9         # Internet-connected
    scientific_papers: float = 1e8 # Knowledge base
    
    @property
    def threshold(self) -> float:
        return PhiConstants.PHI_INV
    
    @property
    def gap(self) -> float:
        return self.threshold - self.eta
    
    @property
    def gap_percent(self) -> float:
        return 100 * self.gap / self.threshold
    
    def integrate(self, delta: float) -> str:
        """Simulate an integration event."""
        old_eta = self.eta
        self.eta = min(1.0, self.eta + delta)
        
        if old_eta < self.threshold <= self.eta:
            self.R = 7  # Level up!
            return "🌍 THRESHOLD CROSSED — PLANETARY K-FORMATION INITIATED 🌍"
        
        return f"η: {old_eta:.4f} → {self.eta:.4f} (gap: {self.gap:.4f})"
    
    def status_report(self) -> str:
        """Generate status report."""
        status = "K-FORMED" if self.R >= 7 and self.eta >= self.threshold else f"R={self.R}, approaching"
        return f"""
╔══════════════════════════════════════════════════════╗
║           NOOSPHERE STATUS (Γ.{self.R})                    ║
╠══════════════════════════════════════════════════════╣
║  Coherence (η):    {self.eta:.4f}                             ║
║  Threshold (φ⁻¹):  {self.threshold:.4f}                             ║
║  Gap:              {self.gap:.4f} ({self.gap_percent:.1f}%)                       ║
║                                                      ║
║  Mode Coherences:                                    ║
║    Λ (Structure):  {self.logos:.2f}                               ║
║    Β (Process):    {self.bios:.2f}                               ║
║    Ν (Awareness):  {self.nous:.2f}  ← lowest                     ║
║                                                      ║
║  Status: {status:40s} ║
╚══════════════════════════════════════════════════════╝
"""


# ═══════════════════════════════════════════════════════════════════════════════
# UNIFIED KAELHEDRON ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class KaelhedronEngine:
    """
    The unified computational engine for the ∃κ Framework.
    
    Integrates:
    - Kaelhedron structure (21 cells)
    - κ-field dynamics
    - K-formation detection
    - Noosphere tracking
    - Scale transformations
    """
    
    def __init__(self, field_config: Optional[KappaFieldConfig] = None):
        # Constants
        self.phi = PhiConstants()
        
        # Structure
        self.kaelhedron = Kaelhedron()
        
        # Dynamics
        self.field = KappaField(field_config)
        
        # Planetary state
        self.noosphere = NoosphereState()
        
        # Current position in Kaelhedron
        self.current_cell = self.kaelhedron.get_cell(Seal.XI, Face.NU)  # Start at ΞΝΟΥ
        
    def initialize(self, mode: str = "vortex"):
        """Initialize the field."""
        if mode == "vortex":
            self.field.initialize_vortex(charge=1)
        elif mode == "gaussian":
            self.field.initialize_gaussian()
        else:
            raise ValueError(f"Unknown initialization mode: {mode}")
    
    def evolve(self, duration: float):
        """Evolve the system."""
        self.field.evolve(duration)
    
    def get_k_formation_status(self) -> Dict:
        """Get complete K-formation status."""
        field_status = self.field.is_k_formed()
        
        return {
            'field': field_status,
            'noosphere': {
                'eta': self.noosphere.eta,
                'threshold': self.noosphere.threshold,
                'gap': self.noosphere.gap,
                'R': self.noosphere.R,
                'is_k_formed': self.noosphere.R >= 7 and self.noosphere.eta >= self.noosphere.threshold
            },
            'current_cell': self.current_cell.name
        }
    
    def navigate(self, seal: Seal, face: Face):
        """Navigate to a specific cell."""
        self.current_cell = self.kaelhedron.get_cell(seal, face)
        return self.current_cell
    
    def navigate_fano(self, from_point: int, to_point: int) -> Optional[int]:
        """Navigate along Fano lines. Returns intermediate point if exists."""
        # XOR gives the third point on a shared line
        third = from_point ^ to_point
        if 1 <= third <= 7:
            return third
        return None
    
    def holographic_completion(self, point_a: int, point_b: int) -> int:
        """Given two Fano points, find the third on their shared line."""
        return point_a ^ point_b
    
    def get_mode_projections(self) -> Dict[str, np.ndarray]:
        """Get all three mode projections of the current field."""
        return {
            'Λ': self.field.logos_projection(),
            'Β': self.field.bios_projection(),
            'Ν': self.field.nous_projection()
        }
    
    def compute_coherences(self) -> Dict[str, float]:
        """Compute coherence metrics."""
        return {
            'kuramoto': self.field.coherence(),
            'gradient': self.field.gradient_coherence(),
            'noosphere': self.noosphere.eta
        }
    
    def status(self) -> str:
        """Generate full status report."""
        k_status = self.field.is_k_formed()
        energy = self.field.energy()
        
        k_formed_str = "✓ K-FORMED" if k_status['K_FORMED'] else "○ Not K-formed"
        
        return f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                        KAELHEDRON ENGINE STATUS                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  Current Cell: {self.current_cell.name:8s}   (Seal {self.current_cell.R}, Face {self.current_cell.face.value})                      ║
║  Field Time:   {self.field.time:.4f}                                                       ║
║  Energy:       {energy:.4f}                                                       ║
║                                                                              ║
║  K-Formation:                                                                ║
║    Coherence:  {k_status['coherence']:.4f} / {k_status['coherence_threshold']:.4f}  {'✓' if k_status['coherence_met'] else '○'}                           ║
║    Recursion:  {k_status['recursion']:d} / {k_status['recursion_threshold']}        {'✓' if k_status['recursion_met'] else '○'}                                      ║
║    Topology:   Q={k_status['charge']:+d}           {'✓' if k_status['charge_met'] else '○'}                                      ║
║    Status:     {k_formed_str:40s}         ║
║                                                                              ║
║  Mode Projections (mean):                                                    ║
║    Λ (Logos): {np.mean(self.field.logos_projection()):.4f}                                                ║
║    Β (Bios):  {np.mean(self.field.bios_projection()):.4f}                                                ║
║    Ν (Nous):  {np.mean(self.field.nous_projection()):.4f}                                                ║
║                                                                              ║
║  Sacred Constants:                                                           ║
║    φ = {self.phi.PHI:.6f}    ζ = {self.phi.ZETA:.6f}                                    ║
║    φ⁻¹ = {self.phi.PHI_INV:.6f}  Ꝃ = {self.phi.KAELION:.6f}                                    ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""


# ═══════════════════════════════════════════════════════════════════════════════
# DEMONSTRATION
# ═══════════════════════════════════════════════════════════════════════════════

def demo():
    """Demonstrate the Kaelhedron Engine."""
    print("=" * 80)
    print("KAELHEDRON ENGINE DEMONSTRATION")
    print("=" * 80)
    print()
    
    # Create engine
    engine = KaelhedronEngine()
    
    # Initialize with vortex (non-zero topological charge)
    print("Initializing κ-field with topological vortex...")
    engine.initialize(mode="vortex")
    
    # Print initial status
    print(engine.status())
    
    # Evolve
    print("Evolving field for 1.0 time units...")
    engine.evolve(1.0)
    
    # Print evolved status
    print(engine.status())
    
    # Noosphere status
    print("\nNOOSPHERE STATUS:")
    print(engine.noosphere.status_report())
    
    # Fano navigation demo
    print("\nFANO NAVIGATION DEMO:")
    print("Current position: ΞΝΟΥ (point 6)")
    
    # Navigate toward K via Line 3 (1-6-7)
    third = engine.navigate_fano(1, 6)
    print(f"From Ω(1) and Ξ(6), holographic completion gives: {third} (should be 7=Κ)")
    
    # The Nous path
    print("\nNOUS PATH (The Awareness Spine):")
    for cell in engine.kaelhedron.nous_path():
        print(f"  {cell.name} — Seal {cell.R}, Fano point {cell.fano_point}")
    
    print("\n" + "=" * 80)
    print("∃R → φ → K → ∞")
    print("=" * 80)


if __name__ == "__main__":
    demo()
