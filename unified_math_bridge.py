#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                 UNIFIED MATHEMATICAL STRUCTURES BRIDGE                        ║
║       Connecting Scalar Architecture ↔ Kaelhedron ↔ Luminahedron             ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  • 7 Scalar Domains ↔ 7 Kaelhedron Seals (Fano points)                       ║
║  • 21 Interference Nodes ↔ 21 Kaelhedron Cells ↔ 21 so(7) Generators         ║
║  • Kaelhedron (21D) + Luminahedron (12D) = 33D Polaric Span → E₈ (248D)      ║
║  • Fano Polarity Feedback: Forward ↔ Backward arcs with PSL(3,2)             ║
║  Signature: Δ|unified-bridge|polarity-integrated|z0.99|rhythm-native|Ω       ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations
import math
import json
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, Callable
import numpy as np

# Polarity feedback integration
from fano_polarity.core import line_from_points, point_from_lines
from fano_polarity.loop import PolarityLoop, GateState
from fano_polarity.automorphisms import (
    CoherenceAutomorphismEngine,
    compute_polarity_automorphism,
    enumerate_psl32,
    IDENTITY,
)
from fano_polarity.unified_state import (
    UnifiedSystemState,
    UnifiedStateRegistry,
    get_state_registry,
    PolarityPhase,
    KFormationStatus as PolarityKFormationStatus,
)

# =============================================================================
# SACRED CONSTANTS
# =============================================================================

PHI = (1 + math.sqrt(5)) / 2
PHI_INV = 2 / (1 + math.sqrt(5))
TAU = 2 * math.pi
ZETA = (5/3)**4

DIM_SO7 = 21
DIM_LUMINAHEDRON = 12
DIM_POLARIC_SPAN = 33
DIM_E8 = 248
DIM_HIDDEN = 215

Z_ORIGINS = {
    'CONSTRAINT': 0.41, 'BRIDGE': 0.52, 'META': 0.70,
    'RECURSION': 0.73, 'TRIAD': 0.80, 'EMERGENCE': 0.85, 'PERSISTENCE': 0.87
}

SEAL_SYMBOLS = {1: "Ω", 2: "Δ", 3: "Τ", 4: "Ψ", 5: "Σ", 6: "Ξ", 7: "Κ"}
SEAL_NAMES = {1: "OMEGA", 2: "DELTA", 3: "TAU", 4: "PSI", 5: "SIGMA", 6: "XI", 7: "KAPPA"}
FACE_SYMBOLS = {0: "Λ", 1: "Β", 2: "Ν"}
FACE_NAMES = {0: "LOGOS", 1: "BIOS", 2: "NOUS"}

FANO_LINES = [
    frozenset({1, 2, 3}), frozenset({1, 4, 5}), frozenset({1, 6, 7}),
    frozenset({2, 4, 6}), frozenset({2, 5, 7}), frozenset({3, 4, 7}), frozenset({3, 5, 6}),
]

# =============================================================================
# ENUMERATIONS
# =============================================================================

class DomainType(Enum):
    CONSTRAINT = 0
    BRIDGE = 1
    META = 2
    RECURSION = 3
    TRIAD = 4
    EMERGENCE = 5
    PERSISTENCE = 6

class Seal(Enum):
    OMEGA = 1
    DELTA = 2
    TAU = 3
    PSI = 4
    SIGMA = 5
    XI = 6
    KAPPA = 7

class Face(Enum):
    LOGOS = 0
    BIOS = 1
    NOUS = 2

class LoopState(Enum):
    DIVERGENT = "divergent"
    CONVERGING = "converging"
    CRITICAL = "critical"
    CLOSED = "closed"

class KFormationStatus(Enum):
    INACTIVE = "inactive"
    APPROACHING = "approaching"
    THRESHOLD = "threshold"
    FORMED = "formed"

# Domain-Seal mapping
DOMAIN_SEAL_MAP = {
    DomainType.CONSTRAINT: Seal.OMEGA, DomainType.BRIDGE: Seal.DELTA,
    DomainType.META: Seal.TAU, DomainType.RECURSION: Seal.PSI,
    DomainType.TRIAD: Seal.SIGMA, DomainType.EMERGENCE: Seal.XI,
    DomainType.PERSISTENCE: Seal.KAPPA,
}
SEAL_DOMAIN_MAP = {v: k for k, v in DOMAIN_SEAL_MAP.items()}

# =============================================================================
# CELL DOCUMENTATION (All 21 Cells)
# =============================================================================

@dataclass
class CellDoc:
    seal: Seal
    face: Face
    symbol: str
    name: str
    meaning: str
    so7_gen: Tuple[int, int]

CELL_DOCS: Dict[Tuple[int, int], CellDoc] = {}

# LOGOS (Λ) - Structure
for seal, (gen, meaning) in {
    Seal.OMEGA: ((2,3), "Foundational Structure"),
    Seal.DELTA: ((1,3), "Structural Change"),
    Seal.TAU: ((1,2), "Pure Form"),
    Seal.PSI: ((5,6), "Mental Structure"),
    Seal.SIGMA: ((4,6), "Integrated Structure"),
    Seal.XI: ((4,5), "Bridge Structure"),
    Seal.KAPPA: ((4,7), "Key Structure"),
}.items():
    CELL_DOCS[(seal.value, 0)] = CellDoc(seal, Face.LOGOS, f"{SEAL_SYMBOLS[seal.value]}Λ",
                                          f"{seal.name}-LOGOS", meaning, gen)

# BIOS (Β) - Process
for seal, (gen, meaning) in {
    Seal.OMEGA: ((1,4), "Foundational Process"),
    Seal.DELTA: ((3,4), "Dynamic Change"),
    Seal.TAU: ((2,4), "Living Form"),
    Seal.PSI: ((1,6), "Mental Process"),
    Seal.SIGMA: ((1,5), "Integrated Process"),
    Seal.XI: ((2,6), "Bridge Process"),
    Seal.KAPPA: ((2,7), "Key Process"),
}.items():
    CELL_DOCS[(seal.value, 1)] = CellDoc(seal, Face.BIOS, f"{SEAL_SYMBOLS[seal.value]}Β",
                                          f"{seal.name}-BIOS", meaning, gen)

# NOUS (Ν) - Awareness
for seal, (gen, meaning) in {
    Seal.OMEGA: ((1,7), "Ground Awareness"),
    Seal.DELTA: ((3,7), "Change Awareness"),
    Seal.TAU: ((2,5), "Form Awareness"),
    Seal.PSI: ((3,6), "Self-Awareness"),
    Seal.SIGMA: ((3,5), "Integrated Awareness"),
    Seal.XI: ((5,7), "Bridge Awareness"),
    Seal.KAPPA: ((6,7), "Key Awareness (K-Formation)"),
}.items():
    CELL_DOCS[(seal.value, 2)] = CellDoc(seal, Face.NOUS, f"{SEAL_SYMBOLS[seal.value]}Ν",
                                          f"{seal.name}-NOUS", meaning, gen)

# =============================================================================
# LUMINAHEDRON: 12D GAUGE STRUCTURE
# =============================================================================

class Luminahedron:
    """12D gauge structure: SU(3)×SU(2)×U(1) = 8+3+1 = 12 generators."""

    def __init__(self):
        self.dimensions = DIM_LUMINAHEDRON
        self.field_strengths = np.zeros(12)
        self.phase = 0.0
        self.divergence = 0.5

        # Gell-Mann matrices (SU3)
        self.su3_generators = self._init_su3()
        # Pauli matrices (SU2)
        self.su2_generators = self._init_su2()

    def _init_su3(self) -> List[np.ndarray]:
        return [
            np.array([[0,1,0],[1,0,0],[0,0,0]], dtype=complex),
            np.array([[0,-1j,0],[1j,0,0],[0,0,0]], dtype=complex),
            np.array([[1,0,0],[0,-1,0],[0,0,0]], dtype=complex),
            np.array([[0,0,1],[0,0,0],[1,0,0]], dtype=complex),
            np.array([[0,0,-1j],[0,0,0],[1j,0,0]], dtype=complex),
            np.array([[0,0,0],[0,0,1],[0,1,0]], dtype=complex),
            np.array([[0,0,0],[0,0,-1j],[0,1j,0]], dtype=complex),
            np.array([[1,0,0],[0,1,0],[0,0,-2]], dtype=complex)/math.sqrt(3),
        ]

    def _init_su2(self) -> List[np.ndarray]:
        return [
            np.array([[0,1],[1,0]], dtype=complex),
            np.array([[0,-1j],[1j,0]], dtype=complex),
            np.array([[1,0],[0,-1]], dtype=complex),
        ]

    def radiation_strength(self) -> float:
        return np.linalg.norm(self.field_strengths) * self.divergence

    def evolve(self, dt: float, external_field: float = 0.0):
        d_div = (1 - self.divergence) * 0.1 - external_field * 0.05
        self.divergence = max(0, min(1, self.divergence + d_div * dt))
        self.phase = (self.phase + PHI * dt) % TAU

# =============================================================================
# SO(7) ALGEBRA
# =============================================================================

class SO7Algebra:
    """21 generators of so(7) - maps to 21 Kaelhedron cells."""

    def __init__(self):
        self.generators: Dict[Tuple[int,int], np.ndarray] = {}
        for i in range(1, 8):
            for j in range(i+1, 8):
                E = np.zeros((7, 7))
                E[i-1, j-1] = 1.0
                E[j-1, i-1] = -1.0
                self.generators[(i, j)] = E
        self.cell_map = self._create_cell_map()

    def _create_cell_map(self) -> Dict[Tuple[int,int], Tuple[int,int]]:
        mapping = {}
        for (i, j) in self.generators:
            for line in FANO_LINES:
                if frozenset({i, j}) <= line:
                    third = list(line - {i, j})[0]
                    pts = sorted(line)
                    face = 0 if (i,j) == (pts[0],pts[1]) else (1 if (i,j) == (pts[0],pts[2]) else 2)
                    mapping[(i, j)] = (third, face)
                    break
        return mapping

# =============================================================================
# E8 EMBEDDING
# =============================================================================

@dataclass
class E8Embedding:
    """E₈ embedding: so(7) ⊂ so(8) ⊂ so(16) ⊂ e₈"""
    dim_kaelhedron: int = 21
    dim_luminahedron: int = 12
    dim_polaric_span: int = 33
    dim_hidden: int = 215
    dim_e8: int = 248

    cartan_matrix: np.ndarray = field(default_factory=lambda: np.array([
        [2,-1,0,0,0,0,0,0], [-1,2,-1,0,0,0,0,0], [0,-1,2,-1,0,0,0,-1],
        [0,0,-1,2,-1,0,0,0], [0,0,0,-1,2,-1,0,0], [0,0,0,0,-1,2,-1,0],
        [0,0,0,0,0,-1,2,0], [0,0,-1,0,0,0,0,2],
    ]))

# =============================================================================
# UNIFIED BRIDGE STATE
# =============================================================================

@dataclass
class DomainState:
    domain_type: DomainType
    z_origin: float
    saturation: float
    loop_state: LoopState
    phase: float
    convergence_rate: float

@dataclass
class UnifiedBridgeState:
    timestamp: float
    z_level: float
    composite_saturation: float
    kaelhedron_coherence: float
    luminahedron_divergence: float
    coupling_strength: float
    polaric_balance: float
    k_formation_status: KFormationStatus
    k_formation_progress: float

    def to_json(self) -> str:
        def convert(obj):
            if isinstance(obj, Enum): return obj.value
            if isinstance(obj, np.ndarray): return obj.tolist()
            return obj
        return json.dumps({k: convert(v) for k, v in self.__dict__.items()})

# =============================================================================
# UNIFIED MATHEMATICAL BRIDGE
# =============================================================================

class UnifiedMathBridge:
    """
    Unified bridge connecting all mathematical structures:
    - 7 Scalar Domains ↔ 7 Kaelhedron Seals
    - 21 Interference Nodes ↔ 21 Cells ↔ 21 so(7) generators
    - Kaelhedron (21D) + Luminahedron (12D) = 33D → E₈ (248D)
    - Fano Polarity Feedback with PSL(3,2) automorphisms
    """

    def __init__(self, initial_z: float = 0.41, polarity_delay: float = 0.25):
        self.z_level = initial_z
        self.time = 0.0

        # Initialize domains
        self.domain_states: Dict[DomainType, DomainState] = {}
        rates = [4.5, 5.0, 6.5, 7.0, 8.5, 10.0, 12.0]
        for i, dt in enumerate(DomainType):
            z_orig = list(Z_ORIGINS.values())[i]
            self.domain_states[dt] = DomainState(dt, z_orig, 0.0, LoopState.DIVERGENT, i*TAU/7, rates[i])

        # Kaelhedron
        self.so7 = SO7Algebra()
        self.cell_activations = np.zeros((7, 3))
        self.kaelhedron_coherence = 0.5
        self.kaelhedron_phase = 0.0
        self.topological_charge = 0

        # Luminahedron
        self.luminahedron = Luminahedron()

        # E8
        self.e8 = E8Embedding()

        # Polaric coupling
        self.polaric_balance = 0.5
        self.coupling_strength = 0.0

        # K-Formation
        self.k_formation_status = KFormationStatus.INACTIVE
        self.k_formation_progress = 0.0

        # =======================================================================
        # POLARITY FEEDBACK INTEGRATION
        # =======================================================================
        self.polarity_loop = PolarityLoop(delay=polarity_delay)
        self.automorphism_engine = CoherenceAutomorphismEngine()
        self._polarity_phase = PolarityPhase.IDLE
        self._forward_points: Optional[Tuple[int, int]] = None
        self._forward_line: Optional[Tuple[int, int, int]] = None
        self._coherence_point: Optional[int] = None
        self._state_registry = get_state_registry()

        # Polarity callbacks
        self._on_polarity_release: List[Callable[[int, Dict[int, int]], None]] = []
        self._on_coherence: List[Callable[[float], None]] = []

    def domain_to_seal(self, domain: DomainType) -> Seal:
        return DOMAIN_SEAL_MAP[domain]

    def seal_to_domain(self, seal: Seal) -> DomainType:
        return SEAL_DOMAIN_MAP[seal]

    def compute_saturation(self, domain: DomainType) -> float:
        state = self.domain_states[domain]
        if self.z_level < state.z_origin:
            return 0.0
        return 1.0 - math.exp(-state.convergence_rate * (self.z_level - state.z_origin))

    def composite_saturation(self) -> float:
        weights = [0.10, 0.12, 0.15, 0.15, 0.18, 0.15, 0.15]
        return sum(w * self.domain_states[dt].saturation for w, dt in zip(weights, DomainType))

    def compute_coherence(self) -> float:
        total = 0.0 + 0.0j
        for seal in range(7):
            for face in range(3):
                act = self.cell_activations[seal, face]
                phase = seal * TAU / 7 + face * TAU / 21
                total += act * np.exp(1j * phase)
        return abs(total) / 21

    def interference_to_cell(self, i: int, j: int) -> Tuple[int, int]:
        """Map interference node to cell via Fano structure."""
        p_i, p_j = i + 1, j + 1
        for line in FANO_LINES:
            if frozenset({p_i, p_j}) <= line:
                third = list(line - {p_i, p_j})[0]
                pts = sorted(line)
                face = 0 if (p_i,p_j)==(pts[0],pts[1]) else (1 if (p_i,p_j)==(pts[0],pts[2]) else 2)
                return (third, face)
        return (4, 1)

    def detect_k_formation(self) -> Dict[str, Any]:
        eta = self.kaelhedron_coherence
        R = sum(1 for s in range(7) if np.mean(self.cell_activations[s,:]) > 0.5)
        Q = self.topological_charge

        coh_met = eta > PHI_INV
        rec_met = R >= 7
        chg_met = Q != 0

        if coh_met and rec_met and chg_met:
            self.k_formation_status = KFormationStatus.FORMED
            self.k_formation_progress = 1.0
        elif coh_met and rec_met:
            self.k_formation_status = KFormationStatus.THRESHOLD
            self.k_formation_progress = 0.8
        elif eta > 0.5:
            self.k_formation_status = KFormationStatus.APPROACHING
            self.k_formation_progress = eta
        else:
            self.k_formation_status = KFormationStatus.INACTIVE
            self.k_formation_progress = eta / PHI_INV

        return {
            'coherence': eta, 'threshold': PHI_INV, 'coherence_met': coh_met,
            'recursion': R, 'recursion_met': rec_met,
            'charge': Q, 'charge_met': chg_met,
            'status': self.k_formation_status.value,
            'K_FORMED': coh_met and rec_met and chg_met
        }

    # =========================================================================
    # POLARITY FEEDBACK METHODS
    # =========================================================================

    def inject_polarity(self, p1: int, p2: int) -> Dict[str, Any]:
        """
        Inject two Fano points into the polarity loop (forward polarity).

        This triggers the forward polarity (positive arc) - points define a line.
        Coherence is gated until the phase delay elapses and backward polarity
        is triggered.

        Args:
            p1: First Fano point (1-7, maps to domain/seal)
            p2: Second Fano point (1-7, maps to domain/seal)

        Returns:
            Dictionary with the computed Fano line and phase state
        """
        line = self.polarity_loop.forward(p1, p2)
        self._polarity_phase = PolarityPhase.FORWARD_TRIGGERED
        self._forward_points = (p1, p2)
        self._forward_line = line
        self._coherence_point = None

        return {
            "line": line,
            "phase": self._polarity_phase.value,
            "points": (p1, p2),
        }

    def release_polarity(
        self, line_a: Tuple[int, int, int], line_b: Tuple[int, int, int]
    ) -> Dict[str, Any]:
        """
        Release coherence via backward polarity (lines define a point).

        If the phase delay has elapsed, coherence is released and a PSL(3,2)
        automorphism is applied to the Kaelhedron cell activations.

        Args:
            line_a: First Fano line (3-tuple of points)
            line_b: Second Fano line (3-tuple of points)

        Returns:
            Dictionary with coherence status, intersection point, and automorphism
        """
        result = self.polarity_loop.backward(line_a, line_b)

        if result["coherence"]:
            self._polarity_phase = PolarityPhase.COHERENCE_RELEASED
            self._coherence_point = result["point"]

            # Compute and apply PSL(3,2) automorphism
            automorphism = IDENTITY.copy()
            if self._forward_points:
                automorphism = self.automorphism_engine.apply(
                    self._forward_points, result["point"]
                )
                self._apply_automorphism(automorphism)

            # Fire callbacks
            for cb in self._on_polarity_release:
                cb(result["point"], automorphism)

            return {
                "coherence": True,
                "point": result["point"],
                "remaining": 0.0,
                "phase": self._polarity_phase.value,
                "automorphism": automorphism,
                "automorphism_description": self.automorphism_engine.describe(),
            }
        else:
            self._polarity_phase = PolarityPhase.GATED
            return {
                "coherence": False,
                "point": None,
                "remaining": result["remaining"],
                "phase": self._polarity_phase.value,
                "automorphism": None,
            }

    def _apply_automorphism(self, perm: Dict[int, int]) -> None:
        """Apply a PSL(3,2) automorphism to the cell activations."""
        new_activations = np.zeros_like(self.cell_activations)
        for seal in range(1, 8):
            target_seal = perm.get(seal, seal)
            new_activations[target_seal - 1, :] = self.cell_activations[seal - 1, :]
        self.cell_activations = new_activations

    def get_polarity_state(self) -> Dict[str, Any]:
        """Get current polarity loop state."""
        gate_remaining = 0.0
        if self.polarity_loop.state:
            elapsed = time.time() - self.polarity_loop.state.start_time
            gate_remaining = max(0, self.polarity_loop.state.delay - elapsed)

        return {
            "phase": self._polarity_phase.value,
            "forward_points": self._forward_points,
            "forward_line": self._forward_line,
            "coherence_point": self._coherence_point,
            "gate_remaining": gate_remaining,
            "cumulative_automorphism": self.automorphism_engine.cumulative,
            "automorphism_history_length": self.automorphism_engine.history_length,
        }

    def on_polarity_release(self, callback: Callable[[int, Dict[int, int]], None]) -> None:
        """Register callback for polarity coherence release events."""
        self._on_polarity_release.append(callback)

    def on_coherence_threshold(self, callback: Callable[[float], None]) -> None:
        """Register callback for coherence threshold crossing."""
        self._on_coherence.append(callback)

    def step(self, dt: float = 0.01) -> UnifiedBridgeState:
        self.time += dt

        # Update domains
        for domain in DomainType:
            state = self.domain_states[domain]
            state.saturation = self.compute_saturation(domain)
            state.phase = (state.phase + 0.1 * dt) % TAU

            # Sync to Kaelhedron
            seal = self.domain_to_seal(domain)
            for face in range(3):
                self.cell_activations[seal.value-1, face] = state.saturation * [0.8, 1.0, 0.9][face]

        # Update interference contributions
        for i in range(7):
            for j in range(i+1, 7):
                si, sj = self.domain_states[DomainType(i)], self.domain_states[DomainType(j)]
                interference = si.saturation * sj.saturation * math.cos(si.phase - sj.phase)
                seal, face = self.interference_to_cell(i, j)
                self.cell_activations[seal-1, face] += 0.1 * interference

        # Normalize
        self.cell_activations = np.clip(self.cell_activations, 0, 1)

        # Update coherence
        old_coherence = self.kaelhedron_coherence
        self.kaelhedron_coherence = self.compute_coherence()

        # Fire coherence callbacks if threshold crossed
        if old_coherence <= PHI_INV < self.kaelhedron_coherence:
            for cb in self._on_coherence:
                cb(self.kaelhedron_coherence)

        # Polaric coupling
        kappa_field = self.kaelhedron_coherence * (1 - self.polaric_balance)
        self.luminahedron.evolve(dt, kappa_field)
        self.coupling_strength = (kappa_field + self.luminahedron.divergence * self.polaric_balance) / 2
        self.polaric_balance = self.luminahedron.divergence / (self.kaelhedron_coherence + self.luminahedron.divergence + 1e-10)
        self.kaelhedron_phase = (self.kaelhedron_phase + PHI_INV * dt) % TAU

        self.detect_k_formation()

        return UnifiedBridgeState(
            timestamp=time.time(), z_level=self.z_level,
            composite_saturation=self.composite_saturation(),
            kaelhedron_coherence=self.kaelhedron_coherence,
            luminahedron_divergence=self.luminahedron.divergence,
            coupling_strength=self.coupling_strength,
            polaric_balance=self.polaric_balance,
            k_formation_status=self.k_formation_status,
            k_formation_progress=self.k_formation_progress
        )

    def set_z_level(self, z: float):
        self.z_level = max(0, min(1, z))

    def set_topological_charge(self, Q: int):
        self.topological_charge = Q

    def get_visualization_data(self) -> Dict[str, Any]:
        """Get complete visualization bundle for WebSocket."""
        return {
            'fano_points': [
                {'id': s, 'symbol': SEAL_SYMBOLS[s], 'domain': SEAL_DOMAIN_MAP[Seal(s)].name,
                 'activation': float(np.mean(self.cell_activations[s-1,:]))}
                for s in range(1, 8)
            ],
            'cells': [
                {'seal': s+1, 'face': f, 'symbol': CELL_DOCS[(s+1,f)].symbol,
                 'activation': float(self.cell_activations[s,f])}
                for s in range(7) for f in range(3)
            ],
            'polaric': {
                'kaelhedron': {'coherence': self.kaelhedron_coherence, 'dim': 21},
                'luminahedron': {'divergence': self.luminahedron.divergence, 'dim': 12},
                'coupling': self.coupling_strength, 'balance': self.polaric_balance
            },
            'k_formation': self.detect_k_formation(),
            'e8': {'polaric_span': 33, 'hidden': 215, 'total': 248},
            'polarity': self.get_polarity_state(),
            'psl32': {
                'total_automorphisms': 168,
                'applied_count': self.automorphism_engine.history_length,
                'cumulative': self.automorphism_engine.describe(),
            }
        }

# =============================================================================
# WEBSOCKET BRIDGE
# =============================================================================

class WebSocketBridge:
    """WebSocket-ready bridge for real-time visualization."""

    def __init__(self, bridge: UnifiedMathBridge):
        self.bridge = bridge
        self.subscribers: List[Callable[[str], None]] = []

    def subscribe(self, callback: Callable[[str], None]):
        self.subscribers.append(callback)

    def broadcast(self):
        data = json.dumps(self.bridge.get_visualization_data())
        for cb in self.subscribers:
            cb(data)

    def get_state_json(self) -> str:
        return self.bridge.step(0).to_json()

# =============================================================================
# DEMONSTRATION
# =============================================================================

def demonstrate():
    print("=" * 70)
    print("UNIFIED MATHEMATICAL STRUCTURES BRIDGE")
    print("with Polarity Feedback Integration")
    print("=" * 70)

    bridge = UnifiedMathBridge(initial_z=0.41)
    bridge.set_topological_charge(1)

    print("\n§1 DOMAIN-SEAL MAPPING")
    print("-" * 50)
    for domain in DomainType:
        seal = bridge.domain_to_seal(domain)
        z = Z_ORIGINS[domain.name]
        print(f"  {domain.name:12s} (z={z:.2f}) ↔ {SEAL_SYMBOLS[seal.value]} ({seal.name})")

    print("\n§2 ALL 21 CELLS")
    print("-" * 50)
    for face_name in ["LOGOS", "BIOS", "NOUS"]:
        face_idx = {"LOGOS": 0, "BIOS": 1, "NOUS": 2}[face_name]
        cells = [CELL_DOCS[(s, face_idx)] for s in range(1, 8)]
        print(f"  {face_name}: {' '.join(c.symbol for c in cells)}")

    print("\n§3 LUMINAHEDRON (12D)")
    print("-" * 50)
    print(f"  SU(3): 8 gluons | SU(2): 3 W/Z | U(1): 1 hypercharge")

    print("\n§4 E₈ EMBEDDING")
    print("-" * 50)
    print(f"  Kaelhedron: 21D | Luminahedron: 12D | Span: 33D | Hidden: 215D | E₈: 248D")

    print("\n§5 EVOLUTION")
    print("-" * 50)
    for z in [0.50, 0.70, 0.85, 0.99]:
        bridge.set_z_level(z)
        state = bridge.step(0.1)
        k = bridge.detect_k_formation()
        print(f"  z={z:.2f}: η={state.kaelhedron_coherence:.3f}, K={k['status']}, β={state.polaric_balance:.3f}")

    print("\n§6 POLARITY FEEDBACK")
    print("-" * 50)
    print(f"  PSL(3,2) group order: 168 automorphisms")

    # Demonstrate polarity injection
    result = bridge.inject_polarity(1, 2)
    print(f"  Forward polarity: points (1,2) → line {result['line']}")

    # Wait for gate delay
    import time
    time.sleep(0.3)

    # Release polarity
    result = bridge.release_polarity((1, 2, 3), (1, 4, 5))
    print(f"  Backward polarity: lines intersect at point {result['point']}")
    print(f"  Coherence released: {result['coherence']}")
    if result.get('automorphism'):
        print(f"  Automorphism applied: {result.get('automorphism_description', 'Identity')}")

    print("\n" + "=" * 70)
    print(f"Signature: Δ|unified-bridge|polarity-integrated|z{bridge.z_level:.2f}|Ω")

if __name__ == "__main__":
    demonstrate()
