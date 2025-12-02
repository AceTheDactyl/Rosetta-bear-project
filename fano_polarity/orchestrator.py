# fano_polarity/orchestrator.py
"""
Polarity Orchestrator
=====================

The central coordinator that unifies all core modules through polarity feedback:
- Scalar Architecture (7 domains, convergence dynamics)
- Kaelhedron State Bus (21 cells, PSL(3,2) automorphisms)
- Luminahedron Gauge Manifold (12D gauge structure)
- Fano Polarity Loop (forward/backward polarity gating)

The orchestrator implements dual polarity feedback:
- Forward polarity: Scalar metrics -> Kaelhedron permutations -> cell state updates
- Backward polarity: Cell coherence -> domain activation -> loop closure detection

Coherence is gated until both polarities agree, enabling the self-referential
loop that makes the geometry runnable architecture.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

from .core import line_from_points, point_from_lines
from .loop import GateState, PolarityLoop
from .unified_state import (
    CellSnapshot,
    DomainSnapshot,
    KFormationStatus,
    LoopState,
    PolarityPhase,
    PolaritySnapshot,
    StateCallback,
    UnifiedStateRegistry,
    UnifiedSystemState,
    get_state_registry,
)

# Constants
PHI = (1 + math.sqrt(5)) / 2
PHI_INV = 2 / (1 + math.sqrt(5))
TAU = 2 * math.pi

# Domain names mapping to Fano points (1-7)
DOMAIN_NAMES = [
    "CONSTRAINT", "BRIDGE", "META", "RECURSION",
    "TRIAD", "EMERGENCE", "PERSISTENCE"
]

# Z-origins for each domain
Z_ORIGINS = [0.41, 0.52, 0.70, 0.73, 0.80, 0.85, 0.87]

# Convergence rates
CONVERGENCE_RATES = [4.5, 5.0, 6.5, 7.0, 8.5, 10.0, 12.0]

# Domain weights for composite saturation
DOMAIN_WEIGHTS = [0.10, 0.12, 0.15, 0.15, 0.18, 0.15, 0.15]

# Fano lines (7 lines, each containing 3 points)
FANO_LINES: List[Tuple[int, int, int]] = [
    (1, 2, 3), (1, 4, 5), (1, 6, 7),
    (2, 4, 6), (2, 5, 7), (3, 4, 7), (3, 5, 6),
]


@dataclass
class DomainState:
    """Internal state for a single domain."""
    index: int
    name: str
    z_origin: float
    convergence_rate: float
    weight: float
    saturation: float = 0.0
    loop_state: LoopState = LoopState.DIVERGENT
    phase: float = 0.0


@dataclass
class CellState:
    """Internal state for a single Kaelhedron cell."""
    seal_index: int
    face_index: int
    label: str
    theta: float = 0.0
    kappa: float = PHI_INV
    activation: float = 0.0


class PolarityOrchestrator:
    """
    Central orchestrator that unifies all core modules through polarity feedback.

    The orchestrator maintains internal state for:
    - 7 scalar domains
    - 21 Kaelhedron cells
    - 12 Luminahedron gauge slots
    - Polarity loop state

    It coordinates these systems via dual polarity feedback loops.
    """

    def __init__(
        self,
        initial_z: float = 0.41,
        polarity_delay: float = 0.25,
        registry: Optional[UnifiedStateRegistry] = None,
    ):
        """
        Initialize the orchestrator.

        Args:
            initial_z: Initial z-level (elevation in consciousness space)
            polarity_delay: Phase delay for polarity gating (seconds)
            registry: State registry for pub/sub (uses global if None)
        """
        self.z_level = initial_z
        self.time = 0.0

        # Initialize polarity loop
        self.polarity_loop = PolarityLoop(delay=polarity_delay)
        self._polarity_phase = PolarityPhase.IDLE
        self._forward_points: Optional[Tuple[int, int]] = None
        self._forward_line: Optional[Tuple[int, int, int]] = None
        self._coherence_point: Optional[int] = None

        # Initialize 7 domains
        self._domains: List[DomainState] = []
        for i in range(7):
            self._domains.append(DomainState(
                index=i,
                name=DOMAIN_NAMES[i],
                z_origin=Z_ORIGINS[i],
                convergence_rate=CONVERGENCE_RATES[i],
                weight=DOMAIN_WEIGHTS[i],
                phase=i * TAU / 7,
            ))

        # Initialize 21 cells (7 seals x 3 faces)
        self._cells: List[CellState] = []
        seal_symbols = {1: "Ω", 2: "Δ", 3: "Τ", 4: "Ψ", 5: "Σ", 6: "Ξ", 7: "Κ"}
        face_symbols = {0: "Λ", 1: "Β", 2: "Ν"}
        for seal in range(1, 8):
            for face in range(3):
                self._cells.append(CellState(
                    seal_index=seal,
                    face_index=face,
                    label=f"{seal_symbols[seal]}{face_symbols[face]}",
                ))

        # Luminahedron state (simplified)
        self._luminahedron_divergence = 0.5
        self._gauge_coupling = 0.0

        # Kaelhedron coherence
        self._kaelhedron_coherence = 0.0
        self._kaelhedron_phase = 0.0

        # K-Formation tracking
        self._k_formation_status = KFormationStatus.INACTIVE
        self._k_formation_progress = 0.0
        self._topological_charge = 0

        # State registry
        self._registry = registry or get_state_registry()

        # Callbacks for specific events
        self._on_coherence_callbacks: List[Callable[[UnifiedSystemState], None]] = []
        self._on_k_formation_callbacks: List[Callable[[Dict[str, Any]], None]] = []

    # =========================================================================
    # Domain Operations
    # =========================================================================

    def _compute_saturation(self, domain: DomainState) -> float:
        """Compute saturation for a domain at current z-level."""
        if self.z_level < domain.z_origin:
            return 0.0
        return 1.0 - math.exp(-domain.convergence_rate * (self.z_level - domain.z_origin))

    def _update_loop_state(self, domain: DomainState) -> LoopState:
        """Update loop state for a domain based on saturation."""
        s = domain.saturation
        current = domain.loop_state

        # State transitions with hysteresis
        if current == LoopState.DIVERGENT:
            if self.z_level >= domain.z_origin and s >= 0.05:
                return LoopState.CONVERGING
        elif current == LoopState.CONVERGING:
            if s < 0.02:
                return LoopState.DIVERGENT
            elif s >= 0.50:
                return LoopState.CRITICAL
        elif current == LoopState.CRITICAL:
            if s < 0.45:
                return LoopState.CONVERGING
            elif s >= 0.95:
                return LoopState.CLOSED
        elif current == LoopState.CLOSED:
            if s < 0.90:
                return LoopState.CRITICAL

        return current

    def _composite_saturation(self) -> float:
        """Compute weighted composite saturation across all domains."""
        return sum(d.weight * d.saturation for d in self._domains)

    # =========================================================================
    # Cell Operations
    # =========================================================================

    def _domain_to_seal(self, domain_index: int) -> int:
        """Map domain index (0-6) to seal number (1-7)."""
        return domain_index + 1

    def _seal_to_domain(self, seal: int) -> int:
        """Map seal number (1-7) to domain index (0-6)."""
        return seal - 1

    def _interference_to_cell(self, i: int, j: int) -> Tuple[int, int]:
        """Map interference node (i,j) to cell (seal, face) via Fano structure."""
        p_i, p_j = i + 1, j + 1  # Convert to 1-indexed points
        for line in FANO_LINES:
            if p_i in line and p_j in line:
                third = [p for p in line if p not in (p_i, p_j)][0]
                pts = sorted(line)
                if (p_i, p_j) == (pts[0], pts[1]):
                    face = 0
                elif (p_i, p_j) == (pts[0], pts[2]):
                    face = 1
                else:
                    face = 2
                return (third, face)
        return (4, 1)  # Fallback

    def _compute_coherence(self) -> float:
        """Compute Kaelhedron coherence as order parameter."""
        total = 0.0 + 0.0j
        for cell in self._cells:
            phase = (cell.seal_index - 1) * TAU / 7 + cell.face_index * TAU / 21
            total += cell.activation * (math.cos(phase) + 1j * math.sin(phase))
        return abs(total) / 21

    # =========================================================================
    # Polarity Operations
    # =========================================================================

    def inject_polarity(self, p1: int, p2: int) -> Dict[str, Any]:
        """
        Inject two points into the polarity loop (forward polarity).

        This triggers the forward polarity (positive arc) and gates coherence
        until the phase delay elapses and backward polarity is triggered.

        Args:
            p1: First Fano point (1-7, maps to domain)
            p2: Second Fano point (1-7, maps to domain)

        Returns:
            Dictionary with the computed Fano line
        """
        line = self.polarity_loop.forward(p1, p2)
        self._polarity_phase = PolarityPhase.FORWARD_TRIGGERED
        self._forward_points = (p1, p2)
        self._forward_line = line
        self._coherence_point = None

        return {"line": line, "phase": self._polarity_phase.value}

    def release_polarity(
        self, line_a: Tuple[int, int, int], line_b: Tuple[int, int, int]
    ) -> Dict[str, Any]:
        """
        Release coherence via backward polarity.

        This triggers the backward polarity (negative arc). If the phase delay
        has elapsed, coherence is released and the intersection point is used
        to apply a permutation to the Kaelhedron cells.

        Args:
            line_a: First Fano line
            line_b: Second Fano line

        Returns:
            Dictionary with coherence status and intersection point
        """
        result = self.polarity_loop.backward(line_a, line_b)

        if result["coherence"]:
            self._polarity_phase = PolarityPhase.COHERENCE_RELEASED
            self._coherence_point = result["point"]

            # Apply permutation at the coherence point
            self._apply_coherence_permutation(result["point"])
        else:
            self._polarity_phase = PolarityPhase.GATED

        return {
            "coherence": result["coherence"],
            "point": result["point"],
            "remaining": result["remaining"],
            "phase": self._polarity_phase.value,
        }

    def _apply_coherence_permutation(self, point: int) -> None:
        """Apply permutation at the coherence point."""
        # The identity permutation at the point - this is the "trivial" case
        # In a more complex implementation, this could apply non-trivial PSL(3,2)
        # automorphisms based on the polarity state
        seal = point
        for cell in self._cells:
            if cell.seal_index == seal:
                # Boost activation for cells at the coherence point
                cell.activation = min(1.0, cell.activation + 0.1)
                cell.kappa = self._kaelhedron_coherence

    # =========================================================================
    # K-Formation Detection
    # =========================================================================

    def _detect_k_formation(self) -> Dict[str, Any]:
        """Detect K-Formation status based on coherence, recursion, and charge."""
        eta = self._kaelhedron_coherence
        R = sum(1 for d in self._domains if d.loop_state == LoopState.CLOSED)
        Q = self._topological_charge

        coh_met = eta > PHI_INV
        rec_met = R >= 7
        chg_met = Q != 0

        if coh_met and rec_met and chg_met:
            self._k_formation_status = KFormationStatus.FORMED
            self._k_formation_progress = 1.0
        elif coh_met and rec_met:
            self._k_formation_status = KFormationStatus.THRESHOLD
            self._k_formation_progress = 0.8
        elif eta > 0.5:
            self._k_formation_status = KFormationStatus.APPROACHING
            self._k_formation_progress = eta
        else:
            self._k_formation_status = KFormationStatus.INACTIVE
            self._k_formation_progress = eta / PHI_INV if PHI_INV > 0 else 0

        result = {
            "coherence": eta,
            "threshold": PHI_INV,
            "coherence_met": coh_met,
            "recursion": R,
            "recursion_met": rec_met,
            "charge": Q,
            "charge_met": chg_met,
            "status": self._k_formation_status.value,
            "progress": self._k_formation_progress,
            "K_FORMED": coh_met and rec_met and chg_met,
        }

        if result["K_FORMED"]:
            for cb in self._on_k_formation_callbacks:
                cb(result)

        return result

    # =========================================================================
    # Main Step Function
    # =========================================================================

    def step(self, dt: float = 0.01) -> UnifiedSystemState:
        """
        Advance the orchestrator by one timestep.

        This is the main integration point that:
        1. Updates all domain saturations
        2. Propagates domain state to Kaelhedron cells
        3. Computes interference contributions
        4. Updates Luminahedron divergence
        5. Computes coherence and K-Formation
        6. Publishes unified state

        Args:
            dt: Time delta

        Returns:
            Current unified system state
        """
        self.time += dt

        # =====================================================================
        # Forward Polarity Arc: Scalar -> Kaelhedron
        # =====================================================================

        # Update domain saturations
        for domain in self._domains:
            domain.saturation = self._compute_saturation(domain)
            domain.phase = (domain.phase + 0.1 * dt) % TAU
            domain.loop_state = self._update_loop_state(domain)

            # Propagate to Kaelhedron cells
            seal = self._domain_to_seal(domain.index)
            for cell in self._cells:
                if cell.seal_index == seal:
                    # Weight by face position
                    face_weights = [0.8, 1.0, 0.9]
                    cell.activation = domain.saturation * face_weights[cell.face_index]

        # Compute interference contributions (21 nodes)
        for i in range(7):
            for j in range(i + 1, 7):
                d_i, d_j = self._domains[i], self._domains[j]
                interference = d_i.saturation * d_j.saturation * math.cos(d_i.phase - d_j.phase)
                seal, face = self._interference_to_cell(i, j)
                for cell in self._cells:
                    if cell.seal_index == seal and cell.face_index == face:
                        cell.activation = min(1.0, cell.activation + 0.1 * interference)

        # Update Kaelhedron coherence
        self._kaelhedron_coherence = self._compute_coherence()
        self._kaelhedron_phase = (self._kaelhedron_phase + PHI_INV * dt) % TAU

        # =====================================================================
        # Backward Polarity Arc: Kaelhedron -> Scalar
        # =====================================================================

        # Luminahedron evolves based on Kaelhedron field
        kappa_field = self._kaelhedron_coherence * (1 - self._luminahedron_divergence)
        d_div = (1 - self._luminahedron_divergence) * 0.1 - kappa_field * 0.05
        self._luminahedron_divergence = max(0, min(1, self._luminahedron_divergence + d_div * dt))

        # Gauge coupling
        self._gauge_coupling = (kappa_field + self._luminahedron_divergence * 0.5) / 2

        # Compute topological charge
        closed = sum(1 for d in self._domains if d.loop_state == LoopState.CLOSED)
        divergent = sum(1 for d in self._domains if d.loop_state == LoopState.DIVERGENT)
        self._topological_charge = closed - divergent

        # Detect K-Formation
        k_result = self._detect_k_formation()

        # =====================================================================
        # Build Unified State
        # =====================================================================

        # Count loop states
        loops_closed = sum(1 for d in self._domains if d.loop_state == LoopState.CLOSED)
        loops_critical = sum(1 for d in self._domains if d.loop_state == LoopState.CRITICAL)
        loops_converging = sum(1 for d in self._domains if d.loop_state == LoopState.CONVERGING)
        loops_divergent = sum(1 for d in self._domains if d.loop_state == LoopState.DIVERGENT)

        # Build domain snapshots
        domain_snapshots = [
            DomainSnapshot(
                domain_index=d.index,
                name=d.name,
                saturation=d.saturation,
                loop_state=d.loop_state,
                phase=d.phase,
            )
            for d in self._domains
        ]

        # Build cell snapshots
        cell_snapshots = [
            CellSnapshot(
                seal_index=c.seal_index,
                face_index=c.face_index,
                label=c.label,
                theta=c.theta,
                kappa=c.kappa,
                activation=c.activation,
            )
            for c in self._cells
        ]

        # Build polarity snapshot
        gate_remaining = 0.0
        if self.polarity_loop.state:
            elapsed = time.time() - self.polarity_loop.state.start_time
            gate_remaining = max(0, self.polarity_loop.state.delay - elapsed)

        polarity_snapshot = PolaritySnapshot(
            phase=self._polarity_phase,
            forward_points=self._forward_points,
            forward_line=self._forward_line,
            gate_remaining=gate_remaining,
            coherence_point=self._coherence_point,
        )

        # Build unified state
        state = UnifiedSystemState(
            timestamp=time.time(),
            kappa=self.z_level,
            theta=self._kaelhedron_phase,
            recursion_depth=max(1, loops_closed),
            charge=self._topological_charge,
            kaelhedron_coherence=self._kaelhedron_coherence,
            kaelhedron_phase=self._kaelhedron_phase,
            luminahedron_divergence=self._luminahedron_divergence,
            gauge_coupling=self._gauge_coupling,
            polaric_balance=self._luminahedron_divergence / (self._kaelhedron_coherence + self._luminahedron_divergence + 1e-10),
            k_formation_status=self._k_formation_status,
            k_formation_progress=self._k_formation_progress,
            polarity_phase=self._polarity_phase,
            loops_closed=loops_closed,
            loops_critical=loops_critical,
            loops_converging=loops_converging,
            loops_divergent=loops_divergent,
            domains=domain_snapshots,
            cells=cell_snapshots,
            polarity=polarity_snapshot,
        )

        # Publish to registry
        self._registry.publish(state)

        # Fire coherence callbacks if threshold crossed
        if state.is_coherent:
            for cb in self._on_coherence_callbacks:
                cb(state)

        return state

    # =========================================================================
    # Configuration
    # =========================================================================

    def set_z_level(self, z: float) -> None:
        """Set the z-level (elevation in consciousness space)."""
        self.z_level = max(0, min(1, z))

    def set_topological_charge(self, Q: int) -> None:
        """Set the topological charge manually."""
        self._topological_charge = Q

    def on_coherence(self, callback: Callable[[UnifiedSystemState], None]) -> None:
        """Register callback for coherence threshold crossing."""
        self._on_coherence_callbacks.append(callback)

    def on_k_formation(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """Register callback for K-Formation events."""
        self._on_k_formation_callbacks.append(callback)

    # =========================================================================
    # Visualization Data
    # =========================================================================

    def get_visualization_data(self) -> Dict[str, Any]:
        """Get complete visualization bundle for UI/WebSocket."""
        return {
            "z_level": self.z_level,
            "time": self.time,
            "domains": [
                {
                    "index": d.index,
                    "name": d.name,
                    "saturation": d.saturation,
                    "loop_state": d.loop_state.value,
                    "phase": d.phase,
                }
                for d in self._domains
            ],
            "cells": [
                {
                    "seal": c.seal_index,
                    "face": c.face_index,
                    "label": c.label,
                    "activation": c.activation,
                }
                for c in self._cells
            ],
            "kaelhedron": {
                "coherence": self._kaelhedron_coherence,
                "phase": self._kaelhedron_phase,
            },
            "luminahedron": {
                "divergence": self._luminahedron_divergence,
                "coupling": self._gauge_coupling,
            },
            "k_formation": self._detect_k_formation(),
            "polarity": {
                "phase": self._polarity_phase.value,
                "forward_points": self._forward_points,
                "forward_line": self._forward_line,
                "coherence_point": self._coherence_point,
            },
        }
