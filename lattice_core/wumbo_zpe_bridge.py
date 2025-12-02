#!/usr/bin/env python3
"""
WUMBO-ZPE BRIDGE - Field-Inference Message Passing
==================================================

Bridges WUMBO κ-λ field operations to ZPE Fano plane inference.
Implements bidirectional message passing:
    - Forward: κ-field → Fano nodes (evidence injection)
    - Backward: Fano precision → λ-field coupling (modulation)

Physics Foundation:
    - κ-field observations map to odd Fano nodes (1, 3, 5)
    - λ-field observations map to even Fano nodes (0, 2, 4, 6)
    - Precision updates modulate coupling strength

Integration with LIMNUS:
    - L: Load field states
    - I: Inject to Fano nodes
    - M: Message passing inference
    - N: Normalize beliefs
    - U: Update field coupling
    - S: Synchronize systems

Signature: Δ|wumbo-zpe-bridge|z0.90|message-passing|Ω
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import math

# Golden ratio constants
PHI = (1 + math.sqrt(5)) / 2
PHI_INV = PHI - 1
TAU = 2 * math.pi

# Fano node assignments
KAPPA_NODES = [1, 3, 5]     # Structure nodes (odd)
LAMBDA_NODES = [0, 2, 4, 6]  # Navigation nodes (even)


@dataclass
class FieldToFanoMessage:
    """
    Message from WUMBO field to Fano node.

    Carries field observation as evidence for belief update.
    Precision weights the update strength.
    """
    source_field: str                   # "kappa" or "lambda"
    target_node: int                    # Fano node index (0-6)
    observation: float                  # Field value as observation
    precision: float                    # Confidence in observation
    phase: float                        # Field phase angle

    def to_dict(self) -> Dict[str, Any]:
        """Export message to dictionary."""
        return {
            "source": self.source_field,
            "target": self.target_node,
            "observation": self.observation,
            "precision": self.precision,
            "phase": self.phase,
        }


@dataclass
class FanoToFieldMessage:
    """
    Message from Fano node back to WUMBO field.

    Carries belief update for field coupling adjustment.
    Enables inference results to modulate dynamics.
    """
    source_node: int                    # Fano node index
    target_field: str                   # "kappa" or "lambda"
    belief_update: float                # Belief delta from neutral (0.5)
    precision_update: float             # New precision estimate

    def to_dict(self) -> Dict[str, Any]:
        """Export message to dictionary."""
        return {
            "source": self.source_node,
            "target": self.target_field,
            "belief_delta": self.belief_update,
            "precision": self.precision_update,
        }


@dataclass
class BridgeState:
    """
    Current state of the WUMBO-ZPE bridge.

    Tracks synchronization and message flow metrics.
    """
    phase_alignment: float = 0.0        # κ-λ phase alignment
    messages_forward: int = 0           # Field → Fano count
    messages_backward: int = 0          # Fano → Field count
    fano_convergence: float = 0.0       # Inference convergence
    last_cycle_metrics: Dict[str, float] = field(default_factory=dict)


class WumboZPEBridge:
    """
    Bidirectional bridge between WUMBO fields and ZPE Fano inference.

    Implements message passing protocol:
    1. field_to_fano(): Generate evidence messages from field state
    2. inject_to_fano(): Apply messages to Fano nodes
    3. run_inference(): Execute Fano belief propagation
    4. fano_to_field(): Generate coupling updates from beliefs
    5. apply_to_field(): Modulate field coupling

    Usage:
        bridge = WumboZPEBridge(wumbo_engine, zpe_engine)
        metrics = bridge.run_bridge_cycle(iterations=5)
    """

    def __init__(
        self,
        wumbo_engine: Any,              # WumboEngine instance
        zpe_engine: Any,                # ZeroPointEnergyEngine instance
    ):
        """
        Initialize bridge with WUMBO and ZPE engines.

        Args:
            wumbo_engine: WumboEngine managing κ-λ fields
            zpe_engine: ZeroPointEnergyEngine with Fano inference
        """
        self.wumbo = wumbo_engine
        self.zpe = zpe_engine
        self.fano = getattr(zpe_engine, 'fano_engine', None)

        self.state = BridgeState()
        self._message_history: List[Dict[str, Any]] = []

    def field_to_fano(self) -> List[FieldToFanoMessage]:
        """
        Generate messages from WUMBO fields to Fano nodes.

        Maps field state to evidence for belief propagation:
            - κ-field amplitude × cos(phase + offset) → odd nodes
            - λ-field amplitude × cos(phase + offset) → even nodes
        """
        messages = []

        # Get field states from WUMBO
        kappa = self._get_kappa_state()
        lambda_ = self._get_lambda_state()

        # Generate κ-field → odd node messages
        for i, node_idx in enumerate(KAPPA_NODES):
            obs = kappa['amplitude'] * math.cos(kappa['phase'] + i * PHI)
            messages.append(FieldToFanoMessage(
                source_field="kappa",
                target_node=node_idx,
                observation=obs,
                precision=kappa['amplitude'],
                phase=kappa['phase'],
            ))

        # Generate λ-field → even node messages
        for i, node_idx in enumerate(LAMBDA_NODES):
            obs = lambda_['amplitude'] * math.cos(lambda_['phase'] + i * PHI_INV)
            messages.append(FieldToFanoMessage(
                source_field="lambda",
                target_node=node_idx,
                observation=obs,
                precision=lambda_['amplitude'],
                phase=lambda_['phase'],
            ))

        self.state.messages_forward += len(messages)
        return messages

    def inject_to_fano(self, messages: List[FieldToFanoMessage]) -> None:
        """
        Apply field messages to Fano nodes.

        Updates node beliefs with precision-weighted evidence.
        """
        if self.fano is None:
            return

        for msg in messages:
            self._inject_single_message(msg)

    def _inject_single_message(self, msg: FieldToFanoMessage) -> None:
        """Inject single message into Fano node."""
        if self.fano is None:
            return

        try:
            node = self.fano.nodes[msg.target_node]

            # Precision-weighted belief update
            # Higher precision → stronger influence
            weight = msg.precision

            # Map observation to belief space [0, 1]
            obs_belief = 0.5 + 0.5 * msg.observation  # Normalize to [0, 1]
            obs_belief = max(0.0, min(1.0, obs_belief))

            # Update belief
            node.belief = (
                node.belief * (1 - weight) +
                obs_belief * weight
            )
        except (AttributeError, IndexError):
            pass

    def fano_to_field(self) -> List[FanoToFieldMessage]:
        """
        Generate messages from Fano nodes back to fields.

        Extracts belief updates for field coupling modulation.
        """
        messages = []

        if self.fano is None:
            return messages

        try:
            for node_idx, node in enumerate(self.fano.nodes):
                # Determine target field by node parity
                target = "kappa" if node_idx % 2 == 1 else "lambda"

                messages.append(FanoToFieldMessage(
                    source_node=node_idx,
                    target_field=target,
                    belief_update=node.belief - 0.5,  # Delta from neutral
                    precision_update=getattr(node, 'belief_precision', 1.0),
                ))
        except AttributeError:
            pass

        self.state.messages_backward += len(messages)
        return messages

    def apply_to_field(self, messages: List[FanoToFieldMessage]) -> None:
        """
        Apply Fano belief updates to WUMBO fields.

        Modulates field coupling based on inference results.
        """
        for msg in messages:
            self._apply_single_update(msg)

    def _apply_single_update(self, msg: FanoToFieldMessage) -> None:
        """Apply single update to target field."""
        try:
            if msg.target_field == "kappa":
                # Modulate κ-field coupling
                if hasattr(self.wumbo, 'kappa_field'):
                    delta = 1.0 + 0.1 * msg.belief_update * msg.precision_update
                    self.wumbo.kappa_field.coupling_strength *= delta
            else:
                # Modulate λ-field coupling
                if hasattr(self.wumbo, 'lambda_field'):
                    delta = 1.0 + 0.1 * msg.belief_update * msg.precision_update
                    self.wumbo.lambda_field.coupling_strength *= delta
        except AttributeError:
            pass

    def run_bridge_cycle(self, iterations: int = 1) -> Dict[str, float]:
        """
        Run complete bridge cycle: field → fano → inference → field.

        One cycle consists of:
            1. Generate field → Fano messages
            2. Inject messages into Fano nodes
            3. Run Fano belief propagation
            4. Generate Fano → field messages
            5. Apply coupling updates

        Args:
            iterations: Number of complete cycles

        Returns:
            Dictionary of synchronization metrics
        """
        for _ in range(iterations):
            # Forward pass
            forward_msgs = self.field_to_fano()
            self.inject_to_fano(forward_msgs)

            # Inference
            if self.fano is not None:
                try:
                    self.fano.run_inference(iterations=5)
                except (AttributeError, TypeError):
                    pass

            # Backward pass
            backward_msgs = self.fano_to_field()
            self.apply_to_field(backward_msgs)

        # Compute metrics
        metrics = self._compute_metrics()
        self.state.last_cycle_metrics = metrics

        return metrics

    def _compute_metrics(self) -> Dict[str, float]:
        """Compute synchronization metrics."""
        kappa = self._get_kappa_state()
        lambda_ = self._get_lambda_state()

        # Phase alignment (cos of phase difference)
        phase_alignment = math.cos(kappa['phase'] - lambda_['phase'])
        self.state.phase_alignment = phase_alignment

        # Fano convergence
        fano_conv = 0.0
        if self.fano is not None:
            try:
                fano_conv = self.fano.check_convergence()
            except (AttributeError, TypeError):
                pass
        self.state.fano_convergence = fano_conv

        return {
            "phase_alignment": phase_alignment,
            "kappa_amplitude": kappa['amplitude'],
            "lambda_amplitude": lambda_['amplitude'],
            "fano_convergence": fano_conv,
            "messages_forward": self.state.messages_forward,
            "messages_backward": self.state.messages_backward,
        }

    def _get_kappa_state(self) -> Dict[str, float]:
        """Get κ-field state from WUMBO."""
        try:
            kappa = self.wumbo.kappa_field
            return {
                'amplitude': getattr(kappa, 'amplitude', 1.0),
                'phase': getattr(kappa, 'phase', 0.0),
            }
        except AttributeError:
            return {'amplitude': 1.0, 'phase': 0.0}

    def _get_lambda_state(self) -> Dict[str, float]:
        """Get λ-field state from WUMBO."""
        try:
            lambda_ = self.wumbo.lambda_field
            return {
                'amplitude': getattr(lambda_, 'amplitude', 1.0),
                'phase': getattr(lambda_, 'phase', 0.0),
            }
        except AttributeError:
            return {'amplitude': 1.0, 'phase': 0.0}

    def reset(self) -> None:
        """Reset bridge state."""
        self.state = BridgeState()
        self._message_history.clear()

    def get_history(self) -> List[Dict[str, Any]]:
        """Return message history."""
        return self._message_history.copy()


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_wumbo_zpe_bridge(
    wumbo_engine: Any,
    zpe_engine: Any,
) -> WumboZPEBridge:
    """
    Create WUMBO-ZPE bridge instance.

    Args:
        wumbo_engine: WumboEngine with κ-λ fields
        zpe_engine: ZeroPointEnergyEngine with Fano inference
    """
    return WumboZPEBridge(wumbo_engine, zpe_engine)
