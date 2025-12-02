"""Scalar telemetry bridge feeding the Polaric service."""

from __future__ import annotations

from typing import Callable, Dict, Optional

from asymptotic_scalars.core import (
    AsymptoticScalarSystem,
    AsymptoticState,
    create_asymptotic_scalar_system,
)

from .core import ScalarArchitecture, ScalarArchitectureState


class ScalarSyncService:
    """Streams ScalarArchitecture + AsymptoticScalars telemetry."""

    def __init__(
        self,
        publisher: Optional[Callable[[Dict[str, object]], None]] = None,
        initial_z: float = 0.41,
    ) -> None:
        self._publisher = publisher or (lambda payload: None)
        self._last_arch_payload: Optional[Dict[str, object]] = None
        self._latest_payload: Dict[str, object] = {}

        self.architecture = ScalarArchitecture(
            initial_z=initial_z, telemetry_publisher=self._capture_arch_payload
        )
        self.asymptotic: AsymptoticScalarSystem = create_asymptotic_scalar_system()

    def _capture_arch_payload(self, payload: Dict[str, object]) -> None:
        self._last_arch_payload = payload

    def tick(self, dt: float = 0.05) -> Dict[str, object]:
        arch_state = self.architecture.step(dt)
        asym_state = self.asymptotic.update(arch_state.z_level, dt)
        payload = self._compose_payload(arch_state, asym_state)
        self._latest_payload = payload
        self._publisher(payload)
        return payload

    def latest_payload(self) -> Dict[str, object]:
        return self._latest_payload

    def _compose_payload(
        self,
        arch_state: ScalarArchitectureState,
        asym_state: AsymptoticState,
    ) -> Dict[str, object]:
        arch_payload = self._last_arch_payload or {
            "kappa": arch_state.z_level,
            "theta": arch_state.helix.theta,
            "recursion_depth": 1,
            "charge": 0,
            "domains": {},
            "loop_counts": {},
        }
        payload = dict(arch_payload)
        payload.update(
            {
                "loop_state": asym_state.loop_state.value,
                "loop_closure_confidence": asym_state.loop_closure_confidence,
                "asymptotic": asym_state.to_dict(),
            }
        )
        return payload


__all__ = ["ScalarSyncService"]
