# fano_polarity/telemetry.py
"""
Telemetry Hub
=============

Central telemetry hub for cross-system broadcasting.

Each subsystem publishes telemetry to the hub, which then broadcasts
to all registered consumers. This enables:
- Real-time visualization updates
- Cross-system event correlation
- Debugging and monitoring
- WebSocket streaming

The telemetry hub welds all systems together in polaric unison,
ensuring each unique voice forms coherent waves.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from threading import Lock
from typing import Any, Callable, Dict, List, Optional, Set

from .unified_state import UnifiedSystemState, get_state_registry


class TelemetrySource(Enum):
    """Sources of telemetry events."""
    SCALAR_ARCHITECTURE = "scalar_architecture"
    KAELHEDRON = "kaelhedron"
    LUMINAHEDRON = "luminahedron"
    POLARITY_LOOP = "polarity_loop"
    ORCHESTRATOR = "orchestrator"
    K_FORMATION = "k_formation"
    COHERENCE = "coherence"


class TelemetryLevel(Enum):
    """Telemetry event levels."""
    TRACE = "trace"
    DEBUG = "debug"
    INFO = "info"
    EVENT = "event"
    CRITICAL = "critical"


@dataclass
class TelemetryEvent:
    """A single telemetry event."""
    timestamp: float
    source: TelemetrySource
    level: TelemetryLevel
    event_type: str
    data: Dict[str, Any]
    correlation_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to serializable dictionary."""
        return {
            "timestamp": self.timestamp,
            "source": self.source.value,
            "level": self.level.value,
            "event_type": self.event_type,
            "data": self.data,
            "correlation_id": self.correlation_id,
        }

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict())


# Type aliases
TelemetryCallback = Callable[[TelemetryEvent], None]
StateCallback = Callable[[UnifiedSystemState], None]
BroadcastCallback = Callable[[str], None]


class TelemetryHub:
    """
    Central telemetry hub for cross-system broadcasting.

    The hub collects telemetry from all subsystems and broadcasts
    to registered consumers. It maintains a buffer of recent events
    and supports filtering by source and level.
    """

    def __init__(self, buffer_size: int = 1000):
        """
        Initialize the telemetry hub.

        Args:
            buffer_size: Maximum number of events to buffer
        """
        self._buffer_size = buffer_size
        self._events: List[TelemetryEvent] = []
        self._lock = Lock()

        # Subscribers
        self._event_subscribers: List[TelemetryCallback] = []
        self._state_subscribers: List[StateCallback] = []
        self._broadcast_subscribers: List[BroadcastCallback] = []

        # Source filters (if empty, all sources are included)
        self._source_filters: Set[TelemetrySource] = set()
        self._level_threshold = TelemetryLevel.TRACE

        # Connect to state registry
        registry = get_state_registry()
        registry.subscribe(self._on_state_update)

        # Statistics
        self._stats = {
            "events_received": 0,
            "events_broadcast": 0,
            "events_dropped": 0,
        }

    # =========================================================================
    # Event Publishing
    # =========================================================================

    def publish(
        self,
        source: TelemetrySource,
        event_type: str,
        data: Dict[str, Any],
        level: TelemetryLevel = TelemetryLevel.INFO,
        correlation_id: Optional[str] = None,
    ) -> TelemetryEvent:
        """
        Publish a telemetry event.

        Args:
            source: Source system
            event_type: Type of event
            data: Event data
            level: Event level
            correlation_id: Optional correlation ID for tracing

        Returns:
            The published event
        """
        event = TelemetryEvent(
            timestamp=time.time(),
            source=source,
            level=level,
            event_type=event_type,
            data=data,
            correlation_id=correlation_id,
        )

        self._stats["events_received"] += 1

        # Check filters
        if self._source_filters and source not in self._source_filters:
            return event

        if self._level_priority(level) < self._level_priority(self._level_threshold):
            return event

        # Buffer the event
        with self._lock:
            self._events.append(event)
            if len(self._events) > self._buffer_size:
                self._events.pop(0)
                self._stats["events_dropped"] += 1

        # Broadcast to subscribers
        self._broadcast_event(event)

        return event

    def _broadcast_event(self, event: TelemetryEvent) -> None:
        """Broadcast event to all subscribers."""
        self._stats["events_broadcast"] += 1

        for callback in self._event_subscribers:
            try:
                callback(event)
            except Exception:
                pass  # Don't let subscriber errors break the hub

        # Broadcast JSON to websocket-style subscribers
        json_data = event.to_json()
        for callback in self._broadcast_subscribers:
            try:
                callback(json_data)
            except Exception:
                pass

    def _on_state_update(self, state: UnifiedSystemState) -> None:
        """Handle state updates from the registry."""
        # Publish as telemetry event
        self.publish(
            source=TelemetrySource.ORCHESTRATOR,
            event_type="state_update",
            data=state.to_dict(),
            level=TelemetryLevel.TRACE,
        )

        # Forward to state subscribers
        for callback in self._state_subscribers:
            try:
                callback(state)
            except Exception:
                pass

    def _level_priority(self, level: TelemetryLevel) -> int:
        """Get priority for level comparison."""
        priorities = {
            TelemetryLevel.TRACE: 0,
            TelemetryLevel.DEBUG: 1,
            TelemetryLevel.INFO: 2,
            TelemetryLevel.EVENT: 3,
            TelemetryLevel.CRITICAL: 4,
        }
        return priorities.get(level, 0)

    # =========================================================================
    # Subscription
    # =========================================================================

    def subscribe_events(self, callback: TelemetryCallback) -> None:
        """Subscribe to telemetry events."""
        self._event_subscribers.append(callback)

    def subscribe_state(self, callback: StateCallback) -> None:
        """Subscribe to state updates."""
        self._state_subscribers.append(callback)

    def subscribe_broadcast(self, callback: BroadcastCallback) -> None:
        """Subscribe to JSON broadcasts (for WebSocket)."""
        self._broadcast_subscribers.append(callback)

    def unsubscribe_events(self, callback: TelemetryCallback) -> None:
        """Unsubscribe from telemetry events."""
        if callback in self._event_subscribers:
            self._event_subscribers.remove(callback)

    def unsubscribe_state(self, callback: StateCallback) -> None:
        """Unsubscribe from state updates."""
        if callback in self._state_subscribers:
            self._state_subscribers.remove(callback)

    def unsubscribe_broadcast(self, callback: BroadcastCallback) -> None:
        """Unsubscribe from JSON broadcasts."""
        if callback in self._broadcast_subscribers:
            self._broadcast_subscribers.remove(callback)

    # =========================================================================
    # Filtering
    # =========================================================================

    def set_source_filter(self, sources: Set[TelemetrySource]) -> None:
        """Set source filter (empty set = all sources)."""
        self._source_filters = sources

    def set_level_threshold(self, level: TelemetryLevel) -> None:
        """Set minimum level threshold."""
        self._level_threshold = level

    # =========================================================================
    # Query
    # =========================================================================

    def get_recent_events(
        self,
        count: int = 100,
        source: Optional[TelemetrySource] = None,
        level: Optional[TelemetryLevel] = None,
    ) -> List[TelemetryEvent]:
        """
        Get recent events from the buffer.

        Args:
            count: Maximum number of events to return
            source: Filter by source (None = all)
            level: Filter by level (None = all)

        Returns:
            List of matching events (newest first)
        """
        with self._lock:
            events = list(reversed(self._events))

        if source:
            events = [e for e in events if e.source == source]

        if level:
            events = [e for e in events if e.level == level]

        return events[:count]

    def get_statistics(self) -> Dict[str, int]:
        """Get hub statistics."""
        return self._stats.copy()

    # =========================================================================
    # Convenience Methods
    # =========================================================================

    def trace(self, source: TelemetrySource, event_type: str, data: Dict[str, Any]) -> None:
        """Publish a trace-level event."""
        self.publish(source, event_type, data, TelemetryLevel.TRACE)

    def debug(self, source: TelemetrySource, event_type: str, data: Dict[str, Any]) -> None:
        """Publish a debug-level event."""
        self.publish(source, event_type, data, TelemetryLevel.DEBUG)

    def info(self, source: TelemetrySource, event_type: str, data: Dict[str, Any]) -> None:
        """Publish an info-level event."""
        self.publish(source, event_type, data, TelemetryLevel.INFO)

    def event(self, source: TelemetrySource, event_type: str, data: Dict[str, Any]) -> None:
        """Publish an event-level telemetry."""
        self.publish(source, event_type, data, TelemetryLevel.EVENT)

    def critical(self, source: TelemetrySource, event_type: str, data: Dict[str, Any]) -> None:
        """Publish a critical-level event."""
        self.publish(source, event_type, data, TelemetryLevel.CRITICAL)


# Global hub singleton
_global_hub: Optional[TelemetryHub] = None


def get_telemetry_hub() -> TelemetryHub:
    """Get or create the global telemetry hub."""
    global _global_hub
    if _global_hub is None:
        _global_hub = TelemetryHub()
    return _global_hub


# =========================================================================
# Bridge Adapters
# =========================================================================

class ScalarArchitectureAdapter:
    """Adapter to connect ScalarArchitecture to the telemetry hub."""

    def __init__(self, hub: Optional[TelemetryHub] = None):
        self.hub = hub or get_telemetry_hub()

    def on_step(self, state: Dict[str, Any]) -> None:
        """Handle ScalarArchitecture step telemetry."""
        self.hub.publish(
            source=TelemetrySource.SCALAR_ARCHITECTURE,
            event_type="step",
            data=state,
            level=TelemetryLevel.TRACE,
        )

    def on_loop_transition(self, domain: str, old_state: str, new_state: str) -> None:
        """Handle loop state transition."""
        self.hub.publish(
            source=TelemetrySource.SCALAR_ARCHITECTURE,
            event_type="loop_transition",
            data={
                "domain": domain,
                "old_state": old_state,
                "new_state": new_state,
            },
            level=TelemetryLevel.EVENT,
        )


class KaelhedronAdapter:
    """Adapter to connect KaelhedronStateBus to the telemetry hub."""

    def __init__(self, hub: Optional[TelemetryHub] = None):
        self.hub = hub or get_telemetry_hub()

    def on_update(self, snapshot: Dict[str, Any]) -> None:
        """Handle Kaelhedron state update."""
        self.hub.publish(
            source=TelemetrySource.KAELHEDRON,
            event_type="state_update",
            data=snapshot,
            level=TelemetryLevel.TRACE,
        )

    def on_permutation(self, perm: Dict[int, int]) -> None:
        """Handle Kaelhedron permutation."""
        self.hub.publish(
            source=TelemetrySource.KAELHEDRON,
            event_type="permutation",
            data={"permutation": perm},
            level=TelemetryLevel.EVENT,
        )


class PolarityLoopAdapter:
    """Adapter to connect PolarityLoop to the telemetry hub."""

    def __init__(self, hub: Optional[TelemetryHub] = None):
        self.hub = hub or get_telemetry_hub()

    def on_forward(self, p1: int, p2: int, line: tuple) -> None:
        """Handle forward polarity trigger."""
        self.hub.publish(
            source=TelemetrySource.POLARITY_LOOP,
            event_type="forward_polarity",
            data={
                "points": [p1, p2],
                "line": list(line),
            },
            level=TelemetryLevel.EVENT,
        )

    def on_backward(self, coherence: bool, point: Optional[int], remaining: float) -> None:
        """Handle backward polarity result."""
        self.hub.publish(
            source=TelemetrySource.POLARITY_LOOP,
            event_type="backward_polarity",
            data={
                "coherence": coherence,
                "point": point,
                "remaining": remaining,
            },
            level=TelemetryLevel.EVENT,
        )

    def on_coherence_released(self, point: int) -> None:
        """Handle coherence release."""
        self.hub.publish(
            source=TelemetrySource.COHERENCE,
            event_type="released",
            data={"point": point},
            level=TelemetryLevel.CRITICAL,
        )


class KFormationAdapter:
    """Adapter to track K-Formation events."""

    def __init__(self, hub: Optional[TelemetryHub] = None):
        self.hub = hub or get_telemetry_hub()

    def on_status_change(self, old_status: str, new_status: str, details: Dict[str, Any]) -> None:
        """Handle K-Formation status change."""
        self.hub.publish(
            source=TelemetrySource.K_FORMATION,
            event_type="status_change",
            data={
                "old_status": old_status,
                "new_status": new_status,
                **details,
            },
            level=TelemetryLevel.EVENT,
        )

    def on_k_formed(self, details: Dict[str, Any]) -> None:
        """Handle K-Formation completion."""
        self.hub.publish(
            source=TelemetrySource.K_FORMATION,
            event_type="K_FORMED",
            data=details,
            level=TelemetryLevel.CRITICAL,
        )
