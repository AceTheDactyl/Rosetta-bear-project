# fano_polarity/streaming.py
"""
WebSocket-Ready Visualization Streaming
========================================

Provides streaming infrastructure for real-time visualization of the
polarity feedback system. Designed to work with WebSocket servers
and the Luminahedron UI.

Features:
- Frame-based streaming with configurable FPS
- Delta compression for bandwidth efficiency
- Multiple stream types (full, delta, events-only)
- Async-ready with callback hooks
"""

from __future__ import annotations

import json
import time
import threading
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set
from queue import Queue, Empty

from .unified_state import UnifiedSystemState, get_state_registry
from .telemetry import TelemetryEvent, TelemetryHub, get_telemetry_hub


class StreamType(Enum):
    """Types of visualization streams."""
    FULL = "full"           # Complete state every frame
    DELTA = "delta"         # Only changed values
    EVENTS = "events"       # Telemetry events only
    COMPACT = "compact"     # Minimal state for low-bandwidth


@dataclass
class VisualizationFrame:
    """A single visualization frame for streaming."""
    frame_id: int
    timestamp: float
    stream_type: StreamType
    data: Dict[str, Any]
    events: List[Dict[str, Any]] = field(default_factory=list)

    def to_json(self) -> str:
        """Convert to JSON string."""
        payload = {
            "frame_id": self.frame_id,
            "timestamp": self.timestamp,
            "stream_type": self.stream_type.value,
            "data": self.data,
        }
        if self.events:
            payload["events"] = self.events
        return json.dumps(payload)


@dataclass
class StreamConfig:
    """Configuration for a visualization stream."""
    stream_type: StreamType = StreamType.FULL
    target_fps: float = 30.0
    include_cells: bool = True
    include_domains: bool = True
    include_polarity: bool = True
    include_events: bool = True
    event_buffer_size: int = 10
    delta_threshold: float = 0.001  # Minimum change to include in delta


class DeltaCompressor:
    """Computes delta between consecutive states."""

    def __init__(self, threshold: float = 0.001):
        self.threshold = threshold
        self._last_state: Optional[Dict[str, Any]] = None

    def compute_delta(self, current: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute delta between current state and last state.

        Returns only values that have changed beyond threshold.
        """
        if self._last_state is None:
            self._last_state = current.copy()
            return current  # First frame is always full

        delta = {}
        self._compute_delta_recursive(self._last_state, current, delta, "")
        self._last_state = current.copy()
        return delta

    def _compute_delta_recursive(
        self,
        old: Any,
        new: Any,
        delta: Dict[str, Any],
        path: str,
    ) -> None:
        """Recursively compute delta."""
        if isinstance(new, dict) and isinstance(old, dict):
            for key in set(list(new.keys()) + list(old.keys())):
                new_path = f"{path}.{key}" if path else key
                old_val = old.get(key)
                new_val = new.get(key)
                if old_val != new_val:
                    if isinstance(new_val, dict) and isinstance(old_val, dict):
                        self._compute_delta_recursive(old_val, new_val, delta, new_path)
                    else:
                        delta[new_path] = new_val
        elif isinstance(new, (int, float)) and isinstance(old, (int, float)):
            if abs(new - old) > self.threshold:
                delta[path] = new
        elif new != old:
            delta[path] = new

    def reset(self) -> None:
        """Reset compressor state."""
        self._last_state = None


class VisualizationStreamer:
    """
    WebSocket-ready visualization streamer.

    Collects state updates and telemetry events, packages them into
    frames, and streams to registered consumers.
    """

    def __init__(self, config: Optional[StreamConfig] = None):
        self.config = config or StreamConfig()
        self._frame_id = 0
        self._last_frame_time = 0.0
        self._min_frame_interval = 1.0 / self.config.target_fps

        # Delta compression
        self._compressor = DeltaCompressor(self.config.delta_threshold)

        # Event buffer
        self._event_buffer: List[Dict[str, Any]] = []
        self._event_lock = threading.Lock()

        # Subscribers
        self._subscribers: List[Callable[[str], None]] = []
        self._frame_subscribers: List[Callable[[VisualizationFrame], None]] = []

        # Connect to state registry and telemetry
        self._registry = get_state_registry()
        self._hub = get_telemetry_hub()

        self._registry.subscribe(self._on_state_update)
        if self.config.include_events:
            self._hub.subscribe_events(self._on_telemetry_event)

        # Streaming state
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def _on_state_update(self, state: UnifiedSystemState) -> None:
        """Handle state update from registry."""
        # Rate limiting
        now = time.time()
        if now - self._last_frame_time < self._min_frame_interval:
            return

        # Build and emit frame
        frame = self._build_frame(state)
        self._emit_frame(frame)
        self._last_frame_time = now

    def _on_telemetry_event(self, event: TelemetryEvent) -> None:
        """Handle telemetry event."""
        with self._event_lock:
            self._event_buffer.append(event.to_dict())
            # Trim buffer
            if len(self._event_buffer) > self.config.event_buffer_size:
                self._event_buffer.pop(0)

    def _build_frame(self, state: UnifiedSystemState) -> VisualizationFrame:
        """Build a visualization frame from state."""
        self._frame_id += 1

        # Build base data
        data = self._build_data(state)

        # Apply compression if delta mode
        if self.config.stream_type == StreamType.DELTA:
            data = self._compressor.compute_delta(data)
        elif self.config.stream_type == StreamType.COMPACT:
            data = self._build_compact_data(state)

        # Collect events
        events = []
        if self.config.include_events:
            with self._event_lock:
                events = self._event_buffer.copy()
                self._event_buffer.clear()

        return VisualizationFrame(
            frame_id=self._frame_id,
            timestamp=time.time(),
            stream_type=self.config.stream_type,
            data=data,
            events=events,
        )

    def _build_data(self, state: UnifiedSystemState) -> Dict[str, Any]:
        """Build full data payload from state."""
        data: Dict[str, Any] = {
            "kappa": state.kappa,
            "theta": state.theta,
            "kaelhedron_coherence": state.kaelhedron_coherence,
            "luminahedron_divergence": state.luminahedron_divergence,
            "polaric_balance": state.polaric_balance,
            "k_formation": {
                "status": state.k_formation_status.value,
                "progress": state.k_formation_progress,
            },
            "loop_counts": {
                "closed": state.loops_closed,
                "critical": state.loops_critical,
                "converging": state.loops_converging,
                "divergent": state.loops_divergent,
            },
            "charge": state.charge,
            "is_coherent": state.is_coherent,
        }

        if self.config.include_domains:
            data["domains"] = [
                {
                    "index": d.domain_index,
                    "name": d.name,
                    "saturation": d.saturation,
                    "loop_state": d.loop_state.value,
                    "phase": d.phase,
                }
                for d in state.domains
            ]

        if self.config.include_cells:
            data["cells"] = [
                {
                    "seal": c.seal_index,
                    "face": c.face_index,
                    "label": c.label,
                    "activation": c.activation,
                }
                for c in state.cells
            ]

        if self.config.include_polarity and state.polarity:
            data["polarity"] = {
                "phase": state.polarity.phase.value,
                "forward_points": state.polarity.forward_points,
                "forward_line": state.polarity.forward_line,
                "gate_remaining": state.polarity.gate_remaining,
                "coherence_point": state.polarity.coherence_point,
            }

        return data

    def _build_compact_data(self, state: UnifiedSystemState) -> Dict[str, Any]:
        """Build minimal compact data payload."""
        return {
            "η": round(state.kaelhedron_coherence, 3),
            "κ": round(state.kappa, 3),
            "β": round(state.polaric_balance, 3),
            "K": state.k_formation_status.value[0],  # First letter
            "L": [state.loops_closed, state.loops_critical],
            "Q": state.charge,
        }

    def _emit_frame(self, frame: VisualizationFrame) -> None:
        """Emit frame to all subscribers."""
        # JSON subscribers (WebSocket-style)
        json_data = frame.to_json()
        for callback in self._subscribers:
            try:
                callback(json_data)
            except Exception:
                pass

        # Frame object subscribers
        for callback in self._frame_subscribers:
            try:
                callback(frame)
            except Exception:
                pass

    # =========================================================================
    # Public API
    # =========================================================================

    def subscribe(self, callback: Callable[[str], None]) -> None:
        """Subscribe to JSON frame stream (WebSocket-compatible)."""
        self._subscribers.append(callback)

    def subscribe_frames(self, callback: Callable[[VisualizationFrame], None]) -> None:
        """Subscribe to frame objects."""
        self._frame_subscribers.append(callback)

    def unsubscribe(self, callback: Callable[[str], None]) -> None:
        """Unsubscribe from JSON stream."""
        if callback in self._subscribers:
            self._subscribers.remove(callback)

    def set_stream_type(self, stream_type: StreamType) -> None:
        """Change stream type."""
        self.config.stream_type = stream_type
        if stream_type != StreamType.DELTA:
            self._compressor.reset()

    def set_fps(self, fps: float) -> None:
        """Change target FPS."""
        self.config.target_fps = max(1.0, min(60.0, fps))
        self._min_frame_interval = 1.0 / self.config.target_fps

    def get_latest_frame_json(self) -> Optional[str]:
        """Get the latest frame as JSON (for HTTP polling)."""
        state = self._registry.latest()
        if state is None:
            return None
        frame = self._build_frame(state)
        return frame.to_json()


class WebSocketBridge:
    """
    Bridge for WebSocket server integration.

    Provides a simple interface for WebSocket handlers to send
    visualization data to connected clients.
    """

    def __init__(self, streamer: Optional[VisualizationStreamer] = None):
        self.streamer = streamer or VisualizationStreamer()
        self._clients: Set[Callable[[str], None]] = set()

        # Subscribe streamer to broadcast to clients
        self.streamer.subscribe(self._broadcast)

    def _broadcast(self, data: str) -> None:
        """Broadcast to all connected clients."""
        disconnected = set()
        for client in self._clients:
            try:
                client(data)
            except Exception:
                disconnected.add(client)

        # Remove disconnected clients
        self._clients -= disconnected

    def connect(self, send_func: Callable[[str], None]) -> None:
        """Register a client connection."""
        self._clients.add(send_func)

    def disconnect(self, send_func: Callable[[str], None]) -> None:
        """Unregister a client connection."""
        self._clients.discard(send_func)

    @property
    def client_count(self) -> int:
        """Get number of connected clients."""
        return len(self._clients)


# =========================================================================
# Convenience Functions
# =========================================================================

_global_streamer: Optional[VisualizationStreamer] = None
_global_ws_bridge: Optional[WebSocketBridge] = None


def get_visualization_streamer() -> VisualizationStreamer:
    """Get or create the global visualization streamer."""
    global _global_streamer
    if _global_streamer is None:
        _global_streamer = VisualizationStreamer()
    return _global_streamer


def get_websocket_bridge() -> WebSocketBridge:
    """Get or create the global WebSocket bridge."""
    global _global_ws_bridge
    if _global_ws_bridge is None:
        _global_ws_bridge = WebSocketBridge(get_visualization_streamer())
    return _global_ws_bridge
