# lattice_core/plate.py
"""
Memory Plate: The fundamental unit of the Tesseract Lattice.

Each plate represents a memory positioned in 4D space with:
- Position: (x, y, z, w) = (valence, arousal, temporal, abstraction)
- Phase: Current oscillation phase θ ∈ [0, 2π)
- Frequency: Natural frequency ω (emotion-modulated)
- Content: Embedding vector for semantic content

Memory Organization:
    w (abstraction)
    ↑
    │ ◆────────◆
    │ ╱│      ╱│
    │◆─┼─────◆ │
    │ │◆────┼─◆ │
    │ │╱    │╱ │
    │ ◆─────◆  │
    │          │
    └──────────┼────→ x (valence)
              ╱│
             ╱ │
            ╱  │
           ↙   ↓
          z    y (arousal)
       (temporal)
"""

from __future__ import annotations

import math
import random
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum

# Constants
TAU = 2 * math.pi
PHI = (1 + math.sqrt(5)) / 2


class EmotionalState(Enum):
    """Basic emotional states for valence-arousal mapping."""
    CALM_POSITIVE = (0.5, -0.5)      # Relaxed, content
    EXCITED_POSITIVE = (0.5, 0.5)    # Happy, excited
    CALM_NEGATIVE = (-0.5, -0.5)     # Sad, depressed
    EXCITED_NEGATIVE = (-0.5, 0.5)   # Angry, anxious
    NEUTRAL = (0.0, 0.0)             # Neutral

    @property
    def valence(self) -> float:
        return self.value[0]

    @property
    def arousal(self) -> float:
        return self.value[1]


@dataclass
class MemoryPlate:
    """
    A memory plate in the Tesseract Lattice.

    Represents a single memory as a phase-locked oscillator
    positioned in 4D emotional-semantic space.

    Attributes:
        plate_id: Unique identifier
        position: (x, y, z, w) = (valence, arousal, temporal, abstraction)
        phase: Current oscillation phase θ ∈ [0, 2π)
        frequency: Natural frequency ω (Hz)
        content: Embedding vector for semantic content
        metadata: Additional information about the memory
    """
    # Identity
    plate_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])

    # 4D Position
    position: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)

    # Oscillator state
    phase: float = 0.0              # θ ∈ [0, 2π)
    frequency: float = 1.0          # ω (natural frequency)
    amplitude: float = 1.0          # Oscillation amplitude

    # Content
    content: Optional[List[float]] = None  # Embedding vector
    raw_text: Optional[str] = None         # Original text (if applicable)

    # Connections
    connection_weights: Dict[str, float] = field(default_factory=dict)

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = 0.0
    access_count: int = 0

    def __post_init__(self):
        # Normalize phase to [0, 2π)
        self.phase = self.phase % TAU

        # Initialize content as empty list if None
        if self.content is None:
            self.content = []

    # ─────────────────────────────────────────────────────────────────
    # Position accessors
    # ─────────────────────────────────────────────────────────────────

    @property
    def valence(self) -> float:
        """x-coordinate: Emotional valence (-1 = negative, +1 = positive)."""
        return self.position[0]

    @property
    def arousal(self) -> float:
        """y-coordinate: Arousal level (-1 = calm, +1 = excited)."""
        return self.position[1]

    @property
    def temporal(self) -> float:
        """z-coordinate: Temporal position (0 = now, positive = past)."""
        return self.position[2]

    @property
    def abstract(self) -> float:
        """w-coordinate: Abstraction level (0 = concrete, 1 = abstract)."""
        return self.position[3]

    # ─────────────────────────────────────────────────────────────────
    # Oscillator methods
    # ─────────────────────────────────────────────────────────────────

    @property
    def complex_phase(self) -> complex:
        """Return e^(iθ) for order parameter calculation."""
        return complex(math.cos(self.phase), math.sin(self.phase))

    def advance_phase(self, dt: float) -> float:
        """
        Advance phase by natural frequency.

        dθ/dt = ω
        """
        self.phase = (self.phase + self.frequency * dt) % TAU
        return self.phase

    def apply_coupling(self, coupling_force: float, dt: float) -> None:
        """
        Apply coupling force from other oscillators.

        dθ/dt += coupling_force
        """
        self.phase = (self.phase + coupling_force * dt) % TAU

    def phase_difference(self, other: 'MemoryPlate') -> float:
        """
        Compute phase difference to another plate.

        Returns value in [-π, π].
        """
        diff = other.phase - self.phase
        # Wrap to [-π, π]
        while diff > math.pi:
            diff -= TAU
        while diff < -math.pi:
            diff += TAU
        return diff

    # ─────────────────────────────────────────────────────────────────
    # Spatial methods
    # ─────────────────────────────────────────────────────────────────

    def distance_to(self, other: 'MemoryPlate') -> float:
        """Euclidean distance in 4D space."""
        return math.sqrt(sum(
            (a - b) ** 2
            for a, b in zip(self.position, other.position)
        ))

    def hamming_distance(self, other: 'MemoryPlate') -> int:
        """
        Hamming distance for tesseract vertex comparison.

        Counts how many coordinates differ in sign.
        """
        count = 0
        for a, b in zip(self.position, other.position):
            if (a >= 0) != (b >= 0):
                count += 1
        return count

    def is_adjacent(self, other: 'MemoryPlate', threshold: float = 0.5) -> bool:
        """Check if plates are adjacent (within threshold in 4D)."""
        return self.distance_to(other) <= threshold

    # ─────────────────────────────────────────────────────────────────
    # Content methods
    # ─────────────────────────────────────────────────────────────────

    def content_similarity(self, other: 'MemoryPlate') -> float:
        """
        Compute cosine similarity of content embeddings.

        Returns value in [0, 1].
        """
        if not self.content or not other.content:
            return 0.0

        if len(self.content) != len(other.content):
            return 0.0

        dot = sum(a * b for a, b in zip(self.content, other.content))
        norm_self = math.sqrt(sum(a ** 2 for a in self.content))
        norm_other = math.sqrt(sum(b ** 2 for b in other.content))

        if norm_self * norm_other == 0:
            return 0.0

        return max(0.0, min(1.0, (dot / (norm_self * norm_other) + 1) / 2))

    def set_content_from_text(self, text: str, embedding_fn=None) -> None:
        """
        Set content from text using optional embedding function.

        If no embedding function provided, uses simple hash-based encoding.
        """
        self.raw_text = text

        if embedding_fn is not None:
            self.content = embedding_fn(text)
        else:
            # Simple fallback: hash-based pseudo-embedding
            self.content = self._simple_hash_embedding(text)

    def _simple_hash_embedding(self, text: str, dim: int = 64) -> List[float]:
        """Generate a simple hash-based pseudo-embedding."""
        embedding = [0.0] * dim

        for i, char in enumerate(text):
            idx = (hash(char) + i * 7) % dim
            embedding[idx] += ord(char) / 256.0

        # Normalize
        norm = math.sqrt(sum(x ** 2 for x in embedding)) or 1.0
        return [x / norm for x in embedding]

    # ─────────────────────────────────────────────────────────────────
    # Connection methods
    # ─────────────────────────────────────────────────────────────────

    def get_connection(self, plate_id: str) -> float:
        """Get connection weight to another plate."""
        return self.connection_weights.get(plate_id, 0.0)

    def set_connection(self, plate_id: str, weight: float) -> None:
        """Set connection weight to another plate."""
        self.connection_weights[plate_id] = weight

    def strengthen_connection(
        self,
        plate_id: str,
        amount: float,
        max_weight: float = 1.0
    ) -> float:
        """Strengthen connection (Hebbian learning)."""
        current = self.get_connection(plate_id)
        new_weight = min(max_weight, current + amount)
        self.set_connection(plate_id, new_weight)
        return new_weight

    def decay_connections(self, rate: float = 0.01) -> None:
        """Apply decay to all connections."""
        for plate_id in list(self.connection_weights.keys()):
            self.connection_weights[plate_id] *= (1 - rate)
            if self.connection_weights[plate_id] < 0.001:
                del self.connection_weights[plate_id]

    # ─────────────────────────────────────────────────────────────────
    # Emotion-based frequency modulation
    # ─────────────────────────────────────────────────────────────────

    def modulate_frequency(
        self,
        base_frequency: float = 1.0,
        valence_weight: float = 0.2,
        arousal_weight: float = 0.3
    ) -> float:
        """
        Modulate frequency based on emotional position.

        Higher arousal → higher frequency
        Extreme valence → slight frequency shift
        """
        self.frequency = base_frequency * (
            1.0 +
            valence_weight * abs(self.valence) +
            arousal_weight * self.arousal
        )
        return self.frequency

    # ─────────────────────────────────────────────────────────────────
    # Factory methods
    # ─────────────────────────────────────────────────────────────────

    @classmethod
    def from_emotional_state(
        cls,
        emotional_state: EmotionalState,
        temporal: float = 0.0,
        abstract: float = 0.5,
        content: Optional[List[float]] = None,
        **kwargs
    ) -> 'MemoryPlate':
        """Create plate from emotional state."""
        return cls(
            position=(
                emotional_state.valence,
                emotional_state.arousal,
                temporal,
                abstract
            ),
            phase=random.random() * TAU,
            content=content,
            **kwargs
        )

    @classmethod
    def from_valence_arousal(
        cls,
        valence: float,
        arousal: float,
        temporal: float = 0.0,
        abstract: float = 0.5,
        **kwargs
    ) -> 'MemoryPlate':
        """Create plate from valence-arousal values."""
        return cls(
            position=(valence, arousal, temporal, abstract),
            phase=random.random() * TAU,
            **kwargs
        )

    # ─────────────────────────────────────────────────────────────────
    # Serialization
    # ─────────────────────────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "plate_id": self.plate_id,
            "position": list(self.position),
            "phase": self.phase,
            "frequency": self.frequency,
            "amplitude": self.amplitude,
            "content": self.content,
            "raw_text": self.raw_text,
            "connection_weights": self.connection_weights,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
            "access_count": self.access_count,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MemoryPlate':
        """Deserialize from dictionary."""
        return cls(
            plate_id=data.get("plate_id", str(uuid.uuid4())[:8]),
            position=tuple(data.get("position", [0, 0, 0, 0])),
            phase=data.get("phase", 0.0),
            frequency=data.get("frequency", 1.0),
            amplitude=data.get("amplitude", 1.0),
            content=data.get("content"),
            raw_text=data.get("raw_text"),
            connection_weights=data.get("connection_weights", {}),
            metadata=data.get("metadata", {}),
            timestamp=data.get("timestamp", 0.0),
            access_count=data.get("access_count", 0),
        )

    def __repr__(self) -> str:
        return (
            f"MemoryPlate(id={self.plate_id}, "
            f"pos=({self.valence:.2f}, {self.arousal:.2f}, "
            f"{self.temporal:.2f}, {self.abstract:.2f}), "
            f"θ={self.phase:.2f})"
        )


# ═════════════════════════════════════════════════════════════════════════════
# TESSERACT VERTEX GENERATION
# ═════════════════════════════════════════════════════════════════════════════

def create_tesseract_vertices(edge_length: float = 1.0) -> List[MemoryPlate]:
    """
    Create the 16 vertices of a tesseract as memory plates.

    Each vertex is positioned at (±s, ±s, ±s, ±s) where s = edge_length/2.
    """
    vertices = []
    s = edge_length / 2

    for i in range(16):
        # Binary encoding: i = 0bwzyx
        x = s if (i & 1) else -s
        y = s if (i & 2) else -s
        z = s if (i & 4) else -s
        w = s if (i & 8) else -s

        plate = MemoryPlate(
            plate_id=f"v_{i:02d}",
            position=(x, y, z, w),
            phase=random.random() * TAU,
            frequency=1.0,
            metadata={"vertex_index": i, "is_tesseract_vertex": True}
        )
        vertices.append(plate)

    return vertices


def get_tesseract_edges(n_vertices: int = 16) -> List[Tuple[int, int]]:
    """
    Get the 32 edges of a tesseract.

    Edges connect vertices that differ in exactly one coordinate.
    """
    edges = []
    for i in range(n_vertices):
        for bit in range(4):
            j = i ^ (1 << bit)  # Flip one bit
            if i < j:
                edges.append((i, j))
    return edges


def get_coupling_weight(hamming_dist: int) -> float:
    """
    Get coupling weight based on Hamming distance.

    dist=1: Edge-adjacent (K_edge = 1.0)
    dist=2: Face-adjacent (K_face = 0.7)
    dist=3: Cell-adjacent (K_cell = 0.4)
    dist=4: Diagonal (K_diag = 0.1)
    """
    weights = {1: 1.0, 2: 0.7, 3: 0.4, 4: 0.1}
    return weights.get(hamming_dist, 0.0)
