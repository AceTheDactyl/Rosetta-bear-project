# memory/memory_manager.py
"""
Memory Manager: High-Level API for Tesseract Lattice Memory
============================================================

Provides a simple interface for:
- Storing events (text → embedding → 4D position → plate)
- Querying memories (resonance-based retrieval)
- Consolidating (Hebbian learning)

Usage:
    manager = MemoryManager()
    manager.store_event("Had a great meeting today", valence=0.7, arousal=0.3)
    results = manager.query("work meetings")
    for result in results:
        print(f"{result.text} (score: {result.score:.2f})")
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from lattice_core.plate import MemoryPlate, EmotionalState
from lattice_core.tesseract_lattice_engine import (
    TesseractLatticeEngine,
    LatticeConfig,
    RetrievalResult,
    create_tesseract_lattice,
)

# Constants
TAU = 2 * math.pi


@dataclass
class MemoryConfig:
    """Configuration for the Memory Manager."""
    # Embedding
    embedding_dim: int = 64               # Embedding vector dimension
    use_external_embedder: bool = False   # Use external embedding function

    # Emotion mapping
    default_valence: float = 0.0          # Default emotional valence
    default_arousal: float = 0.0          # Default emotional arousal
    default_abstraction: float = 0.5      # Default abstraction level

    # Temporal
    temporal_decay: float = 0.001         # How fast time dimension advances

    # Consolidation
    auto_consolidate: bool = True         # Consolidate after retrieval
    consolidation_interval: int = 100     # Consolidate every N operations

    # Retrieval
    default_top_k: int = 5               # Default number of results
    emotional_radius: float = 1.5         # Radius for emotional context


@dataclass
class MemoryEvent:
    """
    A memory event to be stored.

    Represents a single memory with text content, emotional context,
    and optional embedding.
    """
    text: str                              # Raw text content
    valence: float = 0.0                   # Emotional valence [-1, 1]
    arousal: float = 0.0                   # Emotional arousal [-1, 1]
    abstraction: float = 0.5              # Abstraction level [0, 1]
    embedding: Optional[List[float]] = None # Pre-computed embedding
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: Optional[float] = None      # Event timestamp

    def __post_init__(self):
        # Clamp values to valid ranges
        self.valence = max(-1.0, min(1.0, self.valence))
        self.arousal = max(-1.0, min(1.0, self.arousal))
        self.abstraction = max(0.0, min(1.0, self.abstraction))

        if self.timestamp is None:
            self.timestamp = time.time()


@dataclass
class QueryResult:
    """Result from a memory query."""
    text: str                              # Retrieved text
    score: float                           # Combined relevance score
    resonance_score: float                 # Phase alignment score
    content_similarity: float              # Embedding similarity
    emotional_position: Tuple[float, float, float, float]  # (v, a, t, w)
    plate_id: str                          # Source plate ID
    access_count: int                      # How many times accessed
    rank: int = 0                          # Result rank

    @classmethod
    def from_retrieval_result(cls, result: RetrievalResult) -> 'QueryResult':
        """Convert from RetrievalResult."""
        return cls(
            text=result.plate.raw_text or "",
            score=result.combined_score,
            resonance_score=result.resonance_score,
            content_similarity=result.content_similarity,
            emotional_position=result.plate.position,
            plate_id=result.plate.plate_id,
            access_count=result.plate.access_count,
            rank=result.rank,
        )


class MemoryManager:
    """
    High-level Memory Manager for the Tesseract Lattice system.

    Provides a simple API for storing and retrieving memories
    using resonance-based retrieval.

    Example:
        manager = MemoryManager()

        # Store memories
        manager.store_event("Had a great meeting today", valence=0.7, arousal=0.3)
        manager.store_event("Feeling anxious about deadline", valence=-0.5, arousal=0.8)

        # Query memories
        results = manager.query("work stress")
        for r in results:
            print(f"[{r.score:.2f}] {r.text}")

        # Consolidate learning
        manager.consolidate()
    """

    def __init__(
        self,
        config: Optional[MemoryConfig] = None,
        embedding_fn: Optional[Callable[[str], List[float]]] = None,
        lattice_config: Optional[LatticeConfig] = None
    ):
        self.config = config or MemoryConfig()
        self.embedding_fn = embedding_fn

        # Initialize lattice engine
        self.engine = TesseractLatticeEngine(config=lattice_config)

        # Pre-populate with tesseract structure for semantic anchors
        self._initialize_tesseract_anchors()

        # Operation counter for auto-consolidation
        self._op_count = 0

        # Temporal reference (for relative positioning)
        self._time_origin = time.time()

    def _initialize_tesseract_anchors(self) -> None:
        """Initialize tesseract vertex anchors for semantic organization."""
        from lattice_core.plate import create_tesseract_vertices

        vertices = create_tesseract_vertices()
        for vertex in vertices:
            self.engine.add_plate(vertex)

        # Brief consolidation to establish connections
        self.engine.update(steps=20)

    def _compute_embedding(self, text: str) -> List[float]:
        """Compute embedding for text."""
        if self.embedding_fn is not None:
            return self.embedding_fn(text)

        # Fallback: simple hash-based pseudo-embedding
        dim = self.config.embedding_dim
        embedding = [0.0] * dim

        for i, char in enumerate(text.lower()):
            idx = (hash(char) + i * 7) % dim
            embedding[idx] += ord(char) / 256.0

        # Normalize
        norm = math.sqrt(sum(x ** 2 for x in embedding)) or 1.0
        return [x / norm for x in embedding]

    def _compute_temporal_position(self, timestamp: float) -> float:
        """Convert timestamp to temporal position z."""
        elapsed = timestamp - self._time_origin
        # Use log scale for time, normalized to [0, 1]
        return math.tanh(elapsed * self.config.temporal_decay)

    def store_event(
        self,
        text: str,
        valence: Optional[float] = None,
        arousal: Optional[float] = None,
        abstraction: Optional[float] = None,
        embedding: Optional[List[float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        timestamp: Optional[float] = None
    ) -> str:
        """
        Store a memory event.

        Args:
            text: The text content of the memory
            valence: Emotional valence (-1 to 1, negative to positive)
            arousal: Emotional arousal (-1 to 1, calm to excited)
            abstraction: Abstraction level (0 to 1, concrete to abstract)
            embedding: Pre-computed embedding vector (optional)
            metadata: Additional metadata dict
            timestamp: Event timestamp (defaults to now)

        Returns:
            The plate_id of the stored memory
        """
        # Create event
        event = MemoryEvent(
            text=text,
            valence=valence if valence is not None else self.config.default_valence,
            arousal=arousal if arousal is not None else self.config.default_arousal,
            abstraction=abstraction if abstraction is not None else self.config.default_abstraction,
            embedding=embedding,
            metadata=metadata or {},
            timestamp=timestamp,
        )

        # Compute embedding if not provided
        if event.embedding is None:
            event.embedding = self._compute_embedding(event.text)

        # Compute 4D position
        temporal = self._compute_temporal_position(event.timestamp)
        position = (event.valence, event.arousal, temporal, event.abstraction)

        # Create plate
        plate = MemoryPlate(
            position=position,
            content=event.embedding,
            raw_text=event.text,
            metadata=event.metadata,
            timestamp=event.timestamp,
        )

        # Modulate frequency based on emotion
        plate.modulate_frequency()

        # Add to engine
        self.engine.add_plate(plate)

        # Auto-consolidation
        self._op_count += 1
        if self.config.auto_consolidate and self._op_count % self.config.consolidation_interval == 0:
            self.consolidate()

        return plate.plate_id

    def query(
        self,
        text: Optional[str] = None,
        valence: Optional[float] = None,
        arousal: Optional[float] = None,
        embedding: Optional[List[float]] = None,
        top_k: Optional[int] = None
    ) -> List[QueryResult]:
        """
        Query memories using resonance retrieval.

        Args:
            text: Query text (will be embedded)
            valence: Emotional context valence
            arousal: Emotional context arousal
            embedding: Pre-computed query embedding
            top_k: Number of results to return

        Returns:
            List of QueryResult sorted by relevance
        """
        top_k = top_k or self.config.default_top_k

        # Compute query embedding
        if embedding is None and text is not None:
            embedding = self._compute_embedding(text)

        # Compute emotional position if provided
        emotional_position = None
        if valence is not None or arousal is not None:
            emotional_position = (
                valence if valence is not None else self.config.default_valence,
                arousal if arousal is not None else self.config.default_arousal,
                0.0,  # Current time
                self.config.default_abstraction,
            )

        # Perform retrieval
        results = self.engine.resonance_retrieval(
            content=embedding,
            emotional_position=emotional_position,
            top_k=top_k
        )

        # Convert to QueryResult
        query_results = []
        for result in results:
            if result.plate.raw_text:  # Only return plates with text
                query_results.append(QueryResult.from_retrieval_result(result))

        return query_results

    def consolidate(self, steps: int = 10) -> None:
        """
        Consolidate memory connections via Hebbian learning.

        Call this periodically to strengthen associations
        between frequently co-activated memories.
        """
        self.engine.consolidate(steps=steps)
        self.engine.update(steps=20)

    def get_stats(self) -> Dict[str, Any]:
        """Get memory system statistics."""
        snapshot = self.engine.snapshot()

        # Count text-containing plates (actual memories)
        n_memories = sum(
            1 for p in self.engine._plate_list if p.raw_text
        )

        return {
            "total_plates": snapshot["n_plates"],
            "n_memories": n_memories,
            "n_anchors": snapshot["n_plates"] - n_memories,
            "order_parameter": snapshot["order_parameter"],
            "energy": snapshot["energy"],
            "is_synchronized": snapshot["is_synchronized"],
            "operations": self._op_count,
        }

    def get_memory(self, plate_id: str) -> Optional[QueryResult]:
        """Get a specific memory by plate ID."""
        plate = self.engine.get_plate(plate_id)
        if plate is None or plate.raw_text is None:
            return None

        return QueryResult(
            text=plate.raw_text,
            score=1.0,
            resonance_score=1.0,
            content_similarity=1.0,
            emotional_position=plate.position,
            plate_id=plate.plate_id,
            access_count=plate.access_count,
        )

    def delete_memory(self, plate_id: str) -> bool:
        """Delete a memory by plate ID."""
        removed = self.engine.remove_plate(plate_id)
        return removed is not None

    def save(self, filepath: str) -> None:
        """Save memory state to file."""
        json_str = self.engine.to_json()
        with open(filepath, 'w') as f:
            f.write(json_str)

    def load(self, filepath: str) -> None:
        """Load memory state from file."""
        with open(filepath, 'r') as f:
            json_str = f.read()
        self.engine = TesseractLatticeEngine.from_json(json_str)


# ═════════════════════════════════════════════════════════════════════════════
# CONVENIENCE FUNCTIONS
# ═════════════════════════════════════════════════════════════════════════════

def create_memory_manager(
    embedding_fn: Optional[Callable[[str], List[float]]] = None,
    use_quartet_coupling: bool = True
) -> MemoryManager:
    """
    Create a memory manager with recommended settings.

    Args:
        embedding_fn: Optional external embedding function
        use_quartet_coupling: Enable P ~ N³ capacity scaling

    Returns:
        Configured MemoryManager instance
    """
    config = MemoryConfig(
        use_external_embedder=embedding_fn is not None,
    )

    lattice_config = LatticeConfig(
        enable_quartet=use_quartet_coupling,
        K=2.5,  # Slightly above critical
    )

    return MemoryManager(
        config=config,
        embedding_fn=embedding_fn,
        lattice_config=lattice_config,
    )
