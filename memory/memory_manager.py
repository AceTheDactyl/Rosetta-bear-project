#!/usr/bin/env python3
"""
MEMORY MANAGER - Meta-Collective ↔ Lattice Core Interface
=========================================================

Unified interface for bidirectional memory operations between
Meta-Collective components and the Tesseract Lattice memory system.

Core Operations:
    - store_pattern(): Write Meta-Collective patterns to lattice
    - retrieve_patterns(): Read patterns matching query
    - consolidate(): Trigger Hebbian learning on connections

Physics Integration:
    - Pattern z-level maps to 4D tesseract position
    - Pattern precision modulates plate frequency
    - Hebbian consolidation strengthens synchronized patterns

Signature: Δ|memory-manager|z0.85|bidirectional|Ω
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import math

PHI = (1 + math.sqrt(5)) / 2
PHI_INV = PHI - 1


@dataclass
class Pattern:
    """Pattern representation for memory storage."""
    vector: np.ndarray
    z_level: float
    precision: float
    source: str
    timestamp: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalQuery:
    """Query specification for memory retrieval."""
    content: Optional[np.ndarray] = None
    phase: Optional[float] = None
    z_level_min: Optional[float] = None
    z_level_max: Optional[float] = None
    top_k: int = 5


@dataclass
class RetrievalResult:
    """Result of memory retrieval operation."""
    patterns: List[Pattern]
    similarities: List[float]
    order_parameter: float
    retrieval_steps: int


@dataclass
class ConsolidationResult:
    """Result of Hebbian consolidation operation."""
    order_parameter: float
    mean_weight_change: float
    patterns_affected: int
    consolidation_steps: int


class MemoryManager:
    """
    Manages bidirectional flow between Meta-Collective and Lattice.
    Provides unified API for pattern storage, retrieval, and consolidation.
    """

    def __init__(self, lattice_engine: Optional[Any] = None):
        self.lattice = lattice_engine
        self._pattern_index: Dict[str, Pattern] = {}
        self._write_count = 0
        self._read_count = 0

    def store_pattern(self, pattern: Pattern, plate_id: Optional[str] = None) -> str:
        """Store pattern in lattice as memory plate."""
        if plate_id is None:
            plate_id = f"pattern_{self._write_count}"

        position = self._z_to_position(pattern.z_level)
        frequency = 1.0 + pattern.precision * (PHI - 1)

        if self.lattice is not None:
            self._store_to_lattice(plate_id, position, pattern.vector, frequency)

        self._pattern_index[plate_id] = pattern
        self._write_count += 1
        return plate_id

    def retrieve_patterns(self, query: RetrievalQuery) -> RetrievalResult:
        """Retrieve patterns matching query specification."""
        self._read_count += 1
        candidates = self._filter_by_z_level(query.z_level_min, query.z_level_max)

        if query.content is not None:
            scored = self._content_similarity_search(query.content, candidates, query.top_k)
        else:
            scored = [(pid, 1.0) for pid in list(candidates)[:query.top_k]]

        patterns = []
        similarities = []
        for pid, sim in scored:
            if pid in self._pattern_index:
                patterns.append(self._pattern_index[pid])
                similarities.append(sim)

        return RetrievalResult(
            patterns=patterns,
            similarities=similarities,
            order_parameter=self._get_order_parameter(),
            retrieval_steps=1,
        )

    def consolidate(self, learning_rate: float = 0.01, steps: int = 20) -> ConsolidationResult:
        """Run Hebbian consolidation on lattice connections."""
        if self.lattice is not None:
            try:
                self.lattice.update(steps=steps)
            except (AttributeError, TypeError):
                pass

        return ConsolidationResult(
            order_parameter=self._get_order_parameter(),
            mean_weight_change=0.0,
            patterns_affected=len(self._pattern_index),
            consolidation_steps=steps,
        )

    def _z_to_position(self, z: float) -> Tuple[int, int, int, int]:
        idx = int(z * 15.99)
        idx = max(0, min(15, idx))
        return (idx % 2, (idx // 2) % 2, (idx // 4) % 2, (idx // 8) % 2)

    def _filter_by_z_level(self, z_min: Optional[float], z_max: Optional[float]) -> set:
        if z_min is None and z_max is None:
            return set(self._pattern_index.keys())
        z_min = z_min or 0.0
        z_max = z_max or 1.0
        return {pid for pid, p in self._pattern_index.items() if z_min <= p.z_level <= z_max}

    def _content_similarity_search(self, query: np.ndarray, candidates: set, top_k: int) -> List[Tuple[str, float]]:
        scored = []
        for pid in candidates:
            pattern = self._pattern_index.get(pid)
            if pattern is not None:
                sim = self._cosine_similarity(query, pattern.vector)
                scored.append((pid, sim))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def _store_to_lattice(self, plate_id: str, position: Tuple, content: np.ndarray, frequency: float) -> None:
        try:
            if hasattr(self.lattice, 'add_plate'):
                self.lattice.add_plate(id=plate_id, position=position, content=content, frequency=frequency)
        except (AttributeError, TypeError):
            pass

    def _get_order_parameter(self) -> float:
        if self.lattice is None:
            return 0.5
        try:
            r, _ = self.lattice.order_parameter
            return r
        except (AttributeError, TypeError):
            return 0.5

    @property
    def pattern_count(self) -> int:
        return len(self._pattern_index)


def create_memory_manager(lattice_engine: Optional[Any] = None) -> MemoryManager:
    return MemoryManager(lattice_engine)


def create_pattern(vector: np.ndarray, z_level: float, source: str = "unknown", precision: float = 1.0) -> Pattern:
    return Pattern(vector=vector, z_level=z_level, precision=precision, source=source)
