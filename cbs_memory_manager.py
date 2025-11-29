"""
Working/long-term memory management for the CBS runtime.

The MemoryManager collaborates with CognitionBootstrap and GHMP utilities to
track recent conversation turns, consolidate them into plates, and surface
relevant context during reasoning.
"""

from __future__ import annotations

import json
import uuid
from collections import deque
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional

from ghmp import Emotion, MemoryNode, encode_plate, save_plate


class MemoryManager:
    """Track working memory and GHMP-backed long-term memory."""

    def __init__(self, bootstrap, max_working_memory: Optional[int] = None):
        self.bootstrap = bootstrap
        config = bootstrap.config.get("memory", {})
        self.max_working_memory = max_working_memory or config.get("max_working_memory", 40)
        self.consolidation_threshold = config.get("consolidation_threshold", 0.65)
        self.working_memory: Deque[MemoryNode] = deque(maxlen=self.max_working_memory)
        self.long_term_cache: List[MemoryNode] = list(bootstrap.memory_index)

    # ----------------------------------------------------------------- helpers
    def _create_node(
        self,
        text: str,
        importance: float,
        tags: Optional[List[str]] = None,
    ) -> MemoryNode:
        node_id = f"MEM-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:6]}"
        emotion = Emotion(valence=max(min(importance, 1.0), -1.0), arousal=importance, label="context")
        return MemoryNode(
            node_id=node_id,
            deck_id=self.bootstrap.config.get("deck_id", "CBS_DEMO"),
            title=f"Memory | {node_id}",
            payload_text=text,
            tags=tags or ["conversation"],
            emotion=emotion,
            links=[],
            metadata={"importance": importance, "created_at": datetime.utcnow().isoformat()},
        )

    def _persist_node(self, node: MemoryNode):
        plate = encode_plate(node, self.bootstrap.encryption_key)
        filename = f"{node.node_id}.png"
        save_plate(plate, self.bootstrap.memory_dir / filename)
        self.long_term_cache.append(node)

    # ----------------------------------------------------------------- exposed
    def add_to_working_memory(
        self,
        text: str,
        importance: float = 0.5,
        tags: Optional[List[str]] = None,
    ) -> MemoryNode:
        node = self._create_node(text, importance, tags)
        self.working_memory.append(node)
        if importance >= self.consolidation_threshold:
            self._persist_node(node)
        return node

    def consolidate_session(self, summary_text: str) -> Path:
        """Persist all working memories into a single plate and clear buffer."""
        if not self.working_memory:
            return self.bootstrap.memory_dir

        combined_payload = {
            "summary": summary_text,
            "nodes": [asdict(node) for node in list(self.working_memory)],
        }
        node = MemoryNode(
            node_id=f"SESSION-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
            deck_id=self.bootstrap.config.get("deck_id", "CBS_DEMO"),
            title=f"Session Summary | {summary_text[:48]}",
            payload_text=json.dumps(combined_payload, ensure_ascii=False),
            tags=["session", "summary"],
            emotion=Emotion(valence=0.3, arousal=0.6, label="reflective"),
            links=[],
            metadata={"node_count": len(self.working_memory)},
        )
        plate = encode_plate(node, self.bootstrap.encryption_key)
        destination = self.bootstrap.memory_dir / f"{node.node_id}.png"
        save_plate(plate, destination)
        self.long_term_cache.append(node)
        self.working_memory.clear()
        return destination

    def retrieve_context(
        self,
        query: str,
        max_items: int = 5,
        include_working: bool = True,
        include_longterm: bool = True,
    ) -> List[MemoryNode]:
        """Simple keyword search across memories."""
        query_lower = query.lower()
        results: List[MemoryNode] = []

        def _search(collection: List[MemoryNode]):
            for node in reversed(collection):
                if len(results) >= max_items:
                    break
                if query_lower in node.payload_text.lower():
                    results.append(node)

        if include_working:
            _search(list(self.working_memory))
        if include_longterm:
            _search(self.long_term_cache)
        return results[:max_items]

    def get_recent_context(self, count: int = 5) -> List[MemoryNode]:
        return list(self.working_memory)[-count:]

    def get_statistics(self) -> Dict[str, Any]:
        return {
            "working_memory": len(self.working_memory),
            "long_term_memory": len(self.long_term_cache),
            "consolidation_threshold": self.consolidation_threshold,
            "max_working_memory": self.max_working_memory,
        }


__all__ = ["MemoryManager"]
