# memory/__init__.py
"""
Memory Package: High-Level Memory Management API
=================================================

Provides a user-friendly interface for storing and retrieving memories
using the Tesseract Lattice Engine.

Architecture:
    ┌─────────────────────────────────────────────────────────────────┐
    │ MEMORY MANAGER API                                               │
    │ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐              │
    │ │ store_event()│ │ query()      │ │ consolidate()│              │
    │ └──────────────┘ └──────────────┘ └──────────────┘              │
    │                                                                  │
    │ • Event → Plate conversion                                       │
    │ • 4D positioning (emotion, time, abstraction)                   │
    │ • Query pattern creation                                         │
    └─────────────────────────────────────────────────────────────────┘
"""

from .memory_manager import (
    MemoryManager,
    MemoryEvent,
    QueryResult,
    MemoryConfig,
)

__version__ = "1.0.0"

__all__ = [
    "MemoryManager",
    "MemoryEvent",
    "QueryResult",
    "MemoryConfig",
]
