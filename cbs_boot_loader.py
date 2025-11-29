"""
Cognition Bootstrap System (CBS) boot loader.

Responsible for preparing the working directory, loading identity plates, and
maintaining configuration/diagnostic files used by other CBS components.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from ghmp import Emotion, MemoryNode, decode_plate, encode_plate, save_plate


@dataclass
class IdentityProfile:
    """In-memory representation of the system identity."""

    node: MemoryNode
    plate_path: Path

    @property
    def name(self) -> str:
        return self.node.title


class CognitionBootstrap:
    """Prepare filesystem layout and load GHMP assets."""

    REQUIRED_DIRS = ("memory", "skills", "backups", "updates", "logs")

    def __init__(
        self,
        base_path: str | Path,
        encryption_key: str,
        config_filename: str = "config.json",
    ):
        self.base_path = Path(base_path)
        self.encryption_key = encryption_key
        self.config_path = self.base_path / config_filename
        self.log_path = self.base_path / "boot_log.txt"
        self.identity_path = self.base_path / "identity.png"
        self.identity: IdentityProfile | None = None
        self.config: Dict[str, Any] = {}
        self.memory_index: List[MemoryNode] = []

    # ------------------------------------------------------------------ booting
    def boot(self) -> "CognitionBootstrap":
        """Execute the full bootstrap routine."""
        self.base_path.mkdir(parents=True, exist_ok=True)
        for dirname in self.REQUIRED_DIRS:
            (self.base_path / dirname).mkdir(exist_ok=True)

        self._log("=== CBS boot sequence started ===")
        self._log(f"Base path: {self.base_path}")

        self.config = self._load_or_initialize_config()
        self.identity = self._load_or_create_identity()
        self.memory_index = self._scan_long_term_memory()

        self._log(f"Loaded {len(self.memory_index)} long-term memory plates")
        self._log("=== CBS boot sequence complete ===")
        return self

    # ----------------------------------------------------------------- helpers
    def _load_or_initialize_config(self) -> Dict[str, Any]:
        if self.config_path.exists():
            with self.config_path.open("r", encoding="utf-8") as fh:
                config = json.load(fh)
            self._log("Configuration loaded")
            return config

        config = {
            "deck_id": "CBS_DEMO",
            "version": "1.0.0",
            "reasoning": {
                "backend": "offline",
                "model": "demo-local",
                "max_context_messages": 12,
                "default_temperature": 0.4,
            },
            "memory": {
                "max_working_memory": 40,
                "consolidation_threshold": 0.65,
                "auto_consolidate_interval": 1800,
            },
            "update": {
                "check_interval": 86400,
                "auto_backup": True,
                "server_url": "",
            },
        }

        with self.config_path.open("w", encoding="utf-8") as fh:
            json.dump(config, fh, indent=2)
        self._log("Configuration initialized with defaults")
        return config

    def _load_or_create_identity(self) -> IdentityProfile:
        if self.identity_path.exists():
            node = decode_plate(self.identity_path, self.encryption_key)
            self._log(f"Identity loaded: {node.title}")
            return IdentityProfile(node=node, plate_path=self.identity_path)

        default_node = MemoryNode(
            node_id="IDENTITY-DEMO-001",
            deck_id=self.config.get("deck_id", "CBS_DEMO"),
            title="DemoBot | Cognition Bootstrap System",
            payload_text="CBS identity profile for local experimentation.",
            tags=["identity", "demo", "cbs"],
            emotion=Emotion(valence=0.2, arousal=0.4, label="steadfast"),
            links=[],
            metadata={"created_at": datetime.utcnow().isoformat()},
        )
        plate = encode_plate(default_node, self.encryption_key)
        save_plate(plate, self.identity_path)
        self._log("Default identity generated")
        return IdentityProfile(node=default_node, plate_path=self.identity_path)

    def _scan_long_term_memory(self) -> List[MemoryNode]:
        memories: List[MemoryNode] = []
        memory_dir = self.base_path / "memory"
        for plate_path in sorted(memory_dir.glob("*.png")):
            try:
                memories.append(decode_plate(plate_path, self.encryption_key))
            except Exception as exc:  # pragma: no cover - diagnostics
                self._log(f"Warning: failed to load {plate_path.name}: {exc}")
        return memories

    # ------------------------------------------------------------------- utils
    def _log(self, message: str):
        timestamp = datetime.utcnow().isoformat()
        with self.log_path.open("a", encoding="utf-8") as fh:
            fh.write(f"{timestamp} {message}\n")

    # ----------------------------------------------------------------- exposed
    @property
    def memory_dir(self) -> Path:
        return self.base_path / "memory"

    @property
    def skills_dir(self) -> Path:
        return self.base_path / "skills"

    @property
    def backups_dir(self) -> Path:
        return self.base_path / "backups"

    @property
    def updates_dir(self) -> Path:
        return self.base_path / "updates"

    def summary(self) -> Dict[str, Any]:
        """Return a compact snapshot of the bootstrap state."""
        return {
            "base_path": str(self.base_path),
            "deck_id": self.config.get("deck_id"),
            "identity": self.identity.node.title if self.identity else None,
            "memory_plates": len(self.memory_index),
            "skills": len(list(self.skills_dir.glob("*.png"))),
        }


__all__ = ["CognitionBootstrap", "IdentityProfile"]
