"""
Self-update, backup, and rollback helper for the CBS runtime.
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional


class UpdateManager:
    """Handle backups, skill installation, and rollback operations."""

    def __init__(self, bootstrap):
        self.bootstrap = bootstrap
        self.history_path = bootstrap.updates_dir / "update_history.json"
        if not self.history_path.exists():
            self.history_path.write_text("[]", encoding="utf-8")

    # ---------------------------------------------------------------- backups
    def backup_system(self, label: Optional[str] = None) -> Path:
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        label_part = f"-{label}" if label else ""
        destination = self.bootstrap.backups_dir / f"backup-{timestamp}{label_part}"
        shutil.copytree(
            self.bootstrap.base_path,
            destination,
            ignore=shutil.ignore_patterns("backups", "__pycache__"),
            dirs_exist_ok=True,
        )
        self._record_history({"event": "backup", "path": str(destination)})
        return destination

    # --------------------------------------------------------------- rollback
    def rollback(self, backup_name: str) -> bool:
        backup_path = self.bootstrap.backups_dir / backup_name
        if not backup_path.exists():
            return False

        for item in backup_path.iterdir():
            target = self.bootstrap.base_path / item.name
            if item.is_dir():
                shutil.copytree(item, target, dirs_exist_ok=True)
            else:
                shutil.copy2(item, target)
        self._record_history({"event": "rollback", "source": str(backup_path)})
        return True

    # ---------------------------------------------------------- skill install
    def install_skill(self, skill_source: Path | str) -> Path:
        skill_source = Path(skill_source)
        destination = self.bootstrap.skills_dir / skill_source.name
        shutil.copy2(skill_source, destination)
        self._record_history({"event": "install_skill", "path": str(destination)})
        return destination

    # ------------------------------------------------------------- update log
    def _record_history(self, event: Dict):
        history = json.loads(self.history_path.read_text(encoding="utf-8"))
        event["timestamp"] = datetime.utcnow().isoformat()
        history.append(event)
        self.history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")

    def get_history(self):
        return json.loads(self.history_path.read_text(encoding="utf-8"))


__all__ = ["UpdateManager"]
