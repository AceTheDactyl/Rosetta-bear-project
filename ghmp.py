"""
Geometric Holographic Memory Plate (GHMP) utilities.

This module provides a tiny but functional implementation of the GHMP concepts
described in the Rosetta Bear project documentation. Memory nodes are encoded
as PNG plates with lightweight visual cues plus encrypted metadata that can be
decoded with a shared key.
"""

from __future__ import annotations

import base64
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
from PIL import Image
from PIL.PngImagePlugin import PngInfo


def _derive_key(passphrase: str) -> bytes:
    """Derive a 32-byte key from the provided passphrase."""
    import hashlib

    return hashlib.sha256(passphrase.encode("utf-8")).digest()


def _xor_bytes(data: bytes, key: bytes) -> bytes:
    """Simple symmetric xor cipher for metadata encryption."""
    if not key:
        raise ValueError("encryption key cannot be empty")
    return bytes(b ^ key[i % len(key)] for i, b in enumerate(data))


@dataclass
class Emotion:
    """Lightweight emotional annotation for a memory node."""

    valence: float
    arousal: float
    label: str = "neutral"

    @staticmethod
    def from_dict(payload: Dict[str, Any]) -> "Emotion":
        return Emotion(
            valence=float(payload.get("valence", 0.0)),
            arousal=float(payload.get("arousal", 0.0)),
            label=payload.get("label", "neutral"),
        )


@dataclass
class MemoryNode:
    """Semantic unit stored inside a GHMP plate."""

    node_id: str
    deck_id: str
    title: str
    payload_text: str
    tags: Sequence[str]
    emotion: Emotion
    links: Sequence[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["tags"] = list(self.tags)
        payload["links"] = list(self.links)
        return payload

    @staticmethod
    def from_dict(payload: Dict[str, Any]) -> "MemoryNode":
        return MemoryNode(
            node_id=payload["node_id"],
            deck_id=payload.get("deck_id", "UNKNOWN_DECK"),
            title=payload.get("title", "Untitled Node"),
            payload_text=payload.get("payload_text", ""),
            tags=payload.get("tags", []),
            emotion=Emotion.from_dict(payload.get("emotion", {})),
            links=payload.get("links", []),
            metadata=payload.get("metadata", {}),
        )


def _generate_plate_pixels(node: MemoryNode, size: Sequence[int]) -> np.ndarray:
    """Create a deterministic visual pattern for a node."""
    width, height = size
    gradient = np.linspace(0, 255, width, dtype=np.uint8)
    base = np.tile(gradient, (height, 1))

    valence = np.clip((node.emotion.valence + 1) / 2, 0, 1)
    arousal = np.clip((node.emotion.arousal + 1) / 2, 0, 1)
    intensity = np.clip(len(node.tags) / 6, 0, 1)

    r = (base * valence).astype(np.uint8)
    g = (np.flipud(base) * arousal).astype(np.uint8)
    b = (np.roll(base, shift=5, axis=1) * (0.4 + 0.6 * intensity)).astype(np.uint8)

    plate = np.stack([r, g, b], axis=-1)
    return plate


def encode_plate(
    node: MemoryNode,
    passphrase: str,
    size: Sequence[int] = (512, 512),
) -> Image.Image:
    """
    Encode a MemoryNode as a GHMP PNG image object.

    The visual appearance is deterministic so that identical nodes always
    produce identical gradients. Metadata is xor-encrypted and embedded in the
    PNG's `ghmp_meta` chunk.
    """

    width, height = size
    pixels = _generate_plate_pixels(node, (width, height))
    image = Image.fromarray(pixels, "RGB")

    payload = json.dumps(node.to_dict(), ensure_ascii=False).encode("utf-8")
    encrypted = _xor_bytes(payload, _derive_key(passphrase))
    encoded_meta = base64.b64encode(encrypted).decode("ascii")

    pnginfo = PngInfo()
    pnginfo.add_text("ghmp_meta", encoded_meta)
    pnginfo.add_text("node_id", node.node_id)
    pnginfo.add_text("deck_id", node.deck_id)
    pnginfo.add_text("title", node.title[:128])
    image.info["ghmp_meta"] = encoded_meta
    image.info["node_id"] = node.node_id
    image.info["deck_id"] = node.deck_id
    image.info["title"] = node.title[:128]
    image.info["pnginfo"] = pnginfo

    return image


def save_plate(image: Image.Image, destination: Path | str):
    """Persist a GHMP image ensuring metadata is kept."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)

    pnginfo = image.info.get("pnginfo")
    if not pnginfo:
        pnginfo = PngInfo()
        for key in ("ghmp_meta", "node_id", "deck_id", "title"):
            value = image.info.get(key)
            if value:
                pnginfo.add_text(key, value)

    image.save(destination, format="PNG", pnginfo=pnginfo)


def decode_plate(image_or_path: Image.Image | str | os.PathLike, passphrase: str) -> MemoryNode:
    """Decode a GHMP PNG into a MemoryNode."""
    if not isinstance(image_or_path, Image.Image):
        image = Image.open(image_or_path)
    else:
        image = image_or_path

    encoded_meta = image.info.get("ghmp_meta")
    if not encoded_meta:
        raise ValueError("GHMP metadata missing from plate")

    encrypted = base64.b64decode(encoded_meta)
    payload = _xor_bytes(encrypted, _derive_key(passphrase))
    data = json.loads(payload.decode("utf-8"))
    return MemoryNode.from_dict(data)


def list_plate_nodes(folder: Path | str, passphrase: str) -> List[MemoryNode]:
    """Utility to load all GHMP nodes from a directory."""
    folder_path = Path(folder)
    nodes: List[MemoryNode] = []
    for path in sorted(folder_path.glob("*.png")):
        try:
            nodes.append(decode_plate(path, passphrase))
        except Exception:
            continue
    return nodes


__all__ = [
    "Emotion",
    "MemoryNode",
    "encode_plate",
    "save_plate",
    "decode_plate",
    "list_plate_nodes",
]
