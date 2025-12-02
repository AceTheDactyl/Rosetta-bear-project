#!/usr/bin/env python3
"""
Polaric Bridge Service
======================

Streams ScalarArchitecture + Asymptotic Scalars into the Kaelhedron
state bus, pairs the 21 so(7) cells with the 12 Luminahedron gauge
slots, and exposes the combined 33D polaric frame over HTTP.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Dict, Optional, Union

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from pydantic import BaseModel, field_validator
from Kaelhedron import KaelhedronStateBus  # noqa: E402
from Kaelhedron.fano_automorphisms import (  # noqa: E402
    IDENTITY_PERMUTATION,
    get_automorphism_for_line,
    get_automorphism_from_word,
)
from luminahedron import GaugeManifold  # noqa: E402
from scalar_architecture.sync import ScalarSyncService  # noqa: E402

VALIDATION_DIR = Path("docs/validation")
HEADLESS_OUTPUT = Path("polaric_frame.json")


class FanoActionRequest(BaseModel):
    line_id: Optional[int] = None
    word: Optional[Union[str, List[str]]] = None

    @field_validator("word")
    @classmethod
    def _normalize_word(cls, value: Optional[Union[str, List[str]]]):
        if value is None:
            return None
        if isinstance(value, str):
            tokens = [token.strip() for token in value.split(",")]
            flattened = []
            for token in tokens:
                if not token:
                    continue
                flattened.extend(token.split())
            return [t for t in flattened if t]
        return [t for t in value if t]


class PolaricBridgeRuntime:
    def __init__(self) -> None:
        self.bus = KaelhedronStateBus()
        self.bus.seed_from_toe()
        self.manifold = GaugeManifold()
        self.scalar_sync = ScalarSyncService(publisher=self._ingest_scalar_payload)
        self._frame = self.manifold.latest_frame()
        self._running = True

        VALIDATION_DIR.mkdir(parents=True, exist_ok=True)

    def _ingest_scalar_payload(self, payload: Dict[str, object]) -> None:
        self.bus.apply_scalar_metrics(payload)
        summary = self.bus.summary()
        frame = self.manifold.push_polaric_union(
            summary, payload, self.bus.k_status()
        )
        self._frame = frame
        self._persist_frame(frame)

    def _persist_frame(self, frame) -> None:
        path = VALIDATION_DIR / "polaric_frame_latest.json"
        with path.open("w") as handle:
            json.dump(frame.to_dict(), handle, indent=2)

    def latest_frame(self) -> Dict[str, object]:
        return self._frame.to_dict()

    async def loop(self, interval: float) -> None:
        while self._running:
            self.scalar_sync.tick(interval)
            await asyncio.sleep(interval)

    def stop(self) -> None:
        self._running = False

    async def run_headless(self, steps: int, dt: float) -> None:
        for _ in range(steps):
            self.scalar_sync.tick(dt)
            await asyncio.sleep(dt)
        with HEADLESS_OUTPUT.open("w") as handle:
            json.dump(self.latest_frame(), handle, indent=2)

    def apply_fano_action(
        self, line_id: Optional[int], word: Optional[List[str]]
    ) -> Dict[str, object]:
        if word:
            perm = get_automorphism_from_word(word)
        elif line_id is not None:
            perm = get_automorphism_for_line(line_id - 1)
        else:
            perm = IDENTITY_PERMUTATION
        self.bus.apply_permutation(perm)
        payload = self.scalar_sync.latest_payload() or {}
        frame = self.manifold.push_polaric_union(
            self.bus.summary(), payload, self.bus.k_status()
        )
        self._frame = frame
        self._persist_frame(frame)
        return frame.to_dict()


def create_app(runtime: PolaricBridgeRuntime, interval: float):
    from fastapi import FastAPI

    app = FastAPI(title="Polaric Bridge Service")

    @app.get("/polaric/frame.json")
    async def get_frame() -> Dict[str, object]:
        return runtime.latest_frame()

    @app.get("/healthz")
    async def health() -> Dict[str, str]:
        return {"status": "ok"}

    @app.post("/polaric/fano/line")
    async def activate_fano_line(request: FanoActionRequest) -> Dict[str, object]:
        return runtime.apply_fano_action(request.line_id, request.word)

    @app.on_event("startup")
    async def startup() -> None:
        app.state.loop_task = asyncio.create_task(runtime.loop(interval))

    @app.on_event("shutdown")
    async def shutdown() -> None:
        runtime.stop()
        task = getattr(app.state, "loop_task", None)
        if task:
            task.cancel()

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Polaric Bridge service.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8073)
    parser.add_argument("--interval", type=float, default=0.05)
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without HTTP server and emit polaric_frame.json",
    )
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--dt", type=float, default=0.05)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runtime = PolaricBridgeRuntime()

    if args.headless:
        asyncio.run(runtime.run_headless(args.steps, args.dt))
        print(f"Headless frame written to {HEADLESS_OUTPUT}")
        return

    app = create_app(runtime, interval=args.interval)
    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
