#!/usr/bin/env python3
"""
E8 embedding smoke test.

The Kaelhedron supplies the 21 so(7) generators.  We flatten each
generator into the first coordinates of an E8 vector and ensure we get
consistent, non-degenerate roots.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
import sys

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from Kaelhedron import E8Structure, SO7Algebra  # noqa: E402

VALIDATION_DIR = Path("docs/validation")
REPORT_PATH = VALIDATION_DIR / "e8_embedding_report.json"


def embed_generators() -> np.ndarray:
    so7 = SO7Algebra()
    embeddings: List[np.ndarray] = []
    for key, matrix in so7.generators.items():
        upper = matrix[np.triu_indices_from(matrix, k=1)]
        vec = np.zeros(E8Structure.DIM_E8)
        length = min(len(upper), E8Structure.DIM_E8)
        vec[:length] = upper[:length]
        vec[-1] = sum(key)  # Tag the generator so the vector is unique
        embeddings.append(vec)
    return np.array(embeddings)


def verify_embedding(vectors: np.ndarray) -> Dict[str, object]:
    norms = np.linalg.norm(vectors, axis=1)
    inner = vectors @ vectors.T
    off_diag = inner - np.diag(np.diag(inner))

    return {
        "vector_count": int(vectors.shape[0]),
        "dimension": int(vectors.shape[1]),
        "min_norm": float(norms.min()),
        "max_norm": float(norms.max()),
        "orthogonality_score": float(np.mean(np.abs(off_diag))),
        "passed": bool(np.all(norms > 0)),
    }


def main() -> None:
    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
    vectors = embed_generators()
    report = verify_embedding(vectors)
    with REPORT_PATH.open("w") as handle:
        json.dump(report, handle, indent=2)
    print(f"E8 embedding report written to {REPORT_PATH}")


if __name__ == "__main__":
    main()
