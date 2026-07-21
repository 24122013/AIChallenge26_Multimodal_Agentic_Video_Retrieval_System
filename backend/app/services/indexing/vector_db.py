"""vector_db — abstraction cho vector index (Team P2).

Cung cấp interface chung để build / save / load / search vector, với 2 backend:
- FaissVectorDB: dùng FAISS (lazy import) — production, khớp artifact hiện có.
- NumpyVectorDB: brute-force cosine bằng NumPy — không cần faiss, tiện cho test và
  dataset nhỏ.

Vector giả định đã L2-normalize; metric mặc định inner product = cosine.
"""
from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np


class VectorDB(Protocol):
    metric: str

    @property
    def ntotal(self) -> int: ...

    def add(self, vectors: np.ndarray) -> None: ...

    def search(self, query: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]: ...

    def save(self, path: str | Path) -> None: ...


def _as_2d_f32(vectors: np.ndarray) -> np.ndarray:
    arr = np.asarray(vectors, dtype="float32")
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr


# ---------------------------------------------------------------------------
# NumPy backend (không cần faiss)
# ---------------------------------------------------------------------------

class NumpyVectorDB:
    """Brute-force vector search bằng NumPy. Dùng khi thiếu faiss / dataset nhỏ."""

    def __init__(self, metric: str = "ip") -> None:
        if metric not in ("ip", "l2"):
            raise ValueError(f"metric không hỗ trợ: {metric}")
        self.metric = metric
        self._vectors: np.ndarray | None = None

    @property
    def ntotal(self) -> int:
        return 0 if self._vectors is None else int(self._vectors.shape[0])

    def add(self, vectors: np.ndarray) -> None:
        arr = _as_2d_f32(vectors)
        if self._vectors is None:
            self._vectors = arr
        else:
            self._vectors = np.concatenate([self._vectors, arr], axis=0)

    def search(self, query: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
        if self._vectors is None or self.ntotal == 0:
            raise ValueError("VectorDB rỗng — hãy add vectors trước khi search.")
        q = _as_2d_f32(query)
        k = max(1, min(int(top_k), self.ntotal))
        if self.metric == "ip":
            sims = q @ self._vectors.T  # (nq, N)
            idx = np.argsort(-sims, axis=1)[:, :k]
            scores = np.take_along_axis(sims, idx, axis=1)
        else:  # l2
            dists = (
                np.sum(q**2, axis=1, keepdims=True)
                - 2 * (q @ self._vectors.T)
                + np.sum(self._vectors**2, axis=1)[None, :]
            )
            idx = np.argsort(dists, axis=1)[:, :k]
            scores = np.take_along_axis(dists, idx, axis=1)
        return scores.astype("float32"), idx.astype("int64")

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        np.save(p.with_suffix(".npy"), self._vectors if self._vectors is not None else np.empty((0, 0)))

    @classmethod
    def load(cls, path: str | Path, metric: str = "ip") -> "NumpyVectorDB":
        p = Path(path)
        arr = np.load(p.with_suffix(".npy"))
        db = cls(metric=metric)
        if arr.size:
            db.add(arr)
        return db


# ---------------------------------------------------------------------------
# FAISS backend
# ---------------------------------------------------------------------------

class FaissVectorDB:
    """Wrapper FAISS IndexFlatIP / IndexFlatL2 (lazy import faiss)."""

    def __init__(self, dim: int | None = None, metric: str = "ip") -> None:
        if metric not in ("ip", "l2"):
            raise ValueError(f"metric không hỗ trợ: {metric}")
        self.metric = metric
        self.dim = dim
        self._index = None

    def _faiss(self):
        try:
            import faiss  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "faiss chưa được cài. pip install faiss-cpu hoặc dùng NumpyVectorDB."
            ) from exc
        return faiss

    def _ensure_index(self, dim: int) -> None:
        if self._index is not None:
            return
        faiss = self._faiss()
        self.dim = dim
        self._index = faiss.IndexFlatIP(dim) if self.metric == "ip" else faiss.IndexFlatL2(dim)

    @property
    def ntotal(self) -> int:
        return 0 if self._index is None else int(self._index.ntotal)

    def add(self, vectors: np.ndarray) -> None:
        arr = _as_2d_f32(vectors)
        self._ensure_index(int(arr.shape[1]))
        self._index.add(arr)

    def search(self, query: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
        if self._index is None:
            raise ValueError("Index chưa build.")
        q = _as_2d_f32(query)
        return self._index.search(q, int(top_k))

    def save(self, path: str | Path) -> None:
        faiss = self._faiss()
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, p.as_posix())

    @classmethod
    def load(cls, path: str | Path, metric: str = "ip") -> "FaissVectorDB":
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"FAISS index not found: {p}")
        db = cls(metric=metric)
        faiss = db._faiss()
        db._index = faiss.read_index(p.as_posix())
        db.dim = db._index.d
        return db


def create_vector_db(backend: str = "faiss", metric: str = "ip", dim: int | None = None) -> VectorDB:
    """Factory: 'faiss' | 'numpy'."""
    if backend in ("faiss", "flat"):
        return FaissVectorDB(dim=dim, metric=metric)
    if backend in ("numpy", "np", "bruteforce"):
        return NumpyVectorDB(metric=metric)
    raise ValueError(f"backend vector_db không hỗ trợ: {backend!r}")


__all__ = [
    "VectorDB",
    "NumpyVectorDB",
    "FaissVectorDB",
    "create_vector_db",
]
