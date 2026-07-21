"""index_manager — điều phối build & load index (Team P2).

Gộp các bước: embeddings (.npy) + embedding metadata (.jsonl) -> vector index
(qua vector_db) + frame_map.json (+ neighbor index tuỳ chọn). Cũng cung cấp
`IndexBundle` để retrieval load 1 phát: index + frame_map + neighbor.

Module này reuse vector_db & neighbor_index; không tự viết search logic.
"""
from __future__ import annotations

import glob
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from backend.app.services.indexing.neighbor_index import build_neighbor_index
from backend.app.services.indexing.vector_db import create_vector_db

logger = logging.getLogger(__name__)


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _l2_normalize(v: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return v / norms


def _frame_map_entry(record: dict, faiss_index: int) -> dict:
    return {
        "frame_id": record.get("frame_id"),
        "video_id": record.get("video_id"),
        "shot_id": record.get("shot_id", ""),
        "segment_id": record.get("segment_id", ""),
        "shot_index": record.get("shot_index"),
        "shot_start": record.get("shot_start"),
        "shot_end": record.get("shot_end"),
        "timestamp": record.get("timestamp"),
        "timestamp_source": record.get("timestamp_source"),
        "timestamp_confidence": record.get("timestamp_confidence"),
        "frame_index": record.get("frame_index"),
        "keyframe_path": record.get("keyframe_path"),
        "thumbnail_path": record.get("thumbnail_path", record.get("keyframe_path")),
        "embedding_id": record.get("embedding_id"),
        "embedding_index": record.get("embedding_index"),
        "faiss_index": faiss_index,
    }


def build_index(
    *,
    embeddings_glob: str,
    embedding_metadata_template: str,
    index_path: str | Path,
    frame_map_path: str | Path,
    embeddings_prefix: str = "openclip_vit_b16_",
    embeddings_suffix: str = "",
    metric: str = "ip",
    backend: str = "faiss",
    normalize: bool = True,
    neighbor_index_path: str | Path | None = None,
    max_neighbors: int = 4,
) -> dict:
    """Build vector index + frame_map (+ neighbor index) từ embedding artifacts."""
    matched = sorted(Path(p) for p in glob.glob(embeddings_glob))
    if not matched:
        raise FileNotFoundError(f"Không tìm thấy embeddings cho glob: {embeddings_glob}")

    vector_batches: list[np.ndarray] = []
    all_records: list[dict] = []
    dim: int | None = None

    for emb_path in matched:
        stem = emb_path.stem
        if not stem.startswith(embeddings_prefix):
            raise ValueError(f"{emb_path} không bắt đầu bằng prefix {embeddings_prefix!r}")
        start = len(embeddings_prefix)
        end = len(stem) - len(embeddings_suffix) if embeddings_suffix else len(stem)
        video_id = stem[start:end]
        meta_path = Path(embedding_metadata_template.format(video_id=video_id))
        if not meta_path.exists():
            raise FileNotFoundError(f"Thiếu metadata cho {emb_path}: {meta_path}")

        vectors = np.load(emb_path).astype("float32", copy=False)
        if vectors.ndim != 2:
            raise ValueError(f"Embeddings phải 2D: {emb_path} shape={vectors.shape}")
        if dim is None:
            dim = int(vectors.shape[1])
        elif vectors.shape[1] != dim:
            raise ValueError(f"Lệch chiều vector ở {emb_path}: {vectors.shape[1]} != {dim}")

        records = _load_jsonl(meta_path)
        if len(records) != vectors.shape[0]:
            raise ValueError(
                f"Lệch số lượng {video_id}: {vectors.shape[0]} vectors vs {len(records)} records"
            )

        base = len(all_records)
        for offset, rec in enumerate(records):
            rec = dict(rec)
            rec["faiss_index"] = base + offset
            all_records.append(rec)
        vector_batches.append(vectors)
        logger.info("Loaded %s: %d vectors", video_id, vectors.shape[0])

    all_vectors = np.concatenate(vector_batches, axis=0)
    if normalize:
        all_vectors = _l2_normalize(all_vectors).astype("float32", copy=False)

    db = create_vector_db(backend=backend, metric=metric, dim=dim)
    db.add(all_vectors)
    db.save(index_path)

    frame_map = {
        str(rec["faiss_index"]): _frame_map_entry(rec, rec["faiss_index"])
        for rec in all_records
    }
    Path(frame_map_path).parent.mkdir(parents=True, exist_ok=True)
    with Path(frame_map_path).open("w", encoding="utf-8") as f:
        json.dump(frame_map, f, ensure_ascii=False, indent=2)

    report = {
        "index_path": str(index_path),
        "frame_map_path": str(frame_map_path),
        "backend": backend,
        "metric": metric,
        "vector_dim": dim,
        "total_vectors": int(all_vectors.shape[0]),
        "ntotal": db.ntotal,
        "normalized": normalize,
    }

    if neighbor_index_path is not None:
        report["neighbor"] = build_neighbor_index(
            frame_map_path=frame_map_path,
            output_path=neighbor_index_path,
            max_neighbors=max_neighbors,
        )

    logger.info("Index built: ntotal=%d dim=%s -> %s", db.ntotal, dim, index_path)
    return report


# ---------------------------------------------------------------------------
# Load bundle cho serving
# ---------------------------------------------------------------------------

@dataclass
class IndexBundle:
    index_path: Path
    frame_map_path: Path
    neighbor_index_path: Path | None = None

    def exists(self) -> dict:
        return {
            "index": self.index_path.exists(),
            "frame_map": self.frame_map_path.exists(),
            "neighbor": self.neighbor_index_path.exists() if self.neighbor_index_path else None,
        }


def main() -> None:  # pragma: no cover - CLI
    import argparse

    parser = argparse.ArgumentParser(description="Build vector index + frame_map (+ neighbor)")
    parser.add_argument("--embeddings-glob", default="data/embeddings/openclip_vit_b16_*.npy")
    parser.add_argument(
        "--embedding-metadata-template",
        default="data/metadata/openclip_vit_b16_embeddings_{video_id}.jsonl",
    )
    parser.add_argument("--embeddings-prefix", default="openclip_vit_b16_")
    parser.add_argument("--embeddings-suffix", default="")
    parser.add_argument("--index-path", default="data/indexes/openclip_vit_b16_flat_ip.faiss")
    parser.add_argument("--frame-map-path", default="data/metadata/openclip_vit_b16_frame_map.json")
    parser.add_argument("--metric", default="ip", choices=("ip", "l2"))
    parser.add_argument("--backend", default="faiss", choices=("faiss", "numpy"))
    parser.add_argument("--skip-normalize", action="store_true")
    parser.add_argument("--neighbor-index-path", default=None)
    parser.add_argument("--max-neighbors", type=int, default=4)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level, format="%(levelname)s %(name)s: %(message)s")
    report = build_index(
        embeddings_glob=args.embeddings_glob,
        embedding_metadata_template=args.embedding_metadata_template,
        index_path=args.index_path,
        frame_map_path=args.frame_map_path,
        embeddings_prefix=args.embeddings_prefix,
        embeddings_suffix=args.embeddings_suffix,
        metric=args.metric,
        backend=args.backend,
        normalize=not args.skip_normalize,
        neighbor_index_path=args.neighbor_index_path,
        max_neighbors=args.max_neighbors,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


__all__ = ["build_index", "IndexBundle"]
