"""build_sigclip_index — encode keyframe bằng SigLIP rồi lưu embeddings.

Song song với build_openclip_index nhưng dùng embedding_factory nên đổi model chỉ
là đổi --kind / --model-name. Output khớp build_openclip_index để build_faiss_index
/ index_manager dùng lại được.

Ví dụ:
    python -B backend/app/services/indexing/build_clip_index.py \
        --metadata-path data/metadata/keyframes_L01_V001.jsonl \
        --embeddings-path data/embeddings/sigclip_vit_b16_L01_V001.npy \
        --embedding-metadata-path data/metadata/sigclip_vit_b16_embeddings_L01_V001.jsonl \
        --kind sigclip
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from backend.app.services.indexing.embedding_factory import (
    create_embedding_model,
    encode_keyframe_records,
    save_encode_artifacts,
)


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Encode keyframes with SigLIP")
    parser.add_argument("--metadata-path", type=Path, required=True)
    parser.add_argument("--embeddings-path", type=Path, required=True)
    parser.add_argument("--embedding-metadata-path", type=Path, required=True)
    parser.add_argument("--skipped-path", type=Path, default=None)
    parser.add_argument("--benchmark-path", type=Path, default=None)
    parser.add_argument("--kind", default="sigclip", help="sigclip | siglip-l16 | ...")
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--pretrained", default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--model-cache-dir", type=Path, default=Path("data/model_cache/sigclip"))
    args = parser.parse_args()

    records = _load_jsonl(args.metadata_path)
    if not records:
        raise SystemExit(f"No records in {args.metadata_path}")

    model = create_embedding_model(
        kind=args.kind,
        model_name=args.model_name,
        pretrained=args.pretrained,
        device=args.device,
        model_cache_dir=args.model_cache_dir,
    )
    artifacts = encode_keyframe_records(records, model, batch_size=args.batch_size)
    save_encode_artifacts(
        artifacts,
        embeddings_path=args.embeddings_path,
        embedding_metadata_path=args.embedding_metadata_path,
        skipped_path=args.skipped_path,
        benchmark_path=args.benchmark_path,
    )
    print(json.dumps(artifacts.benchmark, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
