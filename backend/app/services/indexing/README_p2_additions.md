# Indexing — các module bổ sung (Team P2)

Bổ sung cho baseline OpenCLIP + FAISS đã có. Model nặng (open_clip/torch/faiss)
đều *lazy import*; có backend NumPy để chạy/test không cần faiss.

| File | Vai trò |
|---|---|
| `embedding_factory.py` | Factory chọn/nạp model embedding (OpenCLIP / CLIP / SigLIP) + hàm `encode_keyframe_records` dùng chung |
| `build_clip_index.py` | CLI encode keyframe bằng CLIP (OpenAI) |
| `build_sigclip_index.py` | CLI encode keyframe bằng SigLIP |
| `vector_db.py` | Abstraction vector index: `FaissVectorDB` (faiss) + `NumpyVectorDB` (brute-force) |
| `neighbor_index.py` | Tiền tính frame lân cận cùng shot từ `frame_map.json` |
| `extract_segments.py` | Sinh segment record từ keyframe metadata (gom theo shot / window) |
| `index_manager.py` | Điều phối: embeddings → vector index + frame_map (+ neighbor index) |

## Ví dụ

```bash
export PYTHONPATH=.

# 1) Encode bằng model thay thế (cần open_clip + torch)
python -B backend/app/services/indexing/build_clip_index.py \
  --metadata-path data/metadata/keyframes_L01_V001.jsonl \
  --embeddings-path data/embeddings/clip_vit_b16_L01_V001.npy \
  --embedding-metadata-path data/metadata/clip_vit_b16_embeddings_L01_V001.jsonl \
  --kind clip

# 2) Build index + frame_map + neighbor (backend numpy chạy không cần faiss)
python -B backend/app/services/indexing/index_manager.py \
  --embeddings-glob "data/embeddings/openclip_vit_b16_*.npy" \
  --embedding-metadata-template "data/metadata/openclip_vit_b16_embeddings_{video_id}.jsonl" \
  --index-path data/indexes/openclip_vit_b16_flat_ip.faiss \
  --frame-map-path data/metadata/openclip_vit_b16_frame_map.json \
  --neighbor-index-path data/metadata/openclip_vit_b16_neighbor_index.json \
  --backend faiss

# 3) Segment + neighbor riêng lẻ
python -B backend/app/services/indexing/extract_segments.py \
  --metadata-path data/metadata/keyframes_L01_V001.jsonl \
  --output-path data/segments/segments_L01_V001.jsonl

python -B backend/app/services/indexing/neighbor_index.py \
  --frame-map-path data/metadata/openclip_vit_b16_frame_map.json \
  --output-path data/metadata/openclip_vit_b16_neighbor_index.json
```

## Test

```bash
python -B -m unittest backend.tests.test_indexing_p2
```
