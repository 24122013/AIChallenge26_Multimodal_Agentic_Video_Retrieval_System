# Metadata Schema v1.2

`frame_map.json` là metadata chính cho retrieval. Phase 2 bổ sung encoder
contract để embedding, FAISS index, frame map và query encoder có thể được kiểm
tra đồng bộ.

## Keyframe metadata

```json
{
  "frame_id": "FRAME_L01_V001_000001",
  "video_id": "L01_V001",
  "shot_id": "SHOT_L01_V001_000001",
  "segment_id": "SEG_L01_V001_000001",
  "shot_index": 1,
  "shot_start": 0.0,
  "shot_end": 3.5,
  "timestamp": 1.25,
  "timestamp_source": "video_fps",
  "timestamp_confidence": 1.0,
  "frame_index": 37,
  "keyframe_path": "data/keyframes/L01_V001/000001.jpg",
  "frame_path": "data/keyframes/L01_V001/000001.jpg",
  "thumbnail_path": "data/keyframes/L01_V001/000001.jpg",
  "source_video_path": "data/raw/video/L01_V001.mp4",
  "video_path": "data/raw/video/L01_V001.mp4",
  "selection_reason": "shot_midpoint"
}
```

Các field nhận dạng và đường dẫn phải deterministic. `timestamp_source` có thể
là `video_fps`, `matched_frame`, `interval` hoặc `unknown`.

## SigLIP2 embedding metadata

Mỗi dòng trong
`siglip2_so400m_patch16_384_embeddings_<video_id>.jsonl` tương ứng chính xác với
một row trong file `.npy`.

```json
{
  "embedding_id": "EMB_FRAME_L01_V001_000001",
  "frame_id": "FRAME_L01_V001_000001",
  "video_id": "L01_V001",
  "shot_id": "SHOT_L01_V001_000001",
  "segment_id": "SEG_L01_V001_000001",
  "shot_index": 1,
  "shot_start": 0.0,
  "shot_end": 3.5,
  "timestamp": 1.25,
  "timestamp_source": "video_fps",
  "timestamp_confidence": 1.0,
  "frame_index": 37,
  "keyframe_path": "data/keyframes/L01_V001/000001.jpg",
  "thumbnail_path": "data/keyframes/L01_V001/000001.jpg",
  "source_video_path": "data/raw/video/L01_V001.mp4",
  "video_path": "data/raw/video/L01_V001.mp4",
  "selection_reason": "shot_midpoint",
  "embedding_index": 0,
  "model_family": "siglip2",
  "model_name": "google/siglip2-so400m-patch16-384",
  "model_revision": "<resolved-or-requested-revision>",
  "processor_name": "google/siglip2-so400m-patch16-384",
  "vector_dim": 1152,
  "input_resolution": 384,
  "normalized": true,
  "similarity": "cosine",
  "output_dtype": "float32"
}
```

`vector_dim` trong ví dụ là giá trị dự kiến của checkpoint. Code phải lấy giá
trị từ `features.shape[-1]` và đối chiếu với projection dimension trong model
config khi config cung cấp field đó.

Invariant trong một file:

- `embedding_index` liên tục từ 0 và bằng row offset;
- số metadata records bằng `embeddings.shape[0]`;
- mọi record có cùng `model_name`, `model_revision`, `vector_dim` và
  `normalized`;
- `vector_dim == embeddings.shape[1]`;
- embedding là finite `float32`, không zero và có L2 norm xấp xỉ 1.

## Skipped image metadata

Ảnh không đọc được không tạo vector:

```json
{
  "frame_id": "FRAME_L01_V001_000002",
  "video_id": "L01_V001",
  "keyframe_path": "data/keyframes/L01_V001/000002.jpg",
  "skip_reason": "image_load_error",
  "error": "cannot identify image file"
}
```

Sau khi bỏ ảnh lỗi, `embedding_index` của các ảnh hợp lệ vẫn liên tục.

## Benchmark metadata

```json
{
  "model_family": "siglip2",
  "model_name": "google/siglip2-so400m-patch16-384",
  "model_revision": "<resolved-or-requested-revision>",
  "processor_name": "google/siglip2-so400m-patch16-384",
  "vector_dim": 1152,
  "input_resolution": 384,
  "device": "cuda",
  "compute_dtype": "float16",
  "output_dtype": "float32",
  "normalized": true,
  "requested_batch_size": "auto",
  "selected_batch_size": 8,
  "batch_tuning_results": [],
  "num_workers": 4,
  "prefetch_factor": 2,
  "input_record_count": 1000,
  "encoded_count": 998,
  "skipped_count": 2,
  "embedding_shape": [998, 1152],
  "runtime_sec": 0.0,
  "image_load_sec": 0.0,
  "inference_sec": 0.0,
  "throughput_img_per_sec": 0.0,
  "peak_gpu_memory_mb": 0.0,
  "torch_version": "<version>",
  "transformers_version": "<version>"
}
```

## FAISS frame map

Top-level key là string của `faiss_index`. Các field retrieval cũ được giữ, và
v1.2 bổ sung `model_name`, `model_revision`, `vector_dim`.

```json
{
  "0": {
    "frame_id": "FRAME_L01_V001_000001",
    "video_id": "L01_V001",
    "shot_id": "SHOT_L01_V001_000001",
    "segment_id": "SEG_L01_V001_000001",
    "shot_index": 1,
    "shot_start": 0.0,
    "shot_end": 3.5,
    "timestamp": 1.25,
    "timestamp_source": "video_fps",
    "timestamp_confidence": 1.0,
    "frame_index": 37,
    "keyframe_path": "data/keyframes/L01_V001/000001.jpg",
    "thumbnail_path": "data/keyframes/L01_V001/000001.jpg",
    "source_video_path": "data/raw/video/L01_V001.mp4",
    "video_path": "data/raw/video/L01_V001.mp4",
    "embedding_id": "EMB_FRAME_L01_V001_000001",
    "embedding_index": 0,
    "selection_reason": "shot_midpoint",
    "model_name": "google/siglip2-so400m-patch16-384",
    "model_revision": "<resolved-or-requested-revision>",
    "vector_dim": 1152
  }
}
```

`MetadataStore` tiếp tục đọc được frame map v1.1; các encoder field mới là
optional đối với legacy artifact nhưng bắt buộc với artifact SigLIP2 mới.

## FAISS manifest

```json
{
  "schema_version": "1.2",
  "encoder": {
    "model_family": "siglip2",
    "model_name": "google/siglip2-so400m-patch16-384",
    "model_revision": "<resolved-or-requested-revision>",
    "processor_name": "google/siglip2-so400m-patch16-384",
    "vector_dim": 1152,
    "input_resolution": 384,
    "normalized": true,
    "similarity": "cosine",
    "output_dtype": "float32"
  },
  "index_type": "IndexFlatIP",
  "metric": "ip",
  "vector_count": 0,
  "metadata_record_count": 0,
  "index_path": "data/indexes/siglip2_so400m_patch16_384_flat_ip.faiss",
  "frame_map_path": "data/metadata/siglip2_so400m_patch16_384_frame_map.json",
  "sources": []
}
```

Retrieval phải đọc `encoder` trước khi load text encoder. Không được encode
query bằng OpenCLIP rồi search trong index SigLIP2.

## Metadata ngoài Phase 2

Caption, OCR, ASR và objects giữ schema hiện tại và không được tạo trong Phase
2. Các pipeline OpenCLIP/artifact cũ cũng không bị xóa hoặc overwrite.
