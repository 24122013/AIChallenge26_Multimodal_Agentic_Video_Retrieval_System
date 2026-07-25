# Báo cáo index size và latency

## Phạm vi và giả định

Báo cáo này chỉ đánh giá tầng build index và metadata. Không thay đổi query
understanding, retrieval, ranking, reranking, API hay UI.

Repository không chứa dataset/index artifact thực tế (`data/` chỉ có file
`.gitkeep`), vì vậy số liệu bên dưới là **synthetic micro-benchmark**, không phải
benchmark chất lượng retrieval. FAISS index và embedding model không được thay
đổi hoặc benchmark lại.

Các giả định triển khai:

- `frame_id` là tên chuẩn của keyframe ID theo schema hiện tại.
- `segment_id`, fallback `shot_id`, cùng `shot_start/shot_end` là boundary chuẩn.
- Khi không có boundary, `--strategy fixed` hoặc chế độ `auto` fallback sang
  fixed-duration window.
- Timestamp trong record được ưu tiên. Chỉ khi thiếu timestamp mới dùng
  `frame_index / fps`; FPS theo record (`fps`/`video_fps`) được ưu tiên trước
  `--fps`.
- Neighbor chỉ giữ `frame_id` và `delta_seconds`; các field lớn hơn được resolve
  từ frame map chuẩn. Đây là lựa chọn ID/reference để tránh copy metadata.
- Nếu object không có `track_id`, `occurrence_count` là số detection occurrence,
  không phải số object duy nhất.

## Trước khi tối ưu

Pipeline hiện tại có các artifact chính:

- Keyframe JSONL với `frame_id`, video/shot/segment ID, timestamp, frame index và
  nhiều đường dẫn.
- Caption, OCR, object và ASR là các JSONL frame/chunk-level độc lập.
- Embedding `.npy` là `float32`; FAISS dùng `IndexFlatIP`; `embedding_index` ánh
  xạ sang metadata và frame map JSON.
- Frame map lặp lại nhiều field từ embedding metadata và được pretty-print.
- `build_faiss_index.py` giữ các vector batch, `index_records`, vector nối và
  frame map trong RAM trong lúc build.
- Neighbor được suy ra tại runtime bằng cách quét frame cùng video; chưa có
  artifact neighbor theo cửa sổ timestamp.
- Hai placeholder `neighbor_index.py` và `extract_segments.py` chưa có
  implementation segment-level hoàn chỉnh.

Frame-level artifact vẫn là nguồn provenance và không bị xóa hoặc sửa schema.

## Bottleneck tìm thấy

1. Copy nguyên metadata keyframe vào mỗi neighbor sẽ nhân bản path, timestamp và
   các field model nhiều lần theo độ rộng cửa sổ.
2. Pretty-print frame map làm tăng kích thước production artifact dù whitespace
   không thuộc contract.
3. Quét tuyến tính toàn bộ frame artifact cho mỗi segment có độ phức tạp gần
   `O(segment_count × artifact_count)`.
4. Load dataset lớn vào list Python làm tăng peak memory. Riêng neighbor builder
   có thể tránh việc này hoàn toàn bằng staging theo luồng.
5. FAISS builder hiện vẫn tạo nhiều representation in-memory của vector và
   metadata. Đây là bottleneck còn lại, chưa đổi vì cần một benchmark index thật
   để xác nhận peak RAM và tính tương thích artifact.

## Thay đổi đã thực hiện

- Neighbor builder stream JSONL vào SQLite tạm, có composite primary key
  `(video_id, frame_id)` và index `(video_id, timestamp, frame_index, frame_id)`.
- Range query luôn có `video_id`, loại center frame, xử lý đầu/cuối video và
  sort deterministic.
- Output được ghi compact và atomic; duplicate đồng nhất được bỏ, duplicate xung
  đột báo lỗi.
- Segment builder dùng shot/segment boundary hiện có trước; fixed window là
  fallback cấu hình được.
- Caption được sort theo thời gian, normalize và loại exact/near duplicate.
- OCR được normalize Unicode/whitespace, merge text lặp, giữ max confidence,
  `first_seen`, `last_seen` và `source_ids`.
- ASR dùng interval overlap, giữ thứ tự, merge chunk trùng do overlap nhưng lưu
  mọi `source_intervals` gốc.
- Object được normalize label, giữ max confidence và đếm theo unique track khi
  có; nếu không có track, schema ghi rõ `detection_occurrence`.
- Caption/OCR/object được lập lookup map theo `frame_id`/`segment_id` một lần,
  không quét toàn bộ artifact cho từng segment.
- Frame map và JSONL indexing mới được serialize compact. Schema JSON không đổi,
  nên reader cũ vẫn tương thích.
- Backend placeholder chỉ re-export implementation chuẩn tại `src/indexing`;
  không tồn tại hai pipeline khác nhau.

## Schema trước và sau

Trước: mỗi modality là artifact frame/chunk-level riêng; chưa có neighbor artifact
và chưa có một segment record tổng hợp đầy đủ.

Sau, neighbor record:

```json
{
  "schema_version": "1.0",
  "video_id": "L01_V001",
  "frame_id": "FRAME_L01_V001_000010",
  "frame_index": 300,
  "timestamp": 10.0,
  "timestamp_source": "metadata",
  "neighbors_before": [
    {"frame_id": "FRAME_L01_V001_000009", "delta_seconds": -1.0}
  ],
  "neighbors_after": [
    {"frame_id": "FRAME_L01_V001_000011", "delta_seconds": 1.0}
  ]
}
```

Sau, segment record:

```json
{
  "schema_version": "1.0",
  "segment_id": "SHOT_L01_V001_000003",
  "video_id": "L01_V001",
  "start_time": 10.0,
  "end_time": 15.0,
  "start_frame": 300,
  "end_frame": 425,
  "start_keyframe": "FRAME_L01_V001_000010",
  "end_keyframe": "FRAME_L01_V001_000014",
  "keyframe_ids": ["FRAME_L01_V001_000010"],
  "captions_aggregated": "A person walks beside a car.",
  "caption_source_ids": ["FRAME_L01_V001_000010"],
  "ocr": [],
  "asr": [],
  "objects": []
}
```

`ocr`, `asr` và `objects` chứa `source_ids`; ASR còn có `source_intervals`.
Metadata frame-level cũ được giữ nguyên để truy vết.

## Micro-benchmark trước/sau

Lệnh:

```powershell
.\.venv\Scripts\python.exe -m reports.benchmark_indexing_metadata `
  --output reports/index_size_latency_benchmark.json `
  --videos 4 `
  --frames-per-video 250 `
  --runs 3 `
  --lookup-queries 500
```

Dataset synthetic deterministic:

- 4 video, 250 keyframe/video, tổng 1.000 keyframe;
- FPS 25, keyframe cách nhau 1 giây, 5 keyframe/shot;
- 1.000 caption, 336 OCR record, 1.000 object record, 288 ASR chunk;
- neighbor window 5 giây;
- Python 3.13.9, Windows 11 build 26100, AMD64 Family 25 Model 68.

Mỗi build và lookup chạy 3 lần. Runtime dùng median và ghi min/max; memory là
median peak allocation do `tracemalloc` đo. Số này không gồm native allocation
của SQLite/FAISS.

| Chỉ số | Trước/counterfactual | Sau | Thay đổi |
|---|---:|---:|---:|
| Frame-map serialization | 496.433 B pretty | 410.431 B compact | -17,32% |
| Neighbor artifact | 4.393.104 B nếu copy full metadata | 781.248 B ID/reference | -82,22% |
| Segment metadata | Chưa có | 357.444 B | Artifact mới |
| Frame-level metadata hiện có | 887.408 B | 887.408 B | Giữ nguyên |
| Tổng metadata với artifact mới | 5.637.956 B nếu expanded | 2.026.100 B compact | -64,06% tổng |
| Neighbor build | Chưa có | median 212,86 ms (202,75–215,74) | N/A |
| Segment build | Chưa có | median 364,91 ms (350,00–401,10) | N/A |
| Neighbor peak Python memory | Chưa có | 0,052 MB | SQLite streaming |
| Segment peak Python memory | Chưa có | 4,213 MB | Aggregation in-memory |
| 500 metadata lookup | 26,273 ms linear | 0,094 ms ID map | ~278,32× |

Vì thêm hai capability mới, tổng metadata thật tăng từ 887.408 B lên
2.026.100 B (+128,32%). So sánh hợp lý cho tối ưu là với thiết kế expanded cho
cùng capability: phần artifact mới compact nhỏ hơn 76,03%. Không có FAISS
artifact thật nên **tổng kích thước vector index trước/sau là N/A**; vector index
không đổi.

Raw result: `reports/index_size_latency_benchmark.json`.

## Trade-off

- SQLite staging giảm Python peak memory và hỗ trợ mixed-video input, đổi lại
  thêm I/O đĩa tạm khi build.
- ID/reference giảm mạnh kích thước neighbor nhưng consumer cần frame map để
  resolve timestamp/path của neighbor.
- Near-duplicate caption threshold cấu hình được; threshold quá thấp có thể gộp
  hai caption khác ý. Mặc định 0,92 là bảo thủ.
- Segment aggregation giữ provenance nên lớn hơn output chỉ chứa text tổng hợp.
- Segment builder giữ representation tối thiểu và các modality lookup map trong
  RAM. Đây là lựa chọn cân bằng; neighbor builder là phần hoàn toàn streaming.
- Compact JSON khó đọc bằng mắt hơn pretty JSON nhưng không đổi schema hoặc
  parser contract.

## Tương thích và migration

Không cần migration phá hủy. Hãy rebuild hai artifact mới từ metadata hiện có.
Frame map compact vẫn là JSON object giống schema trước; mọi reader hiện tại đọc
được. Retrieval API hiện tại không được thay đổi để bắt buộc dùng artifact mới.

Lệnh end-to-end cho một video:

```powershell
.\.venv\Scripts\python.exe -m src.indexing.build_neighbor_index `
  --input data/metadata/keyframes_L01_V001.jsonl `
  --output data/metadata/neighbors_L01_V001.jsonl `
  --window-seconds 5

.\.venv\Scripts\python.exe -m src.indexing.build_segment_metadata `
  --input data/metadata/keyframes_L01_V001.jsonl `
  --captions data/metadata/captions_L01_V001.jsonl `
  --ocr data/metadata/ocr_L01_V001.jsonl `
  --asr data/metadata/asr_L01_V001.jsonl `
  --objects data/metadata/objects_L01_V001.jsonl `
  --output data/metadata/segments_L01_V001.jsonl `
  --strategy auto
```

Để build toàn bộ folder, truyền `data/metadata` cho `--input` và từng option
modality; tool tự chọn `keyframes_*.jsonl`, `captions_*.jsonl`, `ocr_*.jsonl`,
`asr_*.jsonl` (không lấy `asr_segments_*`) và `objects_*.jsonl`.

## Kiểm thử

- `.\.venv\Scripts\python.exe -m unittest discover -s backend/tests -v`
- Kết quả: 28 test chạy thành công, 1 smoke test SigLIP2 skip có chủ đích vì cần
  `RUN_SIGLIP2_SMOKE=1` và checkpoint thật.
- Lần chạy đầu bằng Python hệ thống không import được 5 module cũ do thiếu
  NumPy/Pillow/OpenCV. Chạy lại bằng `.venv` của repo đã qua toàn bộ suite.

## Future work

- Đo trên dataset/index thật, gồm `.faiss`, `.npy`, frame map, wall-clock,
  process RSS và peak native memory.
- Refactor FAISS build sang validate hai pass hoặc add vector theo batch để tránh
  giữ `vector_batches + all_vectors` cùng lúc; chỉ làm sau khi có benchmark artifact
  thật để kiểm tra byte/contract compatibility.
- Cân nhắc SQLite/Parquet persistent cho segment lookup khi metadata vượt khả
  năng RAM.
- Chỉ tích hợp neighbor/segment artifact vào retrieval ở giai đoạn sau; không
  thay đổi ranking/reranking trong task này.
