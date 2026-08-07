# AIChallenge26 Multimodal Agentic Video Retrieval System

Repo này xây dựng baseline video retrieval cho kì thi: video -> shot-aware keyframes -> SigLIP2 embeddings -> FAISS -> search trả frame và frame lân cận cùng shot.

## Pipeline hiện tại

```text
data/raw/video/*.mp4
  -> TransNetV2 shot detection
  -> keyframe sampling + FFmpeg frame extraction + pHash dedup
  -> keyframe metadata JSONL
  -> SigLIP2 image embeddings
  -> FAISS IndexFlatIP + frame_map + encoder manifest
  -> text query -> SigLIP2 text embedding -> FAISS top-k -> results + same-shot neighbors
```

Indexing và retrieval mặc định dùng `google/siglip2-so400m-patch16-384`. Retrieval đọc model name, revision và vector dimension từ FAISS manifest để bảo đảm text query dùng đúng embedding space. FAISS dùng `IndexFlatIP` với vector đã normalize, tức inner product hoạt động như cosine similarity.

Pipeline OpenCLIP cũ vẫn được giữ làm legacy baseline và cho tùy chọn CLIP dedup.

## Cài đặt

Chạy từ root repo bằng PowerShell:

```powershell
py -m venv .venv; .\.venv\Scripts\python.exe -m pip install --upgrade pip; .\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

`requirements.txt` đã gồm các thư viện chính: OpenCV, PyTorch, Transformers, OpenCLIP, FAISS CPU, Pillow, TransNetV2 PyTorch, FastAPI/Uvicorn.

Lưu ý: `ffmpeg-python` chỉ là Python wrapper, không tự cài binary FFmpeg. TransNetV2 PyTorch cần `ffmpeg.exe` trong `PATH`.

Kiểm tra:

```powershell
ffmpeg -version
```

Nếu lệnh trên không chạy, cài FFmpeg system trước khi chạy pipeline. Repo chỉ dùng TransNetV2 cho shot detection; thiếu FFmpeg thì extractor sẽ báo lỗi và dừng.

## Chuẩn bị video

Đặt video vào:

```text
data/raw/video/
```

Ví dụ:

```text
data/raw/video/L27_V001.mp4
data/raw/video/L27_V002.mp4
```

Folder dữ liệu lớn đã nằm trong `.gitignore`, không commit video/keyframes/embeddings/index.

## Bước 1: Extract keyframes

Chạy toàn bộ video:

```powershell
.\.venv\Scripts\python.exe -B backend\app\services\indexing\extract_keyframes.py --video-dir data\raw\video --video-glob *.mp4 --output-dir data\keyframes --shot-device auto --shot-threshold 0.5
```

Output cho mỗi video:

```text
data/keyframes/<video_id>/*.jpg
data/metadata/keyframes_<video_id>.jsonl
data/metadata/keyframes_<video_id>_extract_report.json
```

Rule keyframe:

- Shot `duration <= 2s`: lấy midpoint.
- Shot `2s < duration <= 4s`: lấy 2 frame tại 1/3 và 2/3 shot.
- Shot `duration > 4s`: lấy một frame mỗi 2s theo centered sampling (`start+1s`, `+3s`, ...).
- Extract frame bằng FFmpeg theo timestamp đã chọn.
- Conservative dedup chỉ so sánh frame cùng shot và cách nhau tối đa 2s; pHash mặc định dùng Hamming distance <= 6.
- Metadata lưu `timestamp`, `frame_index`, `shot_start`, `shot_end`, `shot_id`, `source_video_path`, `keyframe_strategy`.

Nếu muốn bật CLIP dedup gần nhau:

```powershell
.\.venv\Scripts\python.exe -B backend\app\services\indexing\extract_keyframes.py --video-dir data\raw\video --enable-clip-dedup --clip-similarity-threshold 0.985
```

## Bước 2: Encode keyframes bằng SigLIP2

Chạy cho toàn bộ metadata keyframe:

```powershell
Get-ChildItem data\metadata\keyframes_*.jsonl | ForEach-Object { .\.venv\Scripts\python.exe -B backend\app\services\indexing\build_siglip2_index.py --metadata-path $_.FullName --batch-size auto --num-workers 4 --device auto }
```

Output chính:

```text
data/embeddings/siglip2_so400m_patch16_384_<video_id>.npy
data/metadata/siglip2_so400m_patch16_384_embeddings_<video_id>.jsonl
data/metadata/siglip2_so400m_patch16_384_skipped_<video_id>.jsonl
data/metadata/siglip2_so400m_patch16_384_benchmark_<video_id>.json
```

## Bước 3: Build FAISS index

```powershell
.\.venv\Scripts\python.exe -B backend\app\services\indexing\build_faiss_index.py --embeddings-glob "data/embeddings/siglip2_so400m_patch16_384_*.npy" --embedding-metadata-template "data/metadata/siglip2_so400m_patch16_384_embeddings_{video_id}.jsonl" --embeddings-prefix "siglip2_so400m_patch16_384_" --index-path data\indexes\siglip2_so400m_patch16_384_flat_ip.faiss --index-metadata-path data\metadata\siglip2_so400m_patch16_384_faiss_metadata.jsonl --frame-map-path data\metadata\siglip2_so400m_patch16_384_frame_map.json --manifest-path data\metadata\siglip2_so400m_patch16_384_faiss_manifest.json --report-path data\metadata\siglip2_so400m_patch16_384_index_report.json
```

Output retrieval cần có:

```text
data/indexes/siglip2_so400m_patch16_384_flat_ip.faiss
data/metadata/siglip2_so400m_patch16_384_frame_map.json
data/metadata/siglip2_so400m_patch16_384_faiss_manifest.json
```

## Bước 4: Test retrieval bằng Python

Retrieval tự đọc manifest để load đúng SigLIP2 checkpoint và kiểm tra query vector dimension trước khi search.

```powershell
.\.venv\Scripts\python.exe -c "from backend.app.api.search import search; import json; print(json.dumps(search('a person cooking', top_k=5), ensure_ascii=False, indent=2))"
```

Mỗi result sẽ có:

- `video_id`
- `frame_id`
- `timestamp`
- `keyframe_path`
- `shot_id`
- `score`
- `neighbors`: các keyframe lân cận cùng shot để UI hiển thị thêm ngữ cảnh.

## API

Repo hiện có router/wrapper trong `backend/app/api/search.py`, nhưng chưa có file app tổng kiểu `main.py` tạo `FastAPI()` và `include_router(...)`.

Vì vậy cách test retrieval chắc chắn nhất hiện tại là gọi trực tiếp Python wrapper ở bước trên. Khi team thêm app tổng, router search có thể được include từ `backend.app.api.search`.

## Kiểm tra nhanh

```powershell
.\.venv\Scripts\python.exe -B -m unittest discover -s backend\tests -p "test_*.py"; .\.venv\Scripts\python.exe -B backend\app\services\indexing\extract_keyframes.py --help
```

## Cấu trúc quan trọng

```text
backend/app/services/indexing/extract_keyframes.py       # TransNetV2 + keyframe sampling + dedup
backend/app/services/indexing/build_siglip2_index.py     # encode ảnh keyframe thành SigLIP2 embeddings
backend/app/services/indexing/build_faiss_index.py       # gom embeddings thành FAISS + frame_map
backend/app/services/retrieval/search_visual.py          # text query -> SigLIP2 -> FAISS -> results
backend/app/services/metadata/metadata_store.py          # lookup frame_map và same-shot neighbors
docs/keyframe_extraction.md                              # giải thích chi tiết chiến lược keyframe
```

## Ghi chú cho team

- Dùng `.\.venv\Scripts\python.exe`, không dùng `python` global nếu máy có nhiều Python.
- `siglip2_so400m_patch16_384_frame_map.json`, FAISS index và manifest phải được build cùng một encoder contract.
- Sau khi extract lại keyframes thì cần encode lại embeddings và build lại FAISS.
- Extractor chỉ dùng TransNetV2. Nếu thiếu FFmpeg hoặc TransNetV2 lỗi, sửa môi trường rồi chạy lại.

## Multimodal metadata ingestion

Bốn pipeline trong `backend/app/services/ingestion/` sinh artifact JSONL riêng,
không sửa metadata keyframe gốc, SigLIP2 embeddings hoặc FAISS:

- Caption: `Salesforce/blip-image-captioning-base`, baseline gọn, có batch GPU
  và sinh caption tiếng Anh trực tiếp bằng Transformers.
- OCR: EasyOCR với `vi` + `en`, phù hợp baseline song ngữ, giữ Unicode và trả
  polygon/confidence.
- Objects: Ultralytics YOLO11n (`yolo11n.pt`), checkpoint COCO nhỏ và dễ tái lập.
- ASR: `faster-whisper` model `small`, auto language + VAD; có fallback sang
  `openai-whisper` nếu package đó đã được cài.

### Cài dependency

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
ffprobe -version
```

`ffmpeg-python` không cung cấp binary. Cần cài FFmpeg system sao cho cả
`ffmpeg.exe` và `ffprobe.exe` có trong `PATH`. Model được lazy-load sau khi CLI
parse xong; cache nằm dưới `data/model_cache/` và đã được Git ignore.

### Chạy một video

```powershell
.\.venv\Scripts\python.exe -B backend\app\services\ingestion\run_caption.py --metadata-path data\metadata\keyframes_L27_V001.jsonl --device auto --batch-size 4 --segment-caption
.\.venv\Scripts\python.exe -B backend\app\services\ingestion\run_ocr.py --metadata-path data\metadata\keyframes_L27_V001.jsonl --device auto --batch-size 4 --conf-threshold 0.3
.\.venv\Scripts\python.exe -B backend\app\services\ingestion\run_object_detection.py --metadata-path data\metadata\keyframes_L27_V001.jsonl --device auto --batch-size 8 --conf-threshold 0.25 --iou-threshold 0.7
.\.venv\Scripts\python.exe -B backend\app\services\ingestion\run_asr.py --video-path data\raw\video\L27_V001.mp4 --metadata-path data\metadata\keyframes_L27_V001.jsonl --device auto --backend auto --model-size small
```

### Chạy toàn dataset

CLI nhận trực tiếp thư mục. Chúng tự chọn `keyframes_*.jsonl` hoặc `*.mp4`:

```powershell
.\.venv\Scripts\python.exe -B backend\app\services\ingestion\run_caption.py --metadata-path data\metadata --device auto --batch-size 4
.\.venv\Scripts\python.exe -B backend\app\services\ingestion\run_ocr.py --metadata-path data\metadata --device auto --batch-size 4
.\.venv\Scripts\python.exe -B backend\app\services\ingestion\run_object_detection.py --metadata-path data\metadata --device auto --batch-size 8
.\.venv\Scripts\python.exe -B backend\app\services\ingestion\run_asr.py --video-path data\raw\video --metadata-path data\metadata --video-glob "*.mp4" --device auto
```

Output mặc định:

```text
data/metadata/captions_<video_id>.jsonl
data/metadata/ocr_<video_id>.jsonl
data/metadata/objects_<video_id>.jsonl
data/metadata/asr_<video_id>.jsonl
data/metadata/asr_segments_<video_id>.jsonl
data/metadata/<artifact>_<video_id>_report.json
```

Mặc định, record đã có trong output được skip để resume an toàn. Dùng
`--overwrite` để tạo lại artifact của pipeline đó. Record lỗi ảnh/model được ghi
riêng và không dừng batch; frame không có text/object vẫn là success. ASR dùng
`ffprobe`: video không có audio là `skipped/no_audio_stream`, không phải error.

Schema đầy đủ và JSON mẫu nằm trong `docs/metadata_schema.md`.

### Tài nguyên và giới hạn

Ước lượng thực hành: BLIP base khoảng 2 GB VRAM, EasyOCR dưới 2 GB, YOLO11n
dưới 1 GB và faster-whisper small khoảng 2–3 GB; RAM CPU nên có tối thiểu
8 GB. Batch mặc định phù hợp GPU 8 GB nhưng nên giảm khi ảnh lớn hoặc CUDA OOM.
BLIP và YOLO COCO có thể bỏ sót chữ nhỏ, vật thể miền chuyên biệt hoặc chi tiết
tiếng Việt. EasyOCR không trả language tin cậy theo từng vùng nên artifact ghi
tập ngôn ngữ cấu hình. Confidence ASR được suy ra từ `avg_logprob`, không phải
xác suất đã calibration. `segment_caption` là phép khử trùng lặp caption frame,
không phải một lượt suy luận video-temporal.

## Neighbor index và segment-level metadata

Hai bước này chỉ tạo thêm artifact indexing/metadata, không sửa metadata keyframe
gốc. Neighbor index là tùy chọn; segment metadata phải được build trước text
index. Thứ tự chạy đầy đủ được khuyến nghị:

```text
keyframes_<video_id>.jsonl
  -> neighbor index (tùy chọn)
  -> caption/OCR/object/ASR ingestion
  -> segment-level metadata
  -> Retrieval text index
  -> caption/OCR/ASR/object/hybrid/temporal search
```

### Build neighbor index

Neighbor được tính trong cùng `video_id` theo cửa sổ timestamp. Output chỉ lưu
`frame_id` và `delta_seconds` của neighbor; timestamp, frame index và path đầy đủ
vẫn được resolve từ frame map/keyframe metadata chuẩn.

Chạy cho một video:

```powershell
.\.venv\Scripts\python.exe -m src.indexing.build_neighbor_index `
  --input data\metadata\keyframes_L27_V001.jsonl `
  --output data\metadata\neighbors_L27_V001.jsonl `
  --window-seconds 5
```

Chạy trên toàn bộ `keyframes_*.jsonl` trong thư mục:

```powershell
.\.venv\Scripts\python.exe -m src.indexing.build_neighbor_index `
  --input data\metadata `
  --output data\metadata\neighbors_all.jsonl `
  --window-seconds 5
```

Tool ưu tiên `timestamp` có sẵn. Nếu record chỉ có `frame_index`, FPS được lấy
từ `fps`/`video_fps` của từng record. Chỉ dùng fallback chung khi mọi video đầu
vào có cùng FPS:

```powershell
.\.venv\Scripts\python.exe -m src.indexing.build_neighbor_index `
  --input data\metadata\legacy_keyframes.jsonl `
  --output data\metadata\legacy_neighbors.jsonl `
  --window-seconds 5 `
  --fps 25
```

Neighbor luôn được sort theo timestamp, không chứa center frame và không bao giờ
trộn giữa hai video. Duplicate `(video_id, frame_id)` đồng nhất được bỏ; duplicate
xung đột sẽ báo lỗi.

### Build segment-level metadata

Chế độ mặc định `--strategy auto` dùng `segment_id`, fallback `shot_id`, cùng
`shot_start/shot_end` đã có từ keyframe extraction. Nếu metadata cũ không có
boundary, dùng fixed-duration window:

```powershell
--strategy fixed --fixed-duration-seconds 10
```

Chạy cho một video sau khi đã sinh multimodal metadata:

```powershell
.\.venv\Scripts\python.exe -m src.indexing.build_segment_metadata `
  --input data\metadata\keyframes_L27_V001.jsonl `
  --captions data\metadata\captions_L27_V001.jsonl `
  --ocr data\metadata\ocr_L27_V001.jsonl `
  --asr data\metadata\asr_L27_V001.jsonl `
  --objects data\metadata\objects_L27_V001.jsonl `
  --output data\metadata\segments_L27_V001.jsonl `
  --strategy auto
```

Chạy toàn dataset:

```powershell
.\.venv\Scripts\python.exe -m src.indexing.build_segment_metadata `
  --input data\metadata `
  --captions data\metadata `
  --ocr data\metadata `
  --asr data\metadata `
  --objects data\metadata `
  --output data\metadata\segments_all.jsonl `
  --strategy auto
```

Khi nhận thư mục, tool tự chọn `keyframes_*.jsonl`, `captions_*.jsonl`,
`ocr_*.jsonl`, `objects_*.jsonl` và `asr_*.jsonl`; `asr_segments_*` không được
đọc lại để tránh duplicate.

Mỗi segment chứa:

- `segment_id`, `video_id`, `start_time`, `end_time`;
- `start_frame`/`end_frame` khi nguồn có frame index;
- `start_keyframe`, `end_keyframe`, `keyframe_ids`;
- `captions_aggregated`, OCR, ASR và objects đã aggregate;
- `source_ids`/`source_intervals` để truy ngược frame hoặc ASR chunk nguồn.

Caption và OCR trùng được chuẩn hóa/gộp deterministic. ASR chỉ lấy chunk có
khoảng thời gian giao với segment. Object có `track_id` được đếm theo track;
nếu không có track, `occurrence_count_semantics` ghi rõ đây là số detection
occurrence.

Output JSONL được ghi compact và atomic. Chạy lại với cùng input/config cho cùng
kết quả và tool từ chối ghi đè trực tiếp bất kỳ artifact nguồn nào. Schema,
benchmark và các trade-off chi tiết nằm trong
`reports/index_size_latency.md`.

## Retrieval Phase 2-3: text, hybrid and temporal

Chỉ build text index sau khi đã sinh caption/OCR/ASR/object và build
`segments_<video_id>.jsonl` hoặc `segments_all.jsonl`. Khi nhận một thư mục,
tool ưu tiên đọc `segments_all.jsonl`, sau đó `segments_*.jsonl`, rồi mới
fallback về các artifact multimodal riêng lẻ.

```powershell
.\.venv\Scripts\python.exe -B backend\app\services\indexing\build_text_index.py `
  --metadata data\metadata `
  --output data\indexes\retrieval_text_index.json
```

Mode `hybrid` tự động fallback về visual nếu text index chưa có. Các mode
`caption`, `ocr`, `asr`, `object` sẽ báo rõ artifact Metadata đang thiếu.

### Scoring lexical và temporal

Text search giữ nguyên BM25 artifact nhưng bổ sung xử lý ở query time:

- bỏ các stopword không mô tả nội dung;
- stemming nhẹ cho động từ tiếng Anh, ví dụ `sits`/`sitting` -> `sit`;
- ưu tiên document khớp đủ từ quan trọng và đúng cụm từ;
- giảm trọng số của chủ thể chung như `person` so với hành động như `enter`;
- không hard-filter partial match, vì vậy hệ thống vẫn trả best-effort result
  khi dataset không có cảnh khớp hoàn toàn.

Temporal search vẫn dùng cùng flow `hybrid candidates -> ordered event matching`.
Pass đầu yêu cầu cùng video, timestamp tăng, frame khác nhau và gap nằm trong
`max_gap_seconds`. Nếu không có chain hợp lệ, hệ thống nới gap rồi mới dùng
fallback tương thích cho index quá thưa. Điểm chain ưu tiên event yếu nhất để
một event đúng không che mất event còn lại bị sai.

Thay đổi scoring này tương thích text index hiện có; không đổi schema artifact,
CLI hoặc thứ tự pipeline. Khi query một event không tồn tại trong dataset, kết
quả vẫn được trả nhưng cần đọc `score`, caption và `modality_scores` như một kết
quả gần đúng, không phải ground truth.

Nếu không muốn trả kết quả visual có similarity quá thấp:

```powershell
$env:RETRIEVAL_MIN_SCORE="0.10"
```

Ngưỡng production cần được chọn trên tập query có ground truth, không nên coi
`0.10` là ngưỡng mặc định cho mọi bộ dữ liệu. Cấu hình weights và index path nằm
trong `configs/retrieval.yaml`.

## QA evidence retrieval

Mode `qa`/`qa_evidence` hỗ trợ QA dạng human-in-the-loop: hệ thống không tự sinh
câu trả lời mà tìm các frame liên quan để người dùng nhìn ảnh và chọn đáp án.
Mode này dùng lại nguyên pipeline hybrid hiện có, không cần artifact hoặc model
index mới.

Ví dụ:

```powershell
.\.venv\Scripts\python.exe -B -c "from backend.app.api.search import search; import json; print(json.dumps(search('Người phụ nữ mặc áo đỏ đang ngồi trên bàn cầm cái gì?', 5, 'qa'), ensure_ascii=False, indent=2))"
```

Query planner sẽ:

```text
Người phụ nữ mặc áo đỏ đang ngồi trên bàn cầm cái gì?
  -> Người phụ nữ mặc áo đỏ đang ngồi trên bàn cầm một vật
  -> a woman wearing a red shirt sitting at a table holding an object
  -> a woman wearing a red shirt sitting on a table holding an object
```

Các query Việt/Anh được search bằng hybrid, gộp theo frame, khử trùng cùng shot
và cộng một bonus nhỏ cho frame xuất hiện trong nhiều query. Response chứa:

- `answer_target`: loại thông tin cần quan sát, ví dụ `held_object`;
- `retrieval_queries`: các query bằng chứng đã dùng;
- `answer_mode = manual_visual_inspection`;
- `results[].keyframe_path` và `thumbnail_path` để frontend hiển thị ảnh;
- `timestamp`, `video_id`, caption/OCR/ASR/object và `neighbors` để xem ngữ cảnh.

Endpoint tương ứng là `POST /retrieval/qa-evidence`. Repo chưa có FastAPI app
tổng nên hiện tại có thể test chắc chắn bằng Python wrapper ở trên. Nếu không có
frame khớp hoàn toàn, mode QA vẫn trả best-effort evidence; người dùng cần kiểm
tra ảnh thay vì coi caption hoặc score là đáp án tự động.

## Kiểm tra sau khi build

### 1. Kiểm tra artifact bắt buộc

```powershell
Get-Item `
  data\indexes\siglip2_so400m_patch16_384_flat_ip.faiss, `
  data\metadata\siglip2_so400m_patch16_384_frame_map.json, `
  data\metadata\siglip2_so400m_patch16_384_faiss_manifest.json, `
  data\indexes\retrieval_text_index.json |
  Select-Object FullName, Length, LastWriteTime
```

Nếu một đường dẫn báo `Cannot find path`, pipeline tương ứng chưa được build
xong. Ba artifact SigLIP2/FAISS đầu tiên phục vụ `visual`; text index cuối cùng
phục vụ `caption`, `ocr`, `asr`, `object` và `hybrid`.

### 2. Kiểm tra số document của từng modality trong text index

```powershell
.\.venv\Scripts\python.exe -B -c "import json; from pathlib import Path; p=json.loads(Path('data/indexes/retrieval_text_index.json').read_text(encoding='utf-8')); print(json.dumps({k:v.get('stats', {}) for k,v in p.get('modalities', {}).items()}, ensure_ascii=False, indent=2))"
```

`doc_count` lớn hơn `0` nghĩa là modality đó đã có dữ liệu để search. Nếu một
modality có `doc_count = 0`, kiểm tra lại artifact nguồn và build lại segment,
sau đó build lại text index.

### 3. Test từng search mode

Visual SigLIP2 + FAISS:

```powershell
.\.venv\Scripts\python.exe -B -c "from backend.app.api.search import search; import json; print(json.dumps(search('a person cooking', 5, 'visual'), ensure_ascii=False, indent=2))"
```

Caption:

```powershell
.\.venv\Scripts\python.exe -B -c "from backend.app.api.search import search; import json; print(json.dumps(search('a person cooking', 5, 'caption'), ensure_ascii=False, indent=2))"
```

OCR:

```powershell
.\.venv\Scripts\python.exe -B -c "from backend.app.api.search import search; import json; print(json.dumps(search('restaurant menu', 5, 'ocr'), ensure_ascii=False, indent=2))"
```

ASR:

```powershell
.\.venv\Scripts\python.exe -B -c "from backend.app.api.search import search; import json; print(json.dumps(search('how to prepare food', 5, 'asr'), ensure_ascii=False, indent=2))"
```

Object:

```powershell
.\.venv\Scripts\python.exe -B -c "from backend.app.api.search import search; import json; print(json.dumps(search('person', 5, 'object'), ensure_ascii=False, indent=2))"
```

Hybrid:

```powershell
.\.venv\Scripts\python.exe -B -c "from backend.app.api.search import search; import json; print(json.dumps(search('a person cooking', 5, 'hybrid'), ensure_ascii=False, indent=2))"
```

Temporal:

```powershell
.\.venv\Scripts\python.exe -B -c "from backend.app.api.search import search; import json; print(json.dumps(search('a person enters then sits down', 5, 'temporal'), ensure_ascii=False, indent=2))"
```

QA evidence:

```powershell
.\.venv\Scripts\python.exe -B -c "from backend.app.api.search import search; import json; print(json.dumps(search('Người phụ nữ mặc áo đỏ đang ngồi trên bàn cầm cái gì?', 5, 'qa'), ensure_ascii=False, indent=2))"
```

### 4. Đọc kết quả kiểm tra

- Không có exception và `results` là một list: mode đã load được artifact.
- `results` rỗng không nhất thiết là lỗi; query có thể không khớp dữ liệu.
- Trong kết quả `hybrid`, kiểm tra `modality_scores`. Các key như `caption`,
  `ocr`, `asr` hoặc `objects` cho biết candidate có đóng góp từ text metadata.
- Nếu chưa có `retrieval_text_index.json`, `hybrid` chỉ fallback về `visual`;
  khi đó chưa được coi là kiểm tra multimodal hoàn chỉnh.
- Với `temporal`, kiểm tra từng phần tử trong `events`. Tất cả event phải đúng
  nội dung và có timestamp tăng; điểm thấp cho biết đây có thể là best-effort
  fallback vì dataset không chứa đủ chuỗi hành động.

### 5. Chạy test hồi quy Retrieval

```powershell
.\.venv\Scripts\python.exe -B -m unittest `
  backend.tests.test_retrieval_phase2 `
  backend.tests.test_retrieval_phase3
```

## Public TKIS/VKIS competition

Pipeline dành riêng cho `data/public/` (250 video, TKIS, VKIS và xuất submission
100 đáp án/query) nằm tại [competition/README.md](competition/README.md). Adapter
này tái sử dụng toàn bộ pipeline đã triển khai: SigLIP2/FAISS, caption, OCR,
objects, ASR, segment/text index và hybrid reranking. Toàn bộ artifact sinh mới
được ghi trong `competition/`; pipeline dữ liệu mặc định ở `data/` không bị thay
đổi.
