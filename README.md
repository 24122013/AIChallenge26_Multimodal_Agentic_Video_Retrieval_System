# AIChallenge26 Multimodal Agentic Video Retrieval System

Repo này xây dựng baseline video retrieval cho kì thi: video -> shot-aware keyframes -> OpenCLIP embeddings -> FAISS -> search trả frame và frame lân cận cùng shot.

## Pipeline hiện tại

```text
data/raw/video/*.mp4
  -> TransNetV2 shot detection
  -> keyframe sampling + FFmpeg frame extraction + pHash dedup
  -> keyframe metadata JSONL
  -> OpenCLIP image embeddings
  -> FAISS IndexFlatIP + frame_map
  -> text query -> OpenCLIP text embedding -> FAISS top-k -> results + same-shot neighbors
```

Baseline retrieval đang dùng OpenCLIP ViT-B/16 (`laion2b_s34b_b88k`) và FAISS `IndexFlatIP` với vector đã normalize, tức inner product hoạt động như cosine similarity.

## Cài đặt

Chạy từ root repo bằng PowerShell:

```powershell
py -m venv .venv; .\.venv\Scripts\python.exe -m pip install --upgrade pip; .\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

`requirements.txt` đã gồm các thư viện chính: OpenCV, PyTorch, OpenCLIP, FAISS CPU, Pillow, TransNetV2 PyTorch, FastAPI/Uvicorn.

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
.\.venv\Scripts\python.exe -B backend\app\services\indexing\extract_keyframes.py --video-dir data\raw\video --video-glob *.mp4 --output-dir data\keyframes --shot-device auto --shot-threshold 0.5 --phash-window-sec 12
```

Output cho mỗi video:

```text
data/keyframes/<video_id>/*.jpg
data/metadata/keyframes_<video_id>.jsonl
data/metadata/keyframes_<video_id>_extract_report.json
```

Rule keyframe:

- Shot < 4s: lấy midpoint.
- Shot 4-8s: lấy 2 frame tại 1/3 và 2/3 shot.
- Shot > 8s: lấy mỗi 4s một frame.
- Extract frame bằng FFmpeg theo timestamp đã chọn.
- Dedup trong cùng video bằng pHash trong cửa sổ thời gian gần, mặc định 12s.
- Metadata lưu `timestamp`, `frame_index`, `shot_start`, `shot_end`, `shot_id`, `source_video_path`.

Nếu muốn bật CLIP dedup gần nhau:

```powershell
.\.venv\Scripts\python.exe -B backend\app\services\indexing\extract_keyframes.py --video-dir data\raw\video --enable-clip-dedup --phash-window-sec 12 --clip-similarity-threshold 0.985 --clip-window-sec 12
```

## Bước 2: Encode keyframes bằng OpenCLIP

Chạy cho toàn bộ metadata keyframe:

```powershell
Get-ChildItem data\metadata\keyframes_*.jsonl | ForEach-Object { $videoId = $_.BaseName -replace '^keyframes_', ''; .\.venv\Scripts\python.exe -B backend\app\services\indexing\build_openclip_index.py --metadata-path $_.FullName --embeddings-path "data\embeddings\openclip_vit_b16_$videoId.npy" --embedding-metadata-path "data\metadata\openclip_vit_b16_embeddings_$videoId.jsonl" --skipped-path "data\metadata\openclip_vit_b16_skipped_$videoId.jsonl" --benchmark-path "data\metadata\openclip_vit_b16_benchmark_$videoId.json" --batch-size 32 --device auto }
```

Output chính:

```text
data/embeddings/openclip_vit_b16_<video_id>.npy
data/metadata/openclip_vit_b16_embeddings_<video_id>.jsonl
```

## Bước 3: Build FAISS index

```powershell
.\.venv\Scripts\python.exe -B backend\app\services\indexing\build_faiss_index.py --embeddings-glob "data/embeddings/openclip_vit_b16_*.npy" --embedding-metadata-template "data/metadata/openclip_vit_b16_embeddings_{video_id}.jsonl" --index-path data\indexes\openclip_vit_b16_flat_ip.faiss --frame-map-path data\metadata\openclip_vit_b16_frame_map.json --manifest-path data\metadata\openclip_vit_b16_faiss_manifest.json --report-path data\metadata\openclip_vit_b16_index_report.json
```

Output retrieval cần có:

```text
data/indexes/openclip_vit_b16_flat_ip.faiss
data/metadata/openclip_vit_b16_frame_map.json
```

## Bước 4: Test retrieval bằng Python

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
.\.venv\Scripts\python.exe -B -m unittest backend.tests.test_search_visual; .\.venv\Scripts\python.exe -B backend\app\services\indexing\extract_keyframes.py --help
```

## Cấu trúc quan trọng

```text
backend/app/services/indexing/extract_keyframes.py       # TransNetV2 + keyframe sampling + dedup
backend/app/services/indexing/build_openclip_index.py    # encode ảnh keyframe thành OpenCLIP embeddings
backend/app/services/indexing/build_faiss_index.py       # gom embeddings thành FAISS + frame_map
backend/app/services/retrieval/search_visual.py          # text query -> OpenCLIP -> FAISS -> results
backend/app/services/metadata/metadata_store.py          # lookup frame_map và same-shot neighbors
docs/keyframe_extraction.md                              # giải thích chi tiết chiến lược keyframe
```

## Ghi chú cho team

- Dùng `.\.venv\Scripts\python.exe`, không dùng `python` global nếu máy có nhiều Python.
- `data/metadata/openclip_vit_b16_frame_map.json` phải khớp với FAISS index.
- Sau khi extract lại keyframes thì cần encode lại embeddings và build lại FAISS.
- Extractor chỉ dùng TransNetV2. Nếu thiếu FFmpeg hoặc TransNetV2 lỗi, sửa môi trường rồi chạy lại.
