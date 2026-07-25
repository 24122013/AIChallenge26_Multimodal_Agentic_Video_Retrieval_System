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
