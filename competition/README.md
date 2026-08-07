# Pipeline đầy đủ cho cuộc thi TKIS/VKIS

`competition/` là adapter của toàn bộ phần đã được triển khai trong hệ thống gốc cho
bộ dữ liệu `data/public/`. Adapter không viết lại thuật toán lõi mà gọi trực tiếp các
service hiện có:

- TransNetV2 + pHash để trích keyframe;
- SigLIP2 để tạo image/text embedding;
- FAISS `IndexFlatIP` cho visual retrieval;
- BLIP caption, EasyOCR, YOLO objects và Whisper ASR;
- temporal-neighbor index và segment-level metadata;
- BM25 text index cho caption/OCR/ASR/object;
- `HybridSearchEngine`, `HybridReranker` và weights trong `configs/retrieval.yaml`;
- image-to-image SigLIP2 và local frame refinement cho VKIS.

Không có bước nào tự chạy. Đứng tại thư mục gốc repo và tự chạy lần lượt các lệnh
dưới đây.

## Luồng đầy đủ

```text
videos
  -> keyframes + frame metadata
  -> SigLIP2 embeddings -> FAISS + frame map
  -> caption + OCR + objects + ASR
  -> neighbor metadata
  -> multimodal segments -> BM25 text index
  -> TKIS: visual + caption + OCR + ASR + objects -> hybrid rerank
  -> VKIS: image FAISS -> frame-by-frame refinement
  -> submission.csv
```

## Cấu trúc artifact

```text
competition/
├── keyframes/     # ảnh keyframe theo video
├── metadata/      # keyframe, multimodal, segment, frame map, manifest, report
├── embeddings/    # vector SigLIP2 theo video
├── indexes/       # visual FAISS và retrieval_text_index.json
└── results/       # submission.csv cuối cùng
```

Artifact sinh ra đã được `.gitignore`; `.gitkeep` chỉ giữ cấu trúc thư mục.

Các module `backend/app/services/agent/` hiện là stub rỗng nên không có planner,
query expansion hay tool execution khả dụng để gọi lại. Adapter không tự phát minh
logic thay thế. QA/evaluation không tham gia submission vì public set không cung cấp
ground truth và output cuộc thi chỉ nhận `video,frame_idx`.

## 0. Chuẩn bị và kiểm tra input

Input phải có cấu trúc:

```text
data/public/
├── corpus.csv
├── questions.csv
├── sample_submission.csv
├── videos/*.mp4
└── vkis/frames/*.jpg
```

Cài dependency và kiểm tra cả `ffmpeg` lẫn `ffprobe`:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
ffmpeg -version
ffprobe -version
```

Kiểm tra CSV, 250 video và 50 ảnh VKIS mà không xử lý video:

```powershell
.\.venv\Scripts\python.exe -m competition.pipeline validate-input
```

Public set hợp lệ phải có 250 video, 100 query, gồm 50 TKIS và 50 VKIS.

## 1. Trích keyframe

```powershell
.\.venv\Scripts\python.exe -m competition.pipeline extract --device auto
```

Bước này gọi `extract_keyframes_for_video` của repo. Ảnh nằm trong
`competition/keyframes/<video_id>/`; metadata chứa `frame_index` 0-based,
`shot_start` và `shot_end` nằm trong `competition/metadata/`.

Nếu bị gián đoạn:

```powershell
.\.venv\Scripts\python.exe -m competition.pipeline extract --device auto --resume
```

## 2. Tạo embedding SigLIP2

GPU:

```powershell
.\.venv\Scripts\python.exe -m competition.pipeline embed --device cuda --batch-size auto
```

CPU:

```powershell
.\.venv\Scripts\python.exe -m competition.pipeline embed --device cpu --batch-size 4
```

Model được nạp một lần và dùng lại cho 250 video. Có thể thêm `--resume` khi chạy
lại. Model/revision/vector dimension được ghi vào metadata để kiểm tra contract.

## 3. Tạo visual FAISS index

```powershell
.\.venv\Scripts\python.exe -m competition.pipeline index
```

Lệnh chỉ nhận embedding của đúng video trong `corpus.csv`, rồi gọi
`build_faiss_artifacts` để tạo FAISS, frame map và encoder manifest.

## 4. Sinh toàn bộ multimodal metadata

Có thể chạy cả bốn pipeline trong một lệnh:

```powershell
.\.venv\Scripts\python.exe -m competition.pipeline enrich --device cuda
```

Khuyến nghị chạy tách từng model để dễ theo dõi VRAM và tiếp tục khi gián đoạn:

```powershell
.\.venv\Scripts\python.exe -m competition.pipeline enrich --modalities caption --device cuda
.\.venv\Scripts\python.exe -m competition.pipeline enrich --modalities ocr --device cuda
.\.venv\Scripts\python.exe -m competition.pipeline enrich --modalities objects --device cuda
.\.venv\Scripts\python.exe -m competition.pipeline enrich --modalities asr --device cuda
```

Các service gốc tự bỏ qua record/video đã có output, nên bốn lệnh trên có thể chạy
lại an toàn. Chỉ dùng `--overwrite` khi muốn tạo lại artifact của modality đó.

Output cho mỗi video:

```text
competition/metadata/captions_<video_id>.jsonl
competition/metadata/ocr_<video_id>.jsonl
competition/metadata/objects_<video_id>.jsonl
competition/metadata/asr_<video_id>.jsonl
competition/metadata/asr_segments_<video_id>.jsonl
competition/metadata/<modality>_<video_id>_report.json
```

Mặc định tương ứng hệ thống gốc:

- caption: `Salesforce/blip-image-captioning-base`, batch 4, có segment caption;
- OCR: EasyOCR `vi` + `en`, threshold 0.3, batch 4;
- objects: `yolo11n.pt`, confidence 0.25, IoU 0.7, batch 8;
- ASR: faster-whisper/Whisper `small`, auto language và VAD.

Video không có audio được ASR ghi `skipped/no_audio_stream`, không phải lỗi.

## 5. Tạo temporal-neighbor metadata

```powershell
.\.venv\Scripts\python.exe -m competition.pipeline neighbors --window-seconds 5
```

Output là `competition/metadata/neighbors_all.jsonl`. Đây là artifact tùy chọn của
hệ thống gốc để tra ngữ cảnh trước/sau; visual search hiện cũng lấy same-shot
neighbors trực tiếp từ frame map.

## 6. Aggregate multimodal segment

Chỉ chạy sau khi đủ caption, OCR, objects và ASR của 250 video:

```powershell
.\.venv\Scripts\python.exe -m competition.pipeline segments --strategy auto
```

Lệnh gọi `build_segment_metadata`, dùng boundary `segment_id`/`shot_id`, rồi gộp
caption, OCR, ASR và objects kèm provenance vào
`competition/metadata/segments_all.jsonl`.

## 7. Tạo BM25 text index

```powershell
.\.venv\Scripts\python.exe -m competition.pipeline text-index
```

Output `competition/indexes/retrieval_text_index.json` chứa bốn modality:
`caption`, `ocr`, `asr`, `objects`. Không có file này thì `predict` sẽ dừng thay vì
âm thầm fallback về visual-only; điều này bảo đảm submission thực sự dùng hybrid.

## 8. Chạy TKIS/VKIS và tạo submission

GPU:

```powershell
.\.venv\Scripts\python.exe -m competition.pipeline predict --device cuda
```

CPU:

```powershell
.\.venv\Scripts\python.exe -m competition.pipeline predict --device cpu --batch-size 4
```

Kết quả nằm tại `competition/results/submission.csv`.

### TKIS

TKIS không còn là visual-only. Mỗi query chạy đồng thời qua:

1. SigLIP2 text-to-keyframe visual search;
2. BM25 caption search;
3. BM25 OCR search;
4. BM25 ASR search;
5. BM25 object-label search;
6. candidate merge và `HybridReranker`.

Pipeline đọc weights, same-shot dedupe và các giới hạn retrieval từ
`configs/retrieval.yaml`. Có thể truyền config khác bằng `--retrieval-config`, nhưng
không cần nếu muốn giống hệ thống gốc.

Mặc định `--tkis-routing auto-temporal`: query có cấu trúc thứ tự như `then`,
`after that`, `followed by` sẽ dùng `HybridSearchEngine.temporal_search`; mỗi event
vẫn được truy xuất bằng full hybrid trước khi ghép theo thứ tự thời gian. Query đơn
event dùng `HybridSearchEngine.search`. Dùng `--tkis-routing hybrid` nếu muốn tắt
nhận diện temporal tự động.

### VKIS

VKIS vẫn dùng nhánh phù hợp với truy vấn ảnh:

1. encode 50 query image theo batch bằng cùng model/revision SigLIP2 của corpus;
2. tìm keyframe gần nhất bằng cùng FAISS index;
3. với 20 ứng viên đầu, so ảnh query với từng frame trong ±75 frame quanh keyframe
   và trong biên shot bằng hàm `mse` sẵn có;
4. trả chỉ số frame 0-based khớp nhất.

Bước refine giúp vượt qua việc keyframe đúng video/shot nhưng cách frame gốc quá
dung sai VKIS ±12. Có thể tăng độ phủ:

```powershell
.\.venv\Scripts\python.exe -m competition.pipeline predict `
  --device cuda `
  --search-depth 300 `
  --vkis-refine-top-k 30 `
  --vkis-refine-radius-frames 100
```

`--search-depth` phải từ 100 trở lên. Tăng các giá trị VKIS làm chậm hơn nhưng không
cần build lại artifact.

## 9. Kiểm tra submission

```powershell
.\.venv\Scripts\python.exe -m competition.pipeline validate-submission
```

Validator kiểm tra header, đúng thứ tự 100 query, đủ 100 answer/query, tên video,
giới hạn frame và số câu trả lời trùng chính xác. Report phải có `status: passed`;
lý tưởng là `exact_duplicate_answers: 0`.

## Lệnh đầy đủ theo thứ tự

```powershell
.\.venv\Scripts\python.exe -m competition.pipeline validate-input
.\.venv\Scripts\python.exe -m competition.pipeline extract --device auto
.\.venv\Scripts\python.exe -m competition.pipeline embed --device cuda --batch-size auto
.\.venv\Scripts\python.exe -m competition.pipeline index
.\.venv\Scripts\python.exe -m competition.pipeline enrich --modalities caption --device cuda
.\.venv\Scripts\python.exe -m competition.pipeline enrich --modalities ocr --device cuda
.\.venv\Scripts\python.exe -m competition.pipeline enrich --modalities objects --device cuda
.\.venv\Scripts\python.exe -m competition.pipeline enrich --modalities asr --device cuda
.\.venv\Scripts\python.exe -m competition.pipeline neighbors
.\.venv\Scripts\python.exe -m competition.pipeline segments
.\.venv\Scripts\python.exe -m competition.pipeline text-index
.\.venv\Scripts\python.exe -m competition.pipeline predict --device cuda
.\.venv\Scripts\python.exe -m competition.pipeline validate-submission
```

## Đường dẫn tùy chỉnh

Truyền cùng `--public-root` và `--output-root` cho mọi bước. `predict` mặc định ghi
vào `<output-root>/results/submission.csv`; có thể đặt `--submission-path` khác:

```powershell
.\.venv\Scripts\python.exe -m competition.pipeline predict `
  --public-root D:\dataset\public `
  --output-root D:\artifacts\competition `
  --submission-path D:\artifacts\competition\results\submission.csv `
  --device cuda
```
