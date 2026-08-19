# AIChallenge26 Multimodal Agentic Video Retrieval System

Runbook này mô tả đường chạy canonical của repo trên Windows/PowerShell, từ lúc
tạo môi trường đến khi build offline corpus và query online. Các lệnh bên dưới
được viết để chạy tại thư mục gốc của repo.

Hệ thống hỗ trợ sáu task online:

- `kis`: tìm một khoảnh khắc/đối tượng/sự kiện khá cụ thể.
- `avs`: tìm nhiều video phù hợp với mô tả ngữ nghĩa rộng.
- `temporal`: truy hồi chuỗi sự kiện có thứ tự.
- `trake`: parse nhiều event, căn chỉnh thành sequence cùng video và refine về
  original frame.
- `qa`: truy hồi evidence và có thể sinh grounded answer bằng Qwen.
- `auto`: tự route sang KIS, AVS, temporal hoặc QA. `auto` **không** tự route
  sang TRAKE.

> Phạm vi dữ liệu hiện tại là visual: keyframe, caption, OCR và object labels.
> Repo chưa có ASR, audio retrieval hay transcript index.

## 1. Luồng chạy ngắn nhất

```text
video gốc
  -> shot detection
  -> dense frame candidates
  -> SigLIP2 + caption + OCR + objects cho toàn bộ candidate
  -> chọn canonical keyframes
  -> publish selected FAISS + dense FAISS + BM25 + context + BGE-M3
  -> query qua OnlinePipeline
```

Nếu máy đã có môi trường và model cache, E2E tối thiểu là:

```powershell
# 1) Build toàn bộ corpus từ data/raw/video/*.mp4
.\.venv\Scripts\python.exe -B -m backend.app.pipelines.offline_pipeline `
  --video-dir data\raw\video `
  --video-glob *.mp4 `
  --output-dir data `
  --device auto `
  --build-corpus `
  --resume `
  --verbose

# 2) Query KIS
.\.venv\Scripts\python.exe -B -m backend.app.pipelines.online_pipeline `
  --task kis `
  --query "một người mặc áo đỏ đang mở cửa ô tô" `
  --top-k 20 `
  --debug
```

Đừng skip phần kiểm tra artifact ở dưới. File tồn tại nhưng sai checksum,
dimension, row order hoặc corpus generation vẫn là artifact hỏng.

## 2. Yêu cầu hệ thống

- Windows PowerShell.
- Python 3.11 hoặc 3.12; ví dụ dưới đây dùng Python 3.12.
- FFmpeg và ffprobe có trong `PATH`.
- Kết nối mạng cho lần tải model đầu tiên, hoặc model cache đã chuẩn bị sẵn.
- Dung lượng đĩa đủ cho video gốc, dense JPEG, feature, index và model cache.

Kiểm tra trước khi cài:

```powershell
py -0p
ffmpeg -version
ffprobe -version
nvidia-smi    # Chỉ cần khi định chạy CUDA
```

Repo không áp đặt một con số RAM/VRAM “chuẩn”. Full corpus và Qwen 9B có thể
rất nặng; cần đo thật trên máy đích, không nên đoán từ việc unit test pass.

## 3. Tạo môi trường Python

PyTorch và PaddlePaddle không nằm trong `requirements.txt` để pip không tự thay
wheel CPU/CUDA. Chọn đúng **một** profile bên dưới; không cài đồng thời
`paddlepaddle` và `paddlepaddle-gpu`.

### 3.1. Tạo virtual environment

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
```

Không bắt buộc activate venv vì mọi lệnh trong README gọi trực tiếp
`.venv\Scripts\python.exe`. Nếu vẫn muốn activate:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\.venv\Scripts\Activate.ps1
```

### 3.2. Profile CPU

```powershell
.\.venv\Scripts\python.exe -m pip install `
  torch==2.7.1 torchvision==0.22.1 `
  --index-url https://download.pytorch.org/whl/cpu

.\.venv\Scripts\python.exe -m pip install paddlepaddle==3.2.0
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 3.3. Profile NVIDIA CUDA 11.8

```powershell
.\.venv\Scripts\python.exe -m pip install `
  torch==2.7.1 torchvision==0.22.1 `
  --index-url https://download.pytorch.org/whl/cu118

.\.venv\Scripts\python.exe -m pip install `
  paddlepaddle-gpu==3.2.0 `
  --index-url https://www.paddlepaddle.org.cn/packages/stable/cu118/

.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 3.4. Profile NVIDIA CUDA 12.6

```powershell
.\.venv\Scripts\python.exe -m pip install `
  torch==2.7.1 torchvision==0.22.1 `
  --index-url https://download.pytorch.org/whl/cu126

.\.venv\Scripts\python.exe -m pip install `
  paddlepaddle-gpu==3.2.0 `
  --index-url https://www.paddlepaddle.org.cn/packages/stable/cu126/

.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 3.5. Kiểm tra môi trường

```powershell
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -c "import torch, paddle, transformers, faiss; print({'torch': torch.__version__, 'torch_cuda': torch.cuda.is_available(), 'paddle': paddle.__version__, 'paddle_cuda': paddle.is_compiled_with_cuda(), 'transformers': transformers.__version__})"
```

Nếu `torch_cuda` hoặc `paddle_cuda` là `False` trong profile GPU, dừng tại đây
và sửa wheel/driver trước. Đừng chạy full offline rồi mới phát hiện model đang
fallback sang CPU.

## 4. Cấu hình runtime

Tạo `.env` local từ file mẫu. `.env` đã được Git ignore, nhưng vẫn không nên bỏ
token hay secret vào log/report.

```powershell
Copy-Item .env.example .env
notepad .env
```

Các giá trị nên kiểm tra ngay:

```dotenv
RETRIEVAL_CONFIG_PATH=configs/retrieval.yaml
RETRIEVAL_DEVICE=auto
RETRIEVAL_QUERY_EXPANSION_ENABLED=true
RETRIEVAL_TRAKE_VIDEO_ROOT=data/raw/video
QA_ANSWER_MODE=off
```

- `RETRIEVAL_DEVICE=auto|cpu|cuda` áp dụng cho online retrieval.
- `RETRIEVAL_QUERY_EXPANSION_ENABLED=true` bật Qwen query expansion cho KIS/AVS.
  Auto-query được resolve là temporal/QA sẽ không gọi generic expansion.
- `QA_ANSWER_MODE=off|optional|required` là Qwen answerer riêng của QA; nó không
  phải query expansion.
- Biến đã đặt trong terminal như `$env:RETRIEVAL_DEVICE="cpu"` được ưu tiên hơn
  cùng biến trong `.env`.

Weights và giới hạn retrieval nằm trong
[`configs/retrieval.yaml`](configs/retrieval.yaml). Các path artifact/model và
feature flag runtime nằm trong [`.env.example`](.env.example).

Lần chạy model đầu có thể tải nhiều GB và mất khá lâu. Chỉ dùng
`--local-files-only` hoặc `QA_MODELS_LOCAL_ONLY=true` sau khi cache đã đầy đủ.

## 5. Chuẩn bị dữ liệu

Tạo các thư mục runtime và chép video vào `data/raw/video`. Tên file không được
trùng nhau nếu bỏ qua hoa/thường vì stem của file chính là `video_id`.

```powershell
$runtimeDirs = @(
  "data\raw\video",
  "data\candidates",
  "data\dense_keyframes",
  "data\candidate_features",
  "data\keyframes",
  "data\metadata",
  "data\embeddings",
  "data\indexes",
  "data\model_cache",
  "data\cache",
  "data\reports",
  "data\submissions"
)
$runtimeDirs | ForEach-Object {
  New-Item -ItemType Directory -Force $_ | Out-Null
}

Copy-Item "C:\duong-dan-dataset\*.mp4" data\raw\video\
Get-ChildItem data\raw\video\*.mp4 | Select-Object Name, Length
```

Mặc định full run chỉ discover `*.mp4`. Nếu dataset dùng extension khác, đổi
`--video-glob`; quick mode vẫn kiểm tra suffix được pipeline hỗ trợ.

## 6. Chạy offline pipeline

Entrypoint canonical là `backend.app.pipelines.offline_pipeline`. Nó xử lý xong
mọi stage của video A rồi mới sang video B; sau cùng mới build/publish corpus
index một lần. Repo không có một bộ lệnh stage rời nào tương đương contract này.

Xem toàn bộ option thật của checkout hiện tại:

```powershell
.\.venv\Scripts\python.exe -B -m backend.app.pipelines.offline_pipeline --help
```

### 6.1. Smoke một video trước

Thay `L01_V001.mp4` bằng file thật:

```powershell
.\.venv\Scripts\python.exe -B -m backend.app.pipelines.offline_pipeline `
  --video-path data\raw\video\L01_V001.mp4 `
  --output-dir data `
  --device auto `
  --skip-corpus `
  --resume `
  --verbose
```

`--skip-corpus` là điểm cực quan trọng. Quick mode chỉ xử lý một video; nếu cố
thêm `--build-corpus`, global index sẽ bị thay bằng corpus chỉ chứa đúng video
được request. Chỉ làm vậy khi đó thực sự là mục tiêu.

Có thể chọn theo stem thay vì path:

```powershell
.\.venv\Scripts\python.exe -B -m backend.app.pipelines.offline_pipeline `
  --video-dir data\raw\video `
  --video-id L01_V001 `
  --output-dir data `
  --skip-corpus `
  --resume `
  --verbose
```

### 6.2. Các stage offline chạy bên trong command

Giả sử `$videoId = "L01_V001"`:

```powershell
$videoId = "L01_V001"
```

| Stage | Pipeline làm gì | Artifact/checkpoint chính | Lệnh kiểm tra nhanh |
|---|---|---|---|
| 1. Shot detection | Đọc FPS/frame count và tìm shot boundary bằng TransNetV2 | `data/reports/offline/<video_id>/shots.json` | `Get-Content "data\reports\offline\$videoId\shots.json" -Raw \| ConvertFrom-Json \| Select-Object status, shot_count, detector_name` |
| 2. Dense candidate generation | Sample theo thời gian, thêm shot anchor/boundary guard | `candidate_plan.jsonl`, `candidate_report.json` | `(Get-Content "data\reports\offline\$videoId\candidate_plan.jsonl" \| Measure-Object -Line).Lines` |
| 3. Materialization | Decode toàn bộ candidate thành JPEG và giữ identity/order | `data/candidates/<video_id>.jsonl`, `data/dense_keyframes/<video_id>/` | `(Get-Content "data\candidates\$videoId.jsonl" \| Measure-Object -Line).Lines` |
| 4. Dense features | Chạy SigLIP2, Florence-2 caption, PP-OCRv5 và YOLOE trên full candidate pool | `data/candidate_features/<video_id>/` | `Get-ChildItem "data\candidate_features\$videoId"` |
| 5. Multimodal selection | Protected-event, gap repair, dedup và MMR để chọn keyframe | `selection_report.json`, `candidate_ledger.jsonl` | `Get-Content "data\reports\offline\$videoId\selection_report.json" -Raw \| ConvertFrom-Json \| Select-Object status, selected_count` |
| 6. Canonical persistence | Ghi selected JPEG, metadata, caption/OCR/object và embedding | `data/keyframes/<video_id>/`, `data/metadata/keyframes_<video_id>.jsonl` | `(Get-Content "data\metadata\keyframes_$videoId.jsonl" \| Measure-Object -Line).Lines` |
| 7. Validation/commit | Kiểm identity, count, checksum và chỉ ghi commit marker cuối cùng khi pass | `data/metadata/keyframes_<video_id>_extract_report.json` | `Get-Content "data\metadata\keyframes_${videoId}_extract_report.json" -Raw \| ConvertFrom-Json \| Select-Object status, dense_candidate_count, selected_count` |

File có mặt không tự động đồng nghĩa stage hợp lệ. Khi chạy lại với `--resume`,
pipeline validate contract/checksum rồi mới log `[SKIP]`; checkpoint sai sẽ bị
chạy lại.

### 6.3. Build full dataset và publish corpus

Sau khi smoke một video ổn, chạy toàn bộ thư mục:

```powershell
.\.venv\Scripts\python.exe -B -m backend.app.pipelines.offline_pipeline `
  --video-dir data\raw\video `
  --video-glob *.mp4 `
  --output-dir data `
  --device auto `
  --build-corpus `
  --resume `
  --verbose
```

`--build-corpus` thực hiện và publish atomically:

1. Selected-keyframe SigLIP2 FAISS cho coarse retrieval.
2. Full dense-candidate SigLIP2 FAISS cho global rescue/CSES.
3. BM25 index cho caption/OCR/object text.
4. `neighbors_all.jsonl` và `segments_all.jsonl` từ canonical selected frames.
5. BGE-M3 dense text index, trừ khi truyền `--skip-bge`.
6. `offline_corpus_manifest.json` chứa checksum và `bundle_generation` của cả
   bundle.

Corpus mặc định fail closed: nếu một video lỗi thì global index không được build
từ tập thiếu. `--allow-partial-corpus` chỉ nên dùng cho dev/debug vì nó chủ động
loại video lỗi khỏi corpus. `--skip-bge` cũng chỉ nên dùng khi chấp nhận QA/TRAKE
BGE online không có artifact để bật.

### 6.4. Profile ít VRAM hơn

Nếu OOM, giảm batch trước khi đổi model/quantization:

```powershell
.\.venv\Scripts\python.exe -B -m backend.app.pipelines.offline_pipeline `
  --video-dir data\raw\video `
  --video-glob *.mp4 `
  --output-dir data `
  --device cuda `
  --batch-size 4 `
  --caption-batch-size 1 `
  --ocr-batch-size 2 `
  --object-batch-size 4 `
  --bge-batch-size 4 `
  --build-corpus `
  --resume `
  --verbose
```

Caption backend canonical hiện chỉ hỗ trợ `--caption-quantization none`. Đừng
truyền `4bit`/`8bit` chỉ vì CLI liệt kê choice; backend sẽ từ chối rõ ràng.

### 6.5. Resume và rebuild

```powershell
# Resume là mặc định; ghi rõ để command tự mô tả
.\.venv\Scripts\python.exe -B -m backend.app.pipelines.offline_pipeline `
  --video-dir data\raw\video --output-dir data --build-corpus --resume

# Recompute mọi stage và corpus dù checkpoint đang hợp lệ
.\.venv\Scripts\python.exe -B -m backend.app.pipelines.offline_pipeline `
  --video-dir data\raw\video --output-dir data --build-corpus --force
```

`--force` tốn tài nguyên lớn. Không cần xóa tay artifact trước; xóa lẻ dễ tạo
bundle nửa cũ nửa mới và làm lineage khó audit.

### 6.6. Xác nhận offline đã hoàn tất

```powershell
$requiredArtifacts = @(
  "data\indexes\siglip2_so400m_patch16_384_flat_ip.faiss",
  "data\metadata\siglip2_so400m_patch16_384_faiss_metadata.jsonl",
  "data\metadata\siglip2_so400m_patch16_384_frame_map.json",
  "data\metadata\siglip2_so400m_patch16_384_faiss_manifest.json",
  "data\metadata\siglip2_so400m_patch16_384_index_report.json",
  "data\indexes\siglip2_so400m_patch16_384_dense_flat_ip.faiss",
  "data\metadata\siglip2_so400m_patch16_384_dense_faiss_metadata.jsonl",
  "data\metadata\siglip2_so400m_patch16_384_dense_frame_map.json",
  "data\metadata\siglip2_so400m_patch16_384_dense_faiss_manifest.json",
  "data\metadata\siglip2_so400m_patch16_384_dense_index_report.json",
  "data\indexes\retrieval_text_index.json",
  "data\metadata\neighbors_all.jsonl",
  "data\metadata\segments_all.jsonl",
  "data\metadata\offline_corpus_manifest.json",
  "data\reports\offline\corpus_report.json"
)
$requiredArtifacts | ForEach-Object {
  [pscustomobject]@{ Exists = Test-Path $_; Path = $_ }
}

$manifest = Get-Content data\metadata\offline_corpus_manifest.json -Raw | ConvertFrom-Json
$manifest | Select-Object status, bundle_generation, bge_enabled, video_ids

$report = Get-Content data\reports\offline\corpus_report.json -Raw | ConvertFrom-Json
$report | Select-Object status, video_count, dense_candidate_count, selected_keyframe_count
```

Nếu không dùng `--skip-bge`, kiểm tra thêm:

```powershell
Get-ChildItem data\indexes\bge_m3\
Test-Path data\indexes\bge_m3\bge_m3_flat_ip.faiss
Test-Path data\indexes\bge_m3\bge_m3_frame_map.json
Test-Path data\indexes\bge_m3\bge_m3_manifest.json
```

Tín hiệu pass tối thiểu là mọi artifact bắt buộc tồn tại, `manifest.status` và
`report.status` đều là `passed`, và `video_ids` đúng dataset mong đợi. Online
loader sẽ kiểm lại checksum/generation; lệnh `Test-Path` chỉ là preflight, không
thay thế validation của loader.

## 7. Chạy online pipeline

Entrypoint query canonical:

```powershell
.\.venv\Scripts\python.exe -B -m backend.app.pipelines.online_pipeline --help
```

Mọi query đều in JSON ra stdout. Thêm `--output` để lưu cùng JSON vào file.

### 7.1. Bật Qwen query expansion

`configs/retrieval.yaml` và `.env.example` hiện bật expansion cho KIS/AVS. Để
override trong terminal:

```powershell
$env:RETRIEVAL_QUERY_EXPANSION_ENABLED = "true"
```

Qwen expansion chạy với `kis`, `avs`, hoặc `auto` sau khi auto resolve thành
KIS/AVS. Nó bị skip có chủ đích ở `temporal`, `qa` và `trake` vì các route này có
parser/decomposition riêng. Nếu model/provider unavailable, xem
`routing_trace.query_expansion` và `query_plan.expansion_plan`; đừng nhìn mỗi
result rồi kết luận Qwen đã chạy.

### 7.2. Bật context cho KIS/AVS

Context mặc định tắt để artifact thiếu không block runtime. Muốn context thực sự
được load và tham gia bounded rerank:

```powershell
$env:ONLINE_NEIGHBOR_CONTEXT_ENABLED = "true"
$env:ONLINE_SEGMENT_CONTEXT_ENABLED = "true"
```

Sau đó thêm `--with-context` vào query. Chỉ thêm `--with-context` mà không bật
hai biến trên sẽ tạo request context nhưng artifact index chưa được load; response
sẽ báo fallback thay vì giả vờ đã score.

### 7.3. Query `auto`

Phù hợp khi caller chưa biết query là KIS, AVS, temporal hay QA:

```powershell
.\.venv\Scripts\python.exe -B -m backend.app.pipelines.online_pipeline `
  --task auto `
  --query "sau khi bước vào phòng, người đàn ông ngồi xuống" `
  --top-k 20 `
  --debug `
  --output data\reports\online_auto.json
```

Kiểm `requested_task`, `task` và `query_plan.profile_source` để biết route thật.
Nếu cần TRAKE, phải truyền `--task trake` explicit.

### 7.4. Query KIS

KIS dùng cho một khoảnh khắc/instance khá cụ thể:

```powershell
.\.venv\Scripts\python.exe -B -m backend.app.pipelines.online_pipeline `
  --task kis `
  --query "người đàn ông mặc áo xanh lấy chai nước từ tủ lạnh" `
  --top-k 20 `
  --with-context `
  --debug `
  --output data\reports\online_kis.json
```

Đường canonical KIS/AVS là selected multimodal retrieval + weighted RRF → dense
global rescue → CSES → bounded context score → deterministic rerank/dedup/Top-K.
Trong output, kiểm:

- `routing_trace.coarse_to_dense.executed=true` và `mode="coarse_to_dense"` để
  chứng minh dense path thật sự chạy.
- `mode="selected_only_fallback"` nghĩa là chỉ sparse/selected path chạy.
- `routing_trace.context_scoring` để biết neighbor/segment được request, available
  và executed hay chưa.
- `routing_trace.query_expansion.provider_call_count` để biết Qwen provider có
  được gọi.

### 7.5. Query AVS

AVS dùng cho mô tả rộng, có thể khớp nhiều video:

```powershell
.\.venv\Scripts\python.exe -B -m backend.app.pipelines.online_pipeline `
  --task avs `
  --query "các cảnh ngoài trời có nhiều người đang chơi thể thao" `
  --top-k 50 `
  --with-context `
  --debug `
  --output data\reports\online_avs.json
```

AVS và KIS dùng cùng coarse-to-dense engine nhưng query plan/profile khác nhau;
đừng coi `avs` chỉ là alias đổi tên của `kis`.

### 7.6. Query temporal

Temporal giữ thứ tự event và dùng evidence engine riêng:

```powershell
.\.venv\Scripts\python.exe -B -m backend.app.pipelines.online_pipeline `
  --task temporal `
  --query "một người mở cửa, sau đó bước vào phòng rồi ngồi xuống" `
  --top-k 20 `
  --debug `
  --output data\reports\online_temporal.json
```

Generic Qwen expansion không chạy ở route này để tránh đổi thứ tự/ý nghĩa event.

### 7.7. Query TRAKE

TRAKE phù hợp khi cần đúng một sequence cùng video, mỗi event ánh xạ tới một
original frame. Cấu trúc `Context` + danh sách `Events` giúp deterministic parser
ít mơ hồ hơn:

```powershell
$env:RETRIEVAL_TRAKE_VIDEO_ROOT = "data\raw\video"

.\.venv\Scripts\python.exe -B -m backend.app.pipelines.online_pipeline `
  --task trake `
  --query "Context: a high jump. Events: 1. the athlete first leaves the ground 2. the athlete reaches peak height 3. the athlete lands" `
  --top-k 100 `
  --debug `
  --output data\reports\online_trake.json
```

TRAKE không gọi generic Qwen expansion. Pipeline parse event, retrieve theo từng
event, gate video coverage, align K-best path, rồi decode một cửa sổ bounded quanh
coarse frame và reuse SigLIP2 để refine. Nếu raw video không resolve được dưới
`RETRIEVAL_TRAKE_VIDEO_ROOT`, response có thể giữ coarse fallback và ghi warning.

Muốn bật BGE riêng cho TRAKE:

```powershell
$env:RETRIEVAL_TRAKE_BGE_DENSE_ENABLED = "true"
$env:RETRIEVAL_TRAKE_BGE_RERANKER_ENABLED = "true"
$env:RETRIEVAL_TRAKE_BGE_INDEX_ROOT = "data\indexes\bge_m3"
```

Các flag `RETRIEVAL_TRAKE_BGE_*` độc lập với `QA_BGE_*`.

### 7.8. Query QA chỉ lấy evidence

Đây là mode nhẹ và dễ debug nhất vì chưa load Qwen answer model:

```powershell
$env:QA_ANSWER_MODE = "off"

.\.venv\Scripts\python.exe -B -m backend.app.pipelines.online_pipeline `
  --task qa `
  --query "Người phụ nữ mặc áo đỏ đang cầm vật gì?" `
  --top-k 5 `
  --debug `
  --output data\reports\online_qa_evidence.json
```

### 7.9. Query QA có Qwen grounded answer

`optional` giữ evidence nếu answer model lỗi; `required` dùng khi answer/model là
điều kiện bắt buộc của request.

```powershell
$env:QA_ANSWER_MODE = "optional"

.\.venv\Scripts\python.exe -B -m backend.app.pipelines.online_pipeline `
  --task qa `
  --query "Người phụ nữ mặc áo đỏ đang cầm vật gì?" `
  --top-k 5 `
  --debug `
  --output data\reports\online_qa_answer.json
```

Muốn QA dùng BGE-M3 artifact đã build offline:

```powershell
$env:QA_BGE_DENSE_ENABLED = "true"
$env:QA_BGE_RERANKER_ENABLED = "true"
$env:QA_BGE_INDEX_ROOT = "data\indexes\bge_m3"
```

QA BGE, QA answerer và KIS/AVS query expansion là ba feature độc lập. Bật một cái
không tự bật hai cái còn lại.

Caller có thể cấp expansion riêng cho QA bằng cách lặp `--expanded-query`:

```powershell
.\.venv\Scripts\python.exe -B -m backend.app.pipelines.online_pipeline `
  --task qa `
  --query "Người phụ nữ đang cầm vật gì?" `
  --expanded-query "vật thể trong tay người phụ nữ" `
  --expanded-query "đồ vật được người phụ nữ cầm" `
  --top-k 5
```

Đây là caller-owned QA expansion, không phải generic Qwen KIS/AVS expansion.

## 8. Smoke test online trên artifact thật

Smoke runner gọi lại cùng `search_online()`; nó không build submission. `all`
chỉ gồm KIS, AVS, temporal và QA, **không gồm TRAKE**.

```powershell
.\.venv\Scripts\python.exe -B -m backend.app.services.retrieval.run_task_smoke `
  --task all `
  --top-k 5 `
  --output data\reports\online_smoke.json
```

QA trong smoke runner là strict: để chứng minh full QA path, bật BGE dense,
reranker và answerer bắt buộc trước:

```powershell
$env:QA_BGE_DENSE_ENABLED = "true"
$env:QA_BGE_RERANKER_ENABLED = "true"
$env:QA_BGE_INDEX_ROOT = "data\indexes\bge_m3"
$env:QA_ANSWER_MODE = "required"

.\.venv\Scripts\python.exe -B -m backend.app.services.retrieval.run_task_smoke `
  --task qa `
  --top-k 5 `
  --output data\reports\qa_strict_smoke.json
```

TRAKE phải smoke bằng command `online_pipeline --task trake` ở trên.

## 9. Gọi từ Python

```python
from backend.app.services.retrieval.retrieval_manager import search_online

kis = search_online(
    query="người đàn ông mở cửa xe",
    task="kis",
    top_k=20,
    include_context=True,
    debug=True,
)
print(kis["candidates"])

trake = search_online(
    query=(
        "Context: a jump. Events: "
        "1. first leaves the ground "
        "2. reaches peak height "
        "3. lands"
    ),
    task="trake",
    top_k=100,
)
print(trake["hypotheses"])

qa = search_online(
    query="Người phụ nữ đang cầm vật gì?",
    task="qa",
    top_k=5,
)
print(qa.get("answer"), qa.get("evidence"))
```

`include_context=True` chỉ override request. Context artifact phải được load lúc
factory dựng pipeline, nên vẫn cần hai biến `ONLINE_*_CONTEXT_ENABLED=true`.

## 10. Xuất CSV submission

CLI export hỗ trợ `kis`, `qa` và `trake`, giới hạn 1–100 row:

```powershell
# KIS
.\.venv\Scripts\python.exe -B -m backend.app.services.submission.export_query `
  --task kis `
  --query "người mặc áo đỏ cầm điện thoại" `
  --top-k 100 `
  --output data\submissions\kis_result.csv

# QA: chỉ export khi grounded answer/citation hợp lệ
.\.venv\Scripts\python.exe -B -m backend.app.services.submission.export_query `
  --task qa `
  --query "Người phụ nữ đang cầm vật gì?" `
  --top-k 5 `
  --output data\submissions\qa_result.csv

# TRAKE: mỗi row là một complete same-video sequence
.\.venv\Scripts\python.exe -B -m backend.app.services.submission.export_query `
  --task trake `
  --query "Context: a jump. Events: 1. first leaves ground 2. reaches peak 3. lands" `
  --top-k 100 `
  --output data\submissions\trake_result.csv
```

Trong submission, `frame_id` phải là original zero-based `frame_index`, không
phải thứ tự keyframe, timestamp, tên JPEG, internal frame ID hay FAISS row.

## 11. HTTP/API status

Repo hiện có `APIRouter` contract tại `backend/app/api/retrieval.py` và
`backend/app/api/search.py`, nhưng **chưa có** `FastAPI()` app factory, router
mounting, health endpoint hoặc host/port chuẩn. Vì vậy README không đưa lệnh
`uvicorn` giả vờ chạy được.

Query production hiện nên đi qua CLI/Python ở trên. Khi app host được bổ sung,
router online đã có contract `POST /retrieval/online` và `POST /search`; chi tiết
body/response nằm tại [`docs/api_contract.md`](docs/api_contract.md).

Frontend hiện cũng chỉ là placeholder, chưa có UI runnable.

## 12. Test code

Chạy toàn bộ test suite:

```powershell
.\.venv\Scripts\python.exe -B -m unittest discover -s backend/tests -v
```

Hoặc dùng pytest collector:

```powershell
.\.venv\Scripts\python.exe -B -m pytest -q
```

Kiểm import/syntax:

```powershell
.\.venv\Scripts\python.exe -B -m compileall -q backend src
```

Unit/integration test chủ yếu dùng fixture hoặc fake model. Nó chứng minh schema,
ranking policy và artifact contract; nó không chứng minh model thật đạt chất
lượng trên full dataset. Muốn claim E2E phải có log offline thật, committed
manifest, online trace và đánh giá có ground truth.

## 13. Artifact chính

| Nhóm | Path mặc định | Vai trò |
|---|---|---|
| Video gốc | `data/raw/video/` | Nguồn offline và TRAKE local refinement |
| Dense candidates | `data/candidates/`, `data/dense_keyframes/` | Full candidate pool trước selection |
| Candidate features | `data/candidate_features/` | SigLIP2/caption/OCR/object theo video |
| Selected keyframes | `data/keyframes/`, `data/metadata/keyframes_*.jsonl` | Canonical sparse frames |
| Selected visual index | `data/indexes/siglip2_so400m_patch16_384_flat_ip.faiss` | Coarse visual retrieval |
| Dense visual index | `data/indexes/siglip2_so400m_patch16_384_dense_flat_ip.faiss` | Global rescue và CSES |
| Sparse text index | `data/indexes/retrieval_text_index.json` | BM25 caption/OCR/objects |
| BGE-M3 | `data/indexes/bge_m3/` | Optional dense text cho QA/TRAKE |
| Context | `data/metadata/neighbors_all.jsonl`, `segments_all.jsonl` | Bounded neighbor/segment evidence |
| Corpus commit | `data/metadata/offline_corpus_manifest.json` | Checksum và bundle generation |
| Reports | `data/reports/` | Stage, corpus và query trace |
| Submission | `data/submissions/` | CSV được export chủ động |

Không trộn index, frame map, manifest và report từ các lần build khác nhau. Online
cache được pin theo `bundle_generation`; publish corpus mới sẽ tạo generation mới.

## 14. Troubleshooting

### FFmpeg không được tìm thấy

```powershell
Get-Command ffmpeg
Get-Command ffprobe
```

Cài FFmpeg, mở terminal mới rồi chạy lại hai lệnh. Đừng sửa code decode để che
lỗi `PATH`.

### CUDA/Paddle lỗi hoặc OOM

```powershell
.\.venv\Scripts\python.exe -c "import torch, paddle; print(torch.cuda.is_available(), paddle.is_compiled_with_cuda())"
```

- Xác nhận chỉ có một Paddle profile.
- Xác nhận wheel CUDA khớp driver.
- Giảm từng batch size như profile mục 6.4.
- Dùng `--device cpu` chỉ để debug vì full run sẽ chậm đáng kể.

### Model download timeout hoặc máy offline

- Chạy một lần có mạng để model vào `data/model_cache/`.
- Chép nguyên cache sang máy đích, giữ đúng revision.
- Sau đó mới dùng `--local-files-only`, `QA_MODELS_LOCAL_ONLY=true` hoặc
  `RETRIEVAL_TRAKE_BGE_LOCAL_FILES_ONLY=true`.

### KIS/AVS chỉ chạy selected-only

Mở JSON output và xem:

```text
routing_trace.coarse_to_dense.mode
routing_trace.coarse_to_dense.fallback_reason
```

`selected_only_fallback` thường do dense bundle thiếu. Bundle có mặt nhưng corrupt
hoặc sai generation phải fail closed; không nên đổi config để nuốt lỗi checksum.

### Qwen expansion không chạy

Kiểm lần lượt:

```powershell
$env:RETRIEVAL_QUERY_EXPANSION_ENABLED
Get-Content configs\retrieval.yaml | Select-String -Pattern "query_expansion|enabled"
```

Sau đó xem `routing_trace.query_expansion` trong response. `temporal`, `qa` và
`trake` skip generic expansion là hành vi đúng; chỉ KIS/AVS mới gọi provider.

### Context không góp điểm

Xác nhận artifact và flag:

```powershell
Test-Path data\metadata\neighbors_all.jsonl
Test-Path data\metadata\segments_all.jsonl
$env:ONLINE_NEIGHBOR_CONTEXT_ENABLED
$env:ONLINE_SEGMENT_CONTEXT_ENABLED
```

Rồi query lại với `--with-context --debug` và xem
`routing_trace.context_scoring`.

### QA trả `insufficient_evidence`

Đây là fail-closed khi evidence, temporal chain hoặc citation chưa đủ; không nên
bù câu trả lời bằng kiến thức ngoài video. Test evidence trước với
`QA_ANSWER_MODE=off`, sau đó mới bật answerer.

### TRAKE không refine được original frame

Kiểm `RETRIEVAL_TRAKE_VIDEO_ROOT`, tên `<video_id>.mp4` và warning trong
`trace.refinement`. Khi video/decode/local scoring unavailable, pipeline có thể
giữ coarse frame thay vì bịa original-frame lineage.

## 15. Giới hạn cần biết

- Chưa có runnable FastAPI app hoặc frontend.
- Không có ASR/audio/transcript retrieval.
- BGE reranker revision `main` là dev default, chưa reproducible tuyệt đối; pin
  commit hash trước benchmark chính thức.
- First-run model download có thể rất lâu.
- TRAKE local refinement đã được wire với shared SigLIP2, nhưng semantic-boundary
  accuracy vẫn cần benchmark có nhãn trên full corpus.
- Test pass không đồng nghĩa retrieval quality đã đạt yêu cầu cuộc thi.

Các tài liệu sâu hơn:

- [`docs/architecture.md`](docs/architecture.md): kiến trúc và boundary giữa các route.
- [`docs/api_contract.md`](docs/api_contract.md): router contract hiện có.
- [`docs/eval_protocol.md`](docs/eval_protocol.md): protocol đánh giá TRAKE.
- [`docs/PIPELINE_AUDIT.md`](docs/PIPELINE_AUDIT.md): evidence/risk audit của pipeline.
- [`data/README.md`](data/README.md): quy ước dữ liệu và `frame_id`.
