# AIChallenge26 Multimodal Agentic Video Retrieval System

Hệ thống truy hồi video đa phương thức cho KIS và hỏi đáp có dẫn chứng (QA).
TRAKE được dự kiến bổ sung sau nhưng chưa được triển khai trong phạm vi hiện tại.
Kiến trúc hiện hành chỉ dùng thông tin thị giác: ảnh keyframe, caption, OCR và
object labels. **Không có ASR, không đọc audio và không lập chỉ mục transcript.**

Các phần chính của hệ thống:

| Phạm vi | Vai trò | Nguồn sự thật | Entrypoint | Input | Output |
|---|---|---|---|---|---|
| `backend/` | Backend chính | Canonical implementation | Các module `python -m backend...` | Video, metadata, query | Keyframe/index, KIS/QA result |
| `src/indexing/` | Tiện ích dùng chung | Code neighbor/segment thực tế | Backend wrapper gọi | Canonical metadata | Neighbor/segment artifacts |
| `frontend/` | Placeholder | Không có implementation | Chưa có | — | Chưa có UI chạy được |

## Pipeline thực tế

```text
OFFLINE
video
  -> dense candidates + shot anchors/endpoints
  -> SigLIP2 + Qwen caption + PP-OCRv5 + YOLOE features
  -> protected-event/MMR/dedup/gap-aware keyframe selection
  -> canonical keyframes
  -> SigLIP2 FAISS + neighbor/segment metadata
  -> BM25 text index + optional BGE-M3 dense text index

ONLINE (một entrypoint: search_online -> OnlinePipeline)
query + task (KIS/AVS/temporal/QA/auto)
  -> typed query plan -> visual/text/temporal retrieval
  -> weighted fusion -> deterministic rerank -> canonical candidates
  -> QA only: evidence bundle -> optional Qwen grounded answer, or abstain
```

- KIS trả về frame/segment đã xếp hạng từ visual, caption, OCR, objects và
  temporal metadata.
- QA dùng cùng retrieval stack, tối đa Top-3 evidence cho câu hỏi thường hoặc
  strict temporal chain tối đa 5 event.
  Answerer chỉ nhận caption/OCR/object/image metadata, sinh JSON có citation và
  phải abstain khi evidence không đủ.
- Query expansion thuộc query planner cho KIS nâng cao. External expansion bị bỏ
  qua ở temporal QA để không phá cấu trúc chuỗi sự kiện; quyết định này có trace.

Chi tiết truy vết theo file/function và risk register nằm tại
[`docs/PIPELINE_AUDIT.md`](docs/PIPELINE_AUDIT.md).

## Model và artifact contract

| Thành phần | Model/checkpoint | Phương pháp | File triển khai | Thiết bị | Artifact |
|---|---|---|---|---|---|
| Shot detection | `TransNetV2` | Shot boundaries + dense sampling | `indexing/extract_keyframes.py`, `keyframe_candidates.py` | CPU/CUDA (`auto`) | keyframe JSONL/report |
| Visual embedding | `google/siglip2-so400m-patch16-384` | normalized embedding, FAISS IP | `build_siglip2_index.py`, `search_visual.py` | CPU/CUDA | `.npy`, FAISS, map/manifest |
| Caption | `Qwen/Qwen3-VL-8B-Instruct` @ `b5bc35aa2d1dc2db88ca1482375afc801511bffb` | Structured frame caption | `ingestion/caption_pipeline.py` | CPU/CUDA | caption JSONL/report |
| OCR | `PP-OCRv5_server_det` + `latin_PP-OCRv5_mobile_rec` | Vietnamese/English OCR | `ingestion/ocr_pipeline.py` | CPU/CUDA | OCR JSONL/report |
| Object evidence | `yoloe-26l-seg.pt` | Open-vocabulary soft evidence | `ingestion/object_pipeline.py` | CPU/CUDA | object JSONL/report |
| Keyframe selection | Không có checkpoint | protected events, gap repair, dedup, MMR | `keyframe_selection.py` | CPU | selected metadata/ledgers |
| Sparse text | Không có checkpoint | BM25 caption/OCR/objects | `build_text_index.py`, `text_index.py` | CPU | text-index JSON |
| Dense text | `BAAI/bge-m3` | 1024-d normalized FAISS IP | `build_bge_m3_index.py`, `bge_dense.py` | CPU/CUDA | index + map + manifest |
| Text reranker | `BAAI/bge-reranker-v2-m3` | Cross-encoder blend | `bge_reranker.py` | CPU/CUDA | scores/trace |
| Grounded QA | `Qwen/Qwen3.5-9B` @ `c202236235762e1c871ad0ccb60c8ee5ba337b9a` | Evidence-only JSON + citation validation | `qa_answerer.py` | CPU/CUDA; lazy-load | answer/cache/report |
| Query expansion | `Qwen/Qwen3.5-9B` @ `c202236` | bounded paraphrase/decomposition | `agent/query_expansion.py`, `query_plan.py` | CPU/CUDA | expansion plan/trace |

Không có model ASR trong runtime hoặc dependency manifest. Các revision `main`
của BGE nên được thay bằng commit hash trước một benchmark chính thức.

## Cài đặt

Khuyến nghị Python 3.11 hoặc 3.12 và FFmpeg/ffprobe có trong `PATH`. PyTorch và
PaddlePaddle được cài riêng trước `requirements.txt` để pip không thay nhầm wheel
CPU/CUDA. Chỉ chọn **một** profile dưới đây.

Code không áp đặt mức RAM/VRAM tối thiểu. Full corpus cần dung lượng cho video,
dense candidates và model cache; Qwen 8B/9B phải được profile batch, VRAM và disk
trên máy đích thay vì dựa vào một ước lượng chung.

### CPU

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install paddlepaddle==3.2.0
python -m pip install -r requirements.txt
```

### NVIDIA CUDA 11.8

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu118
python -m pip install paddlepaddle-gpu==3.2.0 --index-url https://www.paddlepaddle.org.cn/packages/stable/cu118/
python -m pip install -r requirements.txt
```

### NVIDIA CUDA 12.6

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu126
python -m pip install paddlepaddle-gpu==3.2.0 --index-url https://www.paddlepaddle.org.cn/packages/stable/cu126/
python -m pip install -r requirements.txt
```

Không cài đồng thời `paddlepaddle` và `paddlepaddle-gpu`. Wheel phải phù hợp với
Python, hệ điều hành và driver; xem trang cài đặt chính thức của
[PyTorch](https://pytorch.org/get-started/previous-versions/) và
[PaddlePaddle](https://www.paddlepaddle.org.cn/install/quick).

Kiểm tra môi trường:

```powershell
ffmpeg -version
ffprobe -version
python -m pip check
python -c "import torch, paddle, transformers, faiss; print({'torch': torch.__version__, 'cuda': torch.cuda.is_available(), 'paddle': paddle.__version__, 'paddle_cuda': paddle.is_compiled_with_cuda()})"
```

## Cấu hình runtime

[`configs/retrieval.yaml`](configs/retrieval.yaml) giữ hybrid weights và query
expansion. [`.env.example`](.env.example) liệt kê biến artifact/QA quan trọng.
Repository **không tự load file `.env`**; đặt biến bằng shell hoặc process manager.

| Biến/config | Bắt buộc | Mặc định | Phạm vi | Ý nghĩa |
|---|---|---|---|---|
| `RETRIEVAL_CONFIG_PATH` | Không | `configs/retrieval.yaml` | Backend | Weights, limits, query expansion |
| `RETRIEVAL_INDEX_PATH` | Khi search visual | SigLIP2 path dưới `data/indexes` | Backend | FAISS coarse index |
| `RETRIEVAL_FRAME_MAP_PATH` | Khi search visual | Path dưới `data/metadata` | Backend | Map row sang video/frame/timestamp |
| `RETRIEVAL_MANIFEST_PATH` | Khi search visual | Path dưới `data/metadata` | Backend | Encoder dimension/normalization/lineage |
| `RETRIEVAL_TEXT_INDEX_PATH` | Khi bật text | `data/indexes/retrieval_text_index.json` | Backend | Caption/OCR/object sparse index |
| `RETRIEVAL_DEVICE` | Không | `auto` | Backend | `auto`, `cpu` hoặc `cuda` |
| `ONLINE_NEIGHBOR_CONTEXT_ENABLED` | Không | `false` | Online pipeline | Đọc trực tiếp `neighbors_all.jsonl` sau rerank |
| `ONLINE_SEGMENT_CONTEXT_ENABLED` | Không | `false` | Online/temporal | Gắn canonical segment trước temporal matching |
| `ONLINE_NEIGHBOR_PATH` | Khi bật neighbor | `data/metadata/neighbors_all.jsonl` | Online pipeline | Canonical neighbor artifact |
| `ONLINE_SEGMENT_PATH` | Khi bật segment | `data/metadata/segments_all.jsonl` | Online pipeline | Canonical segment artifact |
| `QA_BGE_DENSE_ENABLED` | Không | `false` | QA backend | Bật BGE dense retrieval |
| `QA_BGE_INDEX_ROOT` | Khi bật BGE | `data/indexes/bge_m3` | QA backend | BGE index/map/manifest root |
| `QA_BGE_RERANKER_ENABLED` | Không | `false` | QA backend | Bật cross-encoder rerank |
| `QA_ANSWER_MODE` | Không | `off` | QA backend | `off`, `optional`, `required` |
| `QA_MODELS_LOCAL_ONLY` | Không | `false` | QA backend | Chỉ đọc model cache |

```powershell
$env:RETRIEVAL_INDEX_PATH = "data/indexes/siglip2_so400m_patch16_384_flat_ip.faiss"
$env:RETRIEVAL_FRAME_MAP_PATH = "data/metadata/siglip2_so400m_patch16_384_frame_map.json"
$env:RETRIEVAL_MANIFEST_PATH = "data/metadata/siglip2_so400m_patch16_384_faiss_manifest.json"
$env:RETRIEVAL_TEXT_INDEX_PATH = "data/indexes/retrieval_text_index.json"
$env:RETRIEVAL_CORPUS_MANIFEST_PATH = "data/metadata/offline_corpus_manifest.json"
$env:RETRIEVAL_DEVICE = "cuda"
```

Các lựa chọn QA đáng chú ý:

- `QA_ANSWER_MODE=off|optional|required` (mặc định `off`).
- `QA_BGE_DENSE_ENABLED=true` và `QA_BGE_INDEX_ROOT=...` để bật dense text.
- `QA_BGE_RERANKER_ENABLED=true` để bật cross-encoder.
- `QA_MODELS_LOCAL_ONLY=true` cho máy đã chuẩn bị cache và không có mạng.

Model được lazy-load và cache dưới `data/model_cache/`. Lần đầu cần mạng nếu
checkpoint chưa có. Caption/QA 4-bit hoặc 8-bit cần CUDA và bitsandbytes tương
thích; nếu OOM, giảm batch trước, sau đó mới cân nhắc quantization.

## Chạy backend canonical

Entrypoint dưới đây chạy toàn bộ video `*.mp4` trong `data/raw/video/` theo thứ
tự deterministic. Mỗi video phải hoàn tất dense candidates → materialize toàn
bộ ảnh → SigLIP2/OCR/object/caption cho toàn pool → multimodal selection →
canonical publish/validation trước khi video kế tiếp bắt đầu. Chỉ sau khi mọi
video requested thành công, pipeline mới build FAISS, BM25, BGE-M3, neighbor
mapping và segment/event metadata rồi commit tất cả thành một corpus generation.

```powershell
python -m backend.app.pipelines.offline_pipeline `
  --video-dir data/raw/video `
  --video-glob "*.mp4" `
  --output-dir data `
  --dense-interval 0.5 `
  --device cuda `
  --resume
```

Quick mode cho một video (mặc định chỉ publish/validate artifact per-video và
giữ nguyên corpus index hiện có):

```powershell
python -m backend.app.pipelines.offline_pipeline `
  --video-dir data/raw/video `
  --video-id L01_V001 `
  --output-dir data `
  --device cuda `
  --resume
```

Chỉ thêm `--build-corpus` nếu thực sự muốn **thay** global FAISS/BM25/BGE bằng
corpus chứa đúng tập video đang request. Với quick mode một video, flag này sẽ
tạo corpus một-video; pipeline không tự quét artifact cũ vì làm vậy có thể kéo
metadata stale vào index.

`--resume` là mặc định và chỉ skip stage khi contract, checksum, identity
alignment và artifact validator đều pass. `--force` recompute toàn bộ;
`--skip-bge` tắt riêng BGE-M3; `--allow-partial-corpus` cho phép index các video
thành công khi video khác fail (mặc định corpus indexing bị chặn). Full-dataset
mode build corpus mặc định; `--skip-corpus` chỉ chạy/publish các stage per-video.
Neighbor mặc định nối các selected keyframe cùng video trong cửa sổ ±5 giây;
segment mặc định dùng shot/segment boundary có sẵn và fallback cửa sổ 10 giây.
Có thể đổi bằng `--neighbor-window-seconds`, `--segment-strategy` và
`--segment-fixed-duration-seconds`.

Dense workspace được cô lập tại `data/candidates/`, `data/dense_keyframes/` và
`data/candidate_features/`, nên không thể lọt vào glob retrieval canonical.
OCR được chạy trong một child process chỉ nạp Paddle để tránh xung đột DLL cuDNN
giữa wheel Torch và Paddle trên Windows; parent pipeline vẫn dùng Torch/CUDA cho
SigLIP2, object detection và caption như bình thường.
Selected artifacts nằm tại `data/keyframes/`, `data/embeddings/` và
`data/metadata/`; per-video/corpus reports nằm dưới `data/reports/offline/`.
`data/metadata/neighbors_all.jsonl` và `segments_all.jsonl` cũng được build từ
đúng tập selected keyframe vừa hoàn tất, không scan artifact stale. FAISS, BM25,
BGE-M3, neighbor và segment đều được build/validate trong staging; manifest
corpus được publish cuối cùng. Nếu crash giữa lúc promote, runtime fail-closed
thay vì trộn artifact thuộc hai generation khác nhau.

Các CLI service riêng lẻ vẫn hữu ích để debug một stage, nhưng không thay thế
entrypoint canonical vì chúng không tự đảm bảo thứ tự full-pool multimodal.

Neighbor/segment CLI riêng vẫn dùng được để debug từng builder, nhưng canonical
offline entrypoint đã tự chạy và commit hai artifact này. Kiểm tra output:

```powershell
Test-Path data/metadata/neighbors_all.jsonl
Test-Path data/metadata/segments_all.jsonl
Get-Content data/metadata/neighbors_all.jsonl -TotalCount 1
Get-Content data/metadata/segments_all.jsonl -TotalCount 1
```

Logic chọn keyframe đa phương thức vẫn nằm trong
`backend/app/services/indexing/keyframe_multimodal_pipeline.py`; entrypoint chỉ
điều phối, checkpoint, persist và validate, không copy lại thuật toán selector.

## Cách chạy query online

Sau khi phần offline đã tạo xong index và metadata trong `data/`, mở PowerShell tại
thư mục gốc repository và chạy đúng lệnh sau:

```powershell
.\.venv\Scripts\python.exe -m backend.app.pipelines.online_pipeline --task auto --query "người đàn ông mặc áo đỏ đang mở cửa xe" --top-k 20
```

Trong đó:

- `--query`: câu bạn muốn tìm hoặc câu hỏi cần trả lời.
- `--top-k 20`: lấy tối đa 20 kết quả.
- `--task auto`: để hệ thống tự chọn cách xử lý. Nếu chưa biết chọn gì, cứ giữ
  nguyên `auto`.

Kết quả JSON được in ngay trong terminal. Muốn lưu kết quả vào file thì thêm:

```powershell
--output data/reports/online_query.json
```

Luồng chạy rất đơn giản:

```text
câu query -> OnlinePipeline -> đọc index đã có trong data/ -> trả kết quả
```

`OnlinePipeline` **không chạy lại phần offline**, không cắt video và không build
lại index. Nó chỉ tìm kiếm trên artifact mà offline đã tạo trước đó.

Nếu cần ép loại bài toán thay vì dùng `auto`, thay giá trị của `--task`:

| Giá trị | Dùng khi |
|---|---|
| `kis` | Tìm đúng một cảnh hoặc khoảnh khắc cụ thể |
| `avs` | Tìm nhiều cảnh phù hợp với mô tả |
| `temporal` | Query có thứ tự sự kiện, ví dụ “mở cửa rồi ngồi xuống” |
| `qa` | Đặt câu hỏi và cần câu trả lời dựa trên video |

Tóm lại: query bình thường chỉ chạy module `online_pipeline` ở lệnh trên.
`run_task_smoke` bên dưới chỉ dùng để kiểm tra hệ thống, không phải lệnh query
thứ hai.

### Dành cho code Python và API

Code Python, API, CSV export và smoke đều đi qua cùng một hàm `search_online()` và
cùng một `OnlinePipeline`. Ví dụ gọi trực tiếp từ Python:

```python
from backend.app.services.retrieval.retrieval_manager import search_online

response = search_online(query="người mặc áo đỏ", task="auto", top_k=20)
print(response["candidates"])
```

Neighbor/segment context là tùy chọn nâng cao và mặc định tắt. Các biến cấu hình
tương ứng được liệt kê trong bảng môi trường ở phía trên.

### Kiểm định cùng online pipeline bằng smoke

`run_task_smoke` không phải cách query thứ hai. Đây chỉ là validator gọi lại
`search_online()` để kiểm tra artifact, ảnh evidence và các model bắt buộc. Chỉ
dùng lệnh này khi cần xác nhận môi trường chạy đúng; query bình thường dùng CLI ở
trên. Chương trình không dừng lại để hỏi query trong terminal và tự dùng câu mặc
định nếu không truyền:

| Task | Tham số để đổi query | Query mặc định |
|---|---|---|
| KIS | `--kis-query` | `người phụ nữ mặc áo đỏ đang cầm điện thoại` |
| AVS | `--avs-query` | Query AVS mặc định trong smoke runner |
| Temporal | `--temporal-query` | `a person enters a room, then sits down` |
| QA | `--qa-query` | `Người phụ nữ mặc áo đỏ đang cầm vật gì?` |

Với task QA, đây là **strict smoke**, không phải sanity check tối giản: validator
yêu cầu BGE-M3 dense retrieval, BGE cross-encoder và answer model thực sự được áp
dụng. Vì vậy phải build `data/indexes/bge_m3/{bge_m3_flat_ip.faiss,
bge_m3_frame_map.json,bge_m3_manifest.json}` từ canonical selected-keyframe hoặc
segment metadata, rồi đặt các biến dưới đây trước khi chạy QA:

```powershell
$env:QA_BGE_DENSE_ENABLED = "true"
$env:QA_BGE_RERANKER_ENABLED = "true"
$env:QA_BGE_INDEX_ROOT = "data/indexes/bge_m3"
$env:QA_BGE_DEVICE = "cuda"
```

Không bỏ `--canonical-only` chỉ để smoke qua validation. Nếu build báo metadata
không phải `selected_keyframe` hoặc canonical segment, hãy tạo lại canonical
metadata trước. Repository không tự load `.env`; các biến phải tồn tại trong
đúng terminal đang chạy lệnh.

Chạy với query mặc định:

```powershell
python -m backend.app.services.retrieval.run_task_smoke --task kis --top-k 20
$env:QA_ANSWER_MODE = "required"
python -m backend.app.services.retrieval.run_task_smoke --task qa --top-k 5
```

Chạy với query tự chọn và ghi report để chẩn đoán:

```powershell
python -m backend.app.services.retrieval.run_task_smoke `
  --task kis `
  --kis-query "một người đàn ông mặc áo xanh đang cầm điện thoại" `
  --top-k 20 `
  --output data/reports/kis_smoke.json

$env:QA_ANSWER_MODE = "required"
python -m backend.app.services.retrieval.run_task_smoke `
  --task qa `
  --qa-query "Người phụ nữ đang cầm vật gì?" `
  --top-k 5
```

KIS, AVS và temporal không gọi Qwen answerer. Chỉ QA dùng `QA_ANSWER_MODE`;
query expansion, BGE dense và BGE reranker có feature flag riêng.
Các lỗi `bge_dense_not_enabled`, `bge_reranker_not_enabled`,
`routing_bge_dense_not_applied` hoặc `routing_reranker_not_applied` nghĩa là strict
smoke chưa chứng minh được BGE path, không đồng nghĩa visual retrieval không chạy.

Hiện tại chưa có `FastAPI()` application factory, router mounting, health
endpoint hoặc host/port chuẩn. `backend/app/api/retrieval.py` và `search.py` chỉ
định nghĩa router/service contract; không chạy `uvicorn` cho đến khi bổ sung app
entrypoint. Route online chính khi được mount là `/retrieval/online` hoặc
`/search`; các route cũ và modality-only được giữ làm alias/diagnostic nhưng đều
ủy quyền task public về cùng `OnlinePipeline`.

Contract đã triển khai trong router (nhưng chưa phục vụ HTTP) nhận JSON như
`{"query":"người đang đi xe đạp","top_k":20}`. Response envelope là
`{"success":true,"data":...,"message":null}`; retrieval result chứa video/frame
identity, timestamp, score và modality metadata theo `models/retrieval.py`. QA
nhận thêm `task_mode` và `expanded_queries`, trả answer status/citations cùng
evidence bundle. Score chỉ dùng để xếp hạng trong cùng query/path, không phải xác
suất đã hiệu chỉnh. Hiện không có host, port hay port-conflict policy.

### Xuất CSV KIS và QA

Router `backend/app/api/search.py` cung cấp contract `POST /search/export` để
mount vào FastAPI application sau này:

```json
{"query":"người mặc áo đỏ cầm điện thoại","task":"kis","top_k":100}
```

Response là `text/csv; charset=utf-8` với `Content-Disposition: attachment`.
KIS dùng header `video_id,frame_id`; QA dùng
`video_id,frame_id,answer`. Ranking được giữ nguyên, cặp frame trùng bị loại theo
lần xuất hiện đầu tiên và không tạo row giả. QA chỉ xuất khi grounded answer có
`status=answered`, nội dung không rỗng và citation hợp lệ; abstain hoặc thiếu dẫn
chứng trả lỗi rõ ràng. `top_k` chỉ nhận từ 1 đến 100 và TRAKE bị từ chối vì chưa
được triển khai.

CLI dùng chung serializer/service với API:

```powershell
python -m backend.app.services.submission.export_query `
  --task kis --query "người mặc áo đỏ cầm điện thoại" --top-k 100 `
  --output data/submissions/kis_result.csv
```

Khi chưa có `data/sample_submission.csv` chính thức, `video_id` là stem không có
`.mp4`. `frame_id` luôn lấy từ trường `frame_index` ánh xạ về video gốc; không
dùng ordinal của keyframe, timestamp, tên file hoặc FAISS row.

Để chuẩn bị caption offline, tải đúng `Qwen/Qwen3-VL-8B-Instruct` revision
`b5bc35aa2d1dc2db88ca1482375afc801511bffb` vào
`data/model_cache/caption` trên máy có mạng, sau đó chuyển nguyên cache sang máy
chạy. Cần profile VRAM theo dtype, quantization và batch size của model 8B; repo
không công bố một con số VRAM cố định chưa được đo.

## Artifact và lineage

| Nhóm | Vị trí mặc định/điển hình | Nội dung |
|---|---|---|
| Frame/keyframe | `data/keyframes/` | JPEG và canonical candidate/frame identity |
| Metadata | `data/metadata/` | keyframe, caption, OCR, objects, segments, neighbors |
| Embedding/index | `data/embeddings/`, `data/indexes/` | SigLIP2 arrays/FAISS, BGE FAISS, maps/manifests |
| Cache | `data/model_cache/`, `data/cache/` | Downloaded checkpoints và grounded-answer cache |
| Submission/report | `data/submissions/`, `data/reports/` | CSV đã lưu và báo cáo lineage/runtime |

Không có transcript/ASR artifact trong contract hiện tại. Không ghép file từ các
run chỉ dựa vào tên: loader kiểm tra dimension, normalization, row identity và
manifest lineage; FPS/timestamp được giữ theo từng video.

## Test và trạng thái xác minh

```powershell
python -m unittest discover -s backend/tests -v
python -m compileall -q backend src
```

Unit/integration tests dùng fixture/fake model để kiểm tra schema, deterministic
ranking, failure policy và artifact contract. Chúng không thay thế một E2E run
với checkpoint thật, FFmpeg, Paddle và video dataset thật.

## Troubleshooting

- `ffmpeg`/`ffprobe` not found: cài FFmpeg và mở terminal mới để cập nhật `PATH`.
- Paddle import hoặc CUDA lỗi: gỡ cả hai Paddle packages rồi cài lại đúng một
  profile; kiểm tra `paddle.is_compiled_with_cuda()`.
- CUDA OOM: giảm caption/BGE batch, chạy từng stage/process, rồi cân nhắc 4-bit.
- Model download timeout: chuẩn bị `data/model_cache/`, pin revision và dùng
  `QA_MODELS_LOCAL_ONLY=true` sau khi cache hoàn tất.
- Authentication: code không yêu cầu application API key; nếu model hub yêu cầu
  quyền truy cập, đăng nhập/cấp token qua cơ chế của Hugging Face, không commit token.
- Artifact mismatch: không trộn index, frame map và manifest từ các run/model
  revision khác nhau; rebuild các artifact phụ thuộc trong thư mục output mới.
- Windows/PowerShell: dùng backtick ở cuối dòng và chạy từ repository root; path
  có khoảng trắng phải được quote. Không có port conflict khi chưa có web app.
- Dấu vết ASR cũ: không thêm field/weight để “sửa” artifact; rebuild bằng pipeline
  hiện tại hoặc giữ artifact chỉ để audit, vì runtime scoring không cần ASR.
- QA trả `insufficient_evidence`: đây là fail-closed khi evidence rỗng, temporal
  chain không strict hoặc citation không hợp lệ; không tự bù bằng kiến thức ngoài.

## Giới hạn hiện tại

- Chưa có web application entrypoint, health endpoint và frontend triển khai.
- Chưa có retry cho grounded answer generation; timeout/failure ở mode `required`
  được nâng thành lỗi nhưng evidence vẫn được giữ để chẩn đoán.
- BGE revision mặc định `main` chưa reproducible tuyệt đối.
- Không có ASR nên câu hỏi chỉ xuất hiện trong lời nói có thể giảm recall; đây là
  trade-off tài nguyên có chủ đích, không phải lỗi audio.
- Backend hiện dùng được ở mức library/CLI nhưng **chưa production/E2E-certified**
  cho đến khi chạy smoke thật trên máy có FFmpeg, Paddle, GPU/model cache và dataset.
