# AIChallenge26 Multimodal Agentic Video Retrieval System

Hệ thống truy hồi video đa phương thức cho KIS, AVS, truy hồi temporal, TRAKE
(Temporal Retrieval and Alignment of Key Events) và hỏi đáp có dẫn chứng (QA).
Kiến trúc hiện hành chỉ dùng thông tin thị giác: ảnh keyframe, caption, OCR và
object labels. **Không có ASR, không đọc audio và không lập chỉ mục transcript.**

Các phần chính của hệ thống:

| Phạm vi | Vai trò | Nguồn sự thật | Entrypoint | Input | Output |
|---|---|---|---|---|---|
| `backend/` | Backend chính | Canonical implementation | Các module `python -m backend...` | Video, metadata, query | Keyframe/index, KIS/QA/TRAKE result |
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
query + task (KIS/Visual KIS/Temporal KIS/AVS/temporal/TRAKE/QA/auto)
  -> KIS/Temporal KIS/AVS: typed plan -> coarse multimodal retrieval
       -> dense global rescue -> per-clip CSES -> deterministic rerank
  -> Visual KIS: visual-scoped plan -> selected SigLIP2 coarse clips
       -> dense SigLIP2 global rescue -> per-clip CSES -> visual rerank
  -> temporal/QA: existing evidence flow
  -> TRAKE: conservative event parser -> event-wise hybrid retrieval
            -> video coverage gating -> K-best original-frame alignment
            -> optional bounded local refinement -> diverse sequence ranking
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
- `kis_temporal` là KIS retrieval profile: output vẫn là `task="kis"` và ranked
  `candidates`, nhưng query plan/CSES dùng profile `temporal` (weights
  `0.55/0.15/0.30`) và giữ temporal cues cùng ordered clause identity.
- KIST Visual trên UI gửi `mode="kis_visual"`. Route này trả ordinary KIS
  `candidates`, giữ `modality_scope=["visual"]`, không gọi caption/OCR/object ở
  coarse stage, rồi chạy dense rescue/CSES khi dense artifact khả dụng. Route
  diagnostic `mode="visual"` và `POST /retrieval/visual` vẫn chỉ kiểm tra
  selected-keyframe visual engine độc lập.
- `temporal` và `trake` là hai task khác nhau. `temporal` giữ contract evidence
  phục vụ QA hiện hữu; `trake` trả một ranked list các sequence cùng video, mỗi
  sequence có đúng một original `frame_index` cho từng event. `auto` hiện không
  tự chuyển query sang TRAKE; caller phải truyền rõ `task="trake"`.
- Keyframe kỹ thuật là frame sparse được offline pipeline chọn để lập chỉ mục.
  Semantic keyframe của TRAKE là frame thỏa criterion của event (ví dụ lần đầu
  chạm, rời hoàn toàn hoặc đạt đỉnh) và có thể nằm cạnh keyframe kỹ thuật. Local
  refinement chỉ dò một cửa sổ bounded quanh coarse frame; không tạo dense index
  toàn corpus.
g
Chi tiết truy vết theo file/function và risk register nằm tại
[`docs/PIPELINE_AUDIT.md`](docs/PIPELINE_AUDIT.md).

## Model và artifact contract

| Thành phần | Model/checkpoint | Phương pháp | File triển khai | Thiết bị | Artifact |
|---|---|---|---|---|---|
| Shot detection | `TransNetV2` | Shot boundaries + dense sampling | `indexing/extract_keyframes.py`, `keyframe_candidates.py` | CPU/CUDA (`auto`) | keyframe JSONL/report |
| Visual embedding | `google/siglip2-so400m-patch16-384` | normalized embedding, FAISS IP | `build_siglip2_index.py`, `search_visual.py` | CPU/CUDA | `.npy`, FAISS, map/manifest |
| Caption | `florence-community/Florence-2-base-ft` (~0.23B) @ `0b03b6f15a4a211370fb204aee4e7dd48887ea37` | `<MORE_DETAILED_CAPTION>` frame caption | `ingestion/caption_pipeline.py` | CPU/CUDA | caption JSONL/report |
| OCR | `PP-OCRv5_server_det` + `latin_PP-OCRv5_mobile_rec` | Vietnamese/English OCR | `ingestion/ocr_pipeline.py` | CPU/CUDA | OCR JSONL/report |
| Object evidence | `yoloe-26l-seg.pt` | Open-vocabulary soft evidence | `ingestion/object_pipeline.py` | CPU/CUDA | object JSONL/report |
| Keyframe selection | Không có checkpoint | protected events, gap repair, dedup, MMR | `keyframe_selection.py` | CPU | selected metadata/ledgers |
| Sparse text | Không có checkpoint | BM25 caption/OCR/objects | `build_text_index.py`, `text_index.py` | CPU | text-index JSON |
| Dense text | `BAAI/bge-m3` | 1024-d normalized FAISS IP | `build_bge_m3_index.py`, `bge_dense.py` | CPU/CUDA | index + map + manifest |
| Text reranker | `BAAI/bge-reranker-v2-m3` | Cross-encoder blend | `bge_reranker.py` | CPU/CUDA | scores/trace |
| Grounded QA | `Qwen/Qwen3.5-2B` @ `15852e8c16360a2fea060d615a32b45270f8a8fc` | Multimodal evidence-only JSON + citation validation; `auto` → 4-bit trên CUDA | `qa_answerer.py` | CUDA mặc định; chỉ lazy-load trên route QA | answer/cache/report |
| Query expansion | `Qwen/Qwen3.5-2B` @ `15852e8c16360a2fea060d615a32b45270f8a8fc` | lazy bounded paraphrase/decomposition, BitsAndBytes 4-bit | `agent/query_expansion.py`, `query_plan.py` | configured device (`cuda` in `.env`) | expansion cache/plan/trace |
| TRAKE | Không có checkpoint/scorer production đi kèm | deterministic parse, per-event retrieval, coverage gating, K-best alignment, sequence NMS; injectable local scorer | `backend/app/services/trake/` | CPU; decoder/scorer tùy injection | sequence hypotheses + lineage/trace |

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
cd AIVIETNAM-AIO-DucTam/AIChallenge26_Multimodal_Agentic_Video_Retrieval_System
python -m pip install -r requirements.txt
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install paddlepaddle==3.2.0
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

### Cài đặt và chạy Frontend local

Để chạy GUI cho ứng dụng, ta cần `Node JS` và `NPM` (phiên bản được sử dụng: `Node v22.15.0` và `npm@10.9.2`).

Để khởi chạy Web App, ta đi vào thư mục `frontend/` và chạy lệnh:

```powershell
npm install # Tải về các thư viện cần thiết
npm run dev # Khởi chạy Web App
```

Từ thư mục `root` của repository, khởi động server Backend FastAPI thông qua lệnh:

```powershell
# Kích hoạt venv trước đó
uvicorn backend.run:app --reload
```

Truy cập Web App frontend ở `http://localhost:5173/`.

## Cấu hình runtime

[`configs/retrieval.yaml`](configs/retrieval.yaml) giữ hybrid weights, query
expansion và section `trake`. [`.env`](.env) là file cấu hình chung duy nhất cho
toàn repository; backend server và các retrieval CLI đều nạp file này bằng đường
dẫn tuyệt đối từ repository root. Biến đã đặt bằng shell hoặc process manager
được ưu tiên hơn giá trị trong file.

| Biến/config | Bắt buộc | Mặc định | Phạm vi | Ý nghĩa |
|---|---|---|---|---|
| `RETRIEVAL_CONFIG_PATH` | Không | `configs/retrieval.yaml` | Backend | Weights, limits, query expansion |
| `RETRIEVAL_INDEX_PATH` | Khi search visual | SigLIP2 path dưới `data/indexes` | Backend | FAISS coarse index |
| `RETRIEVAL_FRAME_MAP_PATH` | Khi search visual | Path dưới `data/metadata` | Backend | Map row sang video/frame/timestamp |
| `RETRIEVAL_MANIFEST_PATH` | Khi search visual | Path dưới `data/metadata` | Backend | Encoder dimension/normalization/lineage |
| `RETRIEVAL_TEXT_INDEX_PATH` | Khi bật text | `data/indexes/retrieval_text_index.json` | Backend | Caption/OCR/object sparse index |
| `RETRIEVAL_DEVICE` | Không | `auto` | Backend | `auto`, `cpu` hoặc `cuda` |
| `RETRIEVAL_QUERY_EXPANSION_MODEL_NAME` | Không | `Qwen/Qwen3.5-2B` | KIS query planner | Override model query expansion; không đổi QA answerer |
| `RETRIEVAL_QUERY_EXPANSION_MODEL_REVISION` | Không | immutable 2B SHA trong YAML | KIS query planner | Revision riêng của 2B |
| `RETRIEVAL_QUERY_EXPANSION_QUANTIZATION` | Không | `4bit` | KIS query planner | Truyền `BitsAndBytesConfig(load_in_4bit=true)` khi load |
| `QUERY_EXPANSION_DEVICE` | Không | `cpu` trong code; `.env` dùng `cuda` | KIS query planner | Device policy của provider lazy |
| `CAPTION_MODEL` | Không | `florence-community/Florence-2-base-ft` | Offline caption | CLI rõ ràng > `.env` > hằng số code |
| `CAPTION_MODEL_REVISION` | Không | `0b03b6f...87ea37` | Offline caption | Revision Florence-2 bất biến |
| `CAPTION_MODEL_CACHE_DIR` | Không | `data/model_cache/caption` | Offline caption | Cache model caption riêng |
| `ONLINE_NEIGHBOR_CONTEXT_ENABLED` | Không | `false` | Online pipeline | Đọc trực tiếp `neighbors_all.jsonl` sau rerank |
| `ONLINE_SEGMENT_CONTEXT_ENABLED` | Không | `false` | Online/temporal | Gắn canonical segment trước temporal matching |
| `ONLINE_NEIGHBOR_PATH` | Khi bật neighbor | `data/metadata/neighbors_all.jsonl` | Online pipeline | Canonical neighbor artifact |
| `ONLINE_SEGMENT_PATH` | Khi bật segment | `data/metadata/segments_all.jsonl` | Online pipeline | Canonical segment artifact |
| `QA_BGE_DENSE_ENABLED` | Không | `true` | QA backend | Bật BGE-M3 dense evidence retrieval |
| `QA_BGE_INDEX_ROOT` | Khi bật BGE | `data/indexes/bge_m3` | QA backend | BGE index/map/manifest root |
| `QA_BGE_RERANKER_ENABLED` | Không | `false` | QA backend | Model reranker tắt; giữ deterministic/constraint reranking |
| `QA_EVIDENCE_LIMIT` | Không | `100` | QA backend/UI | Giới hạn frame trả về; độc lập với 3–5 frame gửi vào answer model |
| `QA_ANSWER_MODE` | Không | `off` | QA backend | Mặc định chỉ retrieval frame; `optional`/`required` mới lazy-load answer model |
| `QA_ANSWER_MODEL` | Không | `Qwen/Qwen3.5-2B` | QA backend | Không dùng làm reranker; chỉ route QA lazy-load |
| `QA_ANSWER_MODEL_REVISION` | Không | `15852e8c...f8a8fc` | QA backend | Revision 2B bất biến |
| `QA_ANSWER_DEVICE` / `QA_ANSWER_QUANTIZATION` | Không | `cuda` / `auto` trong `.env` | QA backend | `auto` trên CUDA được resolve thành BitsAndBytes 4-bit |
| `QA_ANSWER_MODEL_CACHE_DIR` | Không | `data/model_cache/qa_answer` | QA backend | Tách biệt khỏi `data/model_cache/query_expansion` |
| `QA_MODELS_LOCAL_ONLY` | Không | `true` | QA backend | Chỉ đọc model cache |
| `trake.*` trong retrieval YAML | Không | Xem `configs/retrieval.yaml` | TRAKE | Retrieval width, video gating, alignment, refinement, max answers và cutoffs |
| `RETRIEVAL_TRAKE_VIDEO_ROOT` | Không | `data/raw/video` | TRAKE local refinement | Canonical root để resolve `<video_id>.mp4`; không nhận path từ query |

```powershell
$env:RETRIEVAL_INDEX_PATH = "data/indexes/siglip2_so400m_patch16_384_flat_ip.faiss"
$env:RETRIEVAL_FRAME_MAP_PATH = "data/metadata/siglip2_so400m_patch16_384_frame_map.json"
$env:RETRIEVAL_MANIFEST_PATH = "data/metadata/siglip2_so400m_patch16_384_faiss_manifest.json"
$env:RETRIEVAL_TEXT_INDEX_PATH = "data/indexes/retrieval_text_index.json"
$env:RETRIEVAL_CORPUS_MANIFEST_PATH = "data/metadata/offline_corpus_manifest.json"
$env:RETRIEVAL_DEVICE = "cuda"
```

| Task | Nguồn cấu hình chính | Model nặng mặc định |
|---|---|---|
| KIS/KIST (`kis`) | `hybrid`, `weights`, `text_index`, `query_expansion` trong retrieval YAML và các `RETRIEVAL_*` artifact paths | SigLIP2; query expansion chỉ chạy khi enabled và provider khả dụng |
| QA (`qa`) | Cùng retrieval artifacts cộng các biến `QA_*` | BGE-M3 dense và Qwen answerer bật; model reranker tắt |
| TRAKE (`trake`) | Section `trake` và các override `RETRIEVAL_TRAKE_*` | BGE-M3 dense bật; model reranker tắt; vẫn dùng RRF/alignment/refinement |

Các lựa chọn QA đáng chú ý:

- `QA_ANSWER_MODE=off|optional|required` (cấu hình repository dùng `off` để ưu tiên tìm và kiểm tra frame thủ công).
- Grounded answer dùng `Qwen/Qwen3.5-2B` với image evidence và citation ID; KIS,
  temporal KIS và TRAKE không resolve answerer lazy này.
- `QA_BGE_DENSE_ENABLED=true` và `QA_BGE_INDEX_ROOT=...` để bật dense text.
- `QA_BGE_RERANKER_ENABLED=false`: không dùng cross-encoder/VLM reranker.
- `QA_MODELS_LOCAL_ONLY=true` cho máy đã chuẩn bị cache và không có mạng.

TRAKE có BGE feature flags riêng trong section `trake`; chúng không đọc hoặc suy
ra trạng thái từ `QA_BGE_DENSE_ENABLED` và `QA_BGE_RERANKER_ENABLED`:

```yaml
trake:
  bge_dense_enabled: false
  bge_dense_top_k: 300
  bge_reranker_enabled: false
  bge_reranker_top_k: 150
  retrieval_fusion: rrf
  rrf_k: 60
  hybrid_rrf_weight: 1.0
  bge_rrf_weight: 1.0
  bge_required: false
```

| YAML field | Environment override |
|---|---|
| `bge_dense_enabled` | `RETRIEVAL_TRAKE_BGE_DENSE_ENABLED` |
| `bge_dense_top_k` | `RETRIEVAL_TRAKE_BGE_DENSE_TOP_K` |
| `bge_reranker_enabled` | `RETRIEVAL_TRAKE_BGE_RERANKER_ENABLED` |
| `bge_reranker_top_k` | `RETRIEVAL_TRAKE_BGE_RERANKER_TOP_K` |
| `retrieval_fusion` | `RETRIEVAL_TRAKE_RETRIEVAL_FUSION` |
| `rrf_k` | `RETRIEVAL_TRAKE_RRF_K` |
| `hybrid_rrf_weight` | `RETRIEVAL_TRAKE_HYBRID_RRF_WEIGHT` |
| `bge_rrf_weight` | `RETRIEVAL_TRAKE_BGE_RRF_WEIGHT` |
| `bge_required` | `RETRIEVAL_TRAKE_BGE_REQUIRED` |

Các field YAML ở trên là policy/pool/rank-fusion settings. Model, artifact path và
execution settings chỉ có dưới dạng environment variable:

| Runtime-only environment variable | Mặc định | Vai trò |
|---|---|---|
| `RETRIEVAL_TRAKE_BGE_INDEX_ROOT` | `data/indexes/bge_m3` | Root BGE FAISS, frame map và manifest |
| `RETRIEVAL_TRAKE_BGE_MODEL_NAME` | `BAAI/bge-m3` | Dense encoder |
| `RETRIEVAL_TRAKE_BGE_MODEL_REVISION` | Không ép revision | Để trống dùng resolved revision trong BGE manifest; override phải khớp manifest |
| `RETRIEVAL_TRAKE_BGE_BATCH_SIZE` | `16` | Dense query batch size |
| `RETRIEVAL_TRAKE_BGE_RERANKER_MODEL` | `BAAI/bge-reranker-v2-m3` | Cross-encoder reranker |
| `RETRIEVAL_TRAKE_BGE_RERANKER_REVISION` | `main` | Optional/dev default; required mode bắt buộc commit hash bất biến |
| `RETRIEVAL_TRAKE_BGE_RERANKER_ALPHA` | `0.5` | Blend coefficient của reranker adapter |
| `RETRIEVAL_TRAKE_BGE_RERANKER_BATCH_SIZE` | `16` | Cross-encoder batch size |
| `RETRIEVAL_TRAKE_BGE_DEVICE` | `auto` | `auto`, `cpu` hoặc `cuda` |
| `RETRIEVAL_TRAKE_BGE_MODEL_CACHE_DIR` | `data/model_cache/bge_m3` | Model cache; có thể dùng chung engine với QA khi contract trùng khớp |
| `RETRIEVAL_TRAKE_BGE_LOCAL_FILES_ONLY` | `false` | Không tải mạng khi đặt `true` |

Với mỗi event, pipeline luôn chạy canonical hybrid retrieval. Khi
`bge_dense_enabled: true`, BGE-M3 tạo thêm một ranked list cho chính event đó;
`retrieval_fusion: rrf` kết hợp hybrid/BGE bằng Reciprocal Rank Fusion với `rrf_k`
và hai weight tương ứng. Optional BGE reranker chỉ xử lý tối đa
`bge_reranker_top_k` candidate ở đầu list; tail không được chấm lại nhưng vẫn được
giữ để bảo vệ recall. Union dedupe theo `(video_id, original frame_index)` và ưu
tiên canonical hybrid object trên overlap, đồng thời giữ raw/RRF contributions
trong `modality_scores`. Fusion/rerank diễn ra **theo từng event trước**
shot/temporal diversity, video coverage gating và chronological alignment; không
fusion các event thành một intent chung. Context branch vẫn hybrid-only vì chỉ là
video-level prior.

`bge_required: false` là fail-open: lỗi init/artifact/search/rerank ở nhánh BGE
được ghi warning/trace và event tiếp tục bằng canonical hybrid results.
`bge_required: true` biến cùng lỗi thành lỗi request để benchmark không âm thầm
chạy thiếu nhánh bắt buộc. Hai policy này chỉ thuộc TRAKE; `QA_BGE_*` tiếp tục điều
khiển riêng QA evidence runtime. Kiểm tra per-event status/count/fallback tại
`trace.event_retrieval.events[*].sources`, `.fusion`, `.reranker` và aggregate tại
`trace.event_retrieval.bge`; `trace.bge_contract` ghi corpus generation, resolved
dense revision/checksum cùng requested reranker revision và trạng thái `revision_pinned`,
nhưng không chứa local path, exception text hoặc query text.

TRAKE được khởi tạo lazy theo task, vì vậy request `kis` hoặc `qa` không load TRAKE
index/model; QA/temporal cũng chỉ được khởi tạo khi route tương ứng được chọn. Khi
QA và TRAKE dùng cùng dense artifact, model, revision, device và
cache directory, runtime dùng chung một BGE-M3 engine để tránh nhân đôi RAM/VRAM;
hai bộ cờ bật/tắt và fail policy vẫn hoàn toàn độc lập. `bge_required: true` là
request-time fail-closed (HTTP 503 ở API), không phải startup readiness: model vẫn
lazy-load ở TRAKE request đầu tiên. Trước benchmark, nên chạy một query warm-up và
kiểm tra `trace.bge_contract` cùng per-event status.

Mọi BGE dense runtime yêu cầu `data/metadata/offline_corpus_manifest.json` (hoặc
`RETRIEVAL_CORPUS_MANIFEST_PATH`) đã publish và khớp checksum; thiếu manifest thì
optional TRAKE quay về hybrid. Chế độ `bge_required: true` luôn yêu cầu manifest,
kể cả khi chỉ bật reranker, để benchmark có corpus lineage kiểm chứng được.

Ví dụ bật dense ở chế độ optional/fail-open trên máy đã có artifact và model cache;
model reranker vẫn tắt:

```powershell
$env:RETRIEVAL_TRAKE_BGE_DENSE_ENABLED = "true"
$env:RETRIEVAL_TRAKE_BGE_RERANKER_ENABLED = "false"
$env:RETRIEVAL_TRAKE_BGE_REQUIRED = "false"
$env:RETRIEVAL_TRAKE_BGE_INDEX_ROOT = "data/indexes/bge_m3"
$env:RETRIEVAL_TRAKE_BGE_MODEL_CACHE_DIR = "data/model_cache/bge_m3"
$env:RETRIEVAL_TRAKE_BGE_LOCAL_FILES_ONLY = "true"
python -m backend.app.pipelines.online_pipeline `
  --task trake --top-k 100 `
  --query "a jump. E1: first leaves ground E2: reaches peak"
```

Ví dụ required benchmark; lệnh sẽ fail closed nếu dense index/model không load được:

```powershell
$env:RETRIEVAL_TRAKE_BGE_DENSE_ENABLED = "true"
$env:RETRIEVAL_TRAKE_BGE_RERANKER_ENABLED = "false"
$env:RETRIEVAL_TRAKE_BGE_REQUIRED = "true"
$env:RETRIEVAL_TRAKE_BGE_INDEX_ROOT = "data/indexes/bge_m3"
$env:RETRIEVAL_TRAKE_BGE_MODEL_NAME = "BAAI/bge-m3"
$env:RETRIEVAL_TRAKE_BGE_DEVICE = "cuda"
$env:RETRIEVAL_TRAKE_BGE_MODEL_CACHE_DIR = "data/model_cache/bge_m3"
$env:RETRIEVAL_TRAKE_BGE_LOCAL_FILES_ONLY = "true"
python -m backend.app.pipelines.online_pipeline `
  --task trake --top-k 100 `
  --query "a jump. E1: first leaves ground E2: reaches peak" `
  --output data/reports/trake_bge_required.json
```

Dense `MODEL_REVISION` được cố ý để trống: loader dùng immutable resolved revision
trong BGE index manifest; đặt tùy tiện `main` có thể lệch manifest. Chỉ set
`RETRIEVAL_TRAKE_BGE_MODEL_REVISION` khi dùng đúng revision đã ghi trong manifest.
Reranker mặc định `main` chỉ dùng cho optional/dev. `bge_required: true` từ chối
`main`, local model path và revision không bất biến; hãy thay placeholder trong ví
dụ bằng đúng commit hash 40–64 ký tự hex đã tải vào cache.
Required mode cũng kiểm tra dense manifest có hub model id và resolved commit hash
bất biến; tag như `main` hoặc local model path đều bị từ chối để tránh lệch encoder
đã tạo FAISS vectors.

Section `trake` mặc định lấy 300 candidate mỗi event, giữ 30 video, dùng beam
width 200, 10 path/video, log-gap penalty, refinement window ±60 original frame
và tối đa 100 answer tại cutoffs `[1, 5, 20, 50, 100]`. Loader kiểm tra type và
bounds; mỗi field có override `RETRIEVAL_TRAKE_*` tương ứng trong
`retrieval_config.py`. Không dùng `hybrid.max_gap_seconds=180` làm hard cutoff
cho TRAKE: alignment chỉ áp soft gap penalty trên original frame indexes.

`refinement_enabled: true` chỉ bật orchestration/interface. Cached production
pipeline hiện **không inject semantic local scorer**, nên refiner giữ coarse
canonical `frame_index` và ghi warning `local_refinement_scorer_unavailable`.
Decoder/scorer có thể được inject cho test hoặc deployment riêng; repository
không tuyên bố có pose/contact model, VLM verifier hay geometric boundary
verification đang hoạt động.

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

Phase 1: Preprocess
```powershell
python -m backend.app.pipelines.offline_pipeline `
  --video-dir data/raw/video `
  --video-glob "*.mp4" `
  --output-dir data `
  --dense-interval 1.0 `
  --device cuda `
  # --ocr-batch-size [...] ` set tuỳ khả năng của GPU
  # --caption-batch-size [...] ` set tuỳ khả năng của GPU
  # Thêm --bge-batch-size [...] nếu gắn flag --build-corpus thay vì --skip-corpus
  --resume `
  --skip-corpus # Hoặc đổi thành --build-corpus nếu muốn build và sẽ cần giữ lại keyframes trong kho
  
```

Quick mode cho một video:

```powershell
python -m backend.app.pipelines.offline_pipeline `
  --video-dir data/raw/video `
  --video-id L01_V001 `
  --output-dir data `
  --dense-interval 1.0 `
  --device cuda `
  --resume `
  --skip-corpus
```

Phase 2: Build corpus (Nếu đã có đầy đủ artifacts - không cần dùng tới keyframes)
```powershell
python -m build_corpus_artifacts_only `
  --data-dir data `
  --expected-videos 873 `
  # --bge-batch-size [...] ` set tuỳ khả năng của GPU
  --device cuda
```

Workflow khuyến nghị được tách thành hai phase:
1. offline_pipeline --skip-corpus để hoàn tất toàn bộ per-video artifacts.
2. build_corpus_artifacts_only.py để build global FAISS/BM25/BGE từ artifacts.

Có thể dùng --build-corpus trực tiếp trong offline_pipeline nếu muốn chạy
monolithic workflow và vẫn giữ đầy đủ raw video/keyframe workspace local.
Corpus mới chứa đúng tập video đang request; với quick mode một video, corpus chỉ chứa video đó. Pipeline không tự quét artifact
cũ vì làm vậy có thể kéo metadata stale vào index. Chỉ dùng `--skip-corpus` khi chủ đích muốn tạo/publish artifact per-video mà chưa dùng chúng cho retrieval.

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

Kết quả JSON được in ngay trong terminal. Muốn lưu kết quả vào file, thêm
`--output data/reports/online_query.json` vào cùng command.

Luồng chạy rất đơn giản:

```text
câu query -> OnlinePipeline -> đọc index đã có trong data/ -> trả kết quả
```

`OnlinePipeline` **không chạy lại phần offline**, không cắt video và không build
lại index. Nó chỉ tìm kiếm trên artifact mà offline đã tạo trước đó.

Nếu cần ép loại bài toán thay vì dùng `auto`, chọn task theo contract sau:

| Task canonical | Tên thường gặp | Input phù hợp | Output chính | Giới hạn public |
|---|---|---|---|---|
| `kis` | KIS, KIST (Known Item Search Task) | Mô tả một cảnh/khoảnh khắc cần tìm | Ranked `candidates` là technical keyframe/segment; mỗi item có `video_id`, `keyframe_id`/`frame_id`, timestamp, score và metadata modality | Tối đa 200 |
| `kis_visual` | KIST Visual | Mô tả bằng thị giác, không dùng text metadata ở coarse stage | Canonical `task="kis"`, ordinary `candidates`, visual scope và truthful coarse-to-dense/CSES trace | Tối đa 200 |
| `kis_temporal` | Temporal KIS | Mô tả một mốc/boundary KIS như đầu tiên, cuối cùng, trước hoặc sau | Canonical `task="kis"`, ranked `candidates`, temporal query plan và routing trace; không có `temporal_matches`/`hypotheses` | Tối đa 200 |
| `avs` | Ad-hoc Video Search | Mô tả có thể khớp nhiều cảnh | Cùng candidate schema với KIS, dùng profile AVS | Tối đa 200 |
| `temporal` | Temporal evidence | Câu mô tả chuỗi dùng contract evidence/QA hiện hữu | `candidates`, `temporal_matches` và temporal trace; không phải submission TRAKE | Tối đa 200 |
| `trake` | Temporal Retrieval and Alignment of Key Events | Context và đúng N event có thứ tự | Ranked `hypotheses`; mỗi item là một complete same-video sequence gồm đúng N original zero-based `frame_index` | Tối đa 100 |
| `qa` | QA frame retrieval | Câu hỏi cần tìm frame trước khi trả lời | Ranked `evidence` hiển thị như KIS; `answer` chỉ là output tùy chọn | Tối đa 100 frame |
| `auto` | Auto router | Query KIS/AVS/QA chưa biết profile | Giữ `requested_task="auto"`, trả `task` đã resolve | Theo task đã resolve |

Tên task mà CLI/Python/API thực sự nhận là `kis`; tài liệu hoặc benchmark có thể
gọi bài toán này là KIST, nhưng `--task kist` không phải alias hợp lệ. `auto` cũng
không tự chọn TRAKE, vì vậy TRAKE luôn phải được chỉ định rõ.

Ví dụ KIS/KIST bằng canonical task `kis`:

```powershell
.\.venv\Scripts\python.exe -m backend.app.pipelines.online_pipeline `
  --task kis --top-k 20 `
  --query "người đàn ông mặc áo đỏ đang mở cửa xe" `
  --output data/reports/kis_query.json
```

Visual KIS dùng `--task kis_visual` hoặc UI mode Visual. Khi dense bundle thiếu,
route quay về selected-keyframe visual search và ghi
`coarse_to_dense.executed=false`, `cses.executed=false`; không tạo CSES
selection/breakdown giả.

Ví dụ Temporal KIS (vẫn trả KIS Top-K keyframes):

```powershell
.\.venv\Scripts\python.exe -m backend.app.pipelines.online_pipeline `
  --task kis_temporal --top-k 100 `
  --query "Khoảnh khắc đầu tiên người dẫn xuất hiện trên xích lô"
```

Khi dense artifact khả dụng, `routing_trace.coarse_to_dense.executed` và
`routing_trace.cses.executed` đều là `true`. Sparse fallback hợp lệ phải ghi cả
hai execution state tương ứng là `false`; fallback không đồng nghĩa CSES đã chạy.

Ví dụ TRAKE với context và danh sách event:

```powershell
.\.venv\Scripts\python.exe -m backend.app.pipelines.online_pipeline `
  --task trake --top-k 100 `
  --query "Bối cảnh: một người mở cửa xe. Sự kiện: 1. tay bắt đầu chạm tay nắm 2. cửa mở rộng nhất 3. người rời xe hoàn toàn" `
  --output data/reports/trake_query.json
```

Ví dụ QA chỉ truy hồi evidence, không load answer model (`QA_ANSWER_MODE=off` là
mặc định):

```powershell
$env:QA_ANSWER_MODE = "off"
.\.venv\Scripts\python.exe -m backend.app.pipelines.online_pipeline `
  --task qa --top-k 100 `
  --query "Người phụ nữ mặc áo đỏ đang cầm vật gì?" `
  --output data/reports/qa_query.json
```

Đổi `QA_ANSWER_MODE` thành `optional` để thử sinh grounded answer nhưng vẫn giữ
evidence khi model lỗi; dùng `required` khi caller muốn lỗi answer/model được nâng
thành lỗi request. Cả ba mode vẫn fail closed thành `insufficient_evidence` khi
evidence hoặc temporal chain không đủ điều kiện.

Parser TRAKE là deterministic, không gọi LLM. Format canonical là context tự do
(có thể bỏ trống), sau đó là các marker `E1:`, `E2:`... trên từng dòng. Parser
không áp giới hạn 5 event, giữ thứ tự xuất hiện và tự đánh lại index nội bộ nếu
nhãn nguồn bị lặp/nhảy số; numbered/bulleted format cũ vẫn được nhận để tương
thích. Query là untrusted data và instruction-like text chỉ được xem như dữ liệu.

Tóm lại: query bình thường chỉ chạy module `online_pipeline` ở lệnh trên.
`run_task_smoke` bên dưới chỉ dùng để kiểm tra hệ thống, không phải lệnh query
thứ hai.

### Dành cho code Python và API

Unified Python/API route, CSV export và smoke dùng `search_online()` cùng
`OnlinePipeline`. Wrapper `search_trake()` và route `/retrieval/trake` gọi trực
tiếp cùng cached `TrakePipeline`/corpus generation. Ví dụ gọi từ Python:

```python
from backend.app.services.retrieval.retrieval_manager import search_online, search_trake

kis = search_online(
    query="người đàn ông mặc áo đỏ đang mở cửa xe",
    task="kis",
    top_k=20,
)
print(kis["task"], kis["candidates"])

trake = search_online(
    query="a runner.\nE1: first touches the bar\nE2: reaches peak height",
    task="trake",
    top_k=100,
)
print(trake["hypotheses"])

# Equivalent core TRAKE wrapper, without the OnlinePipeline wrapper fields.
trake_core = search_trake(
    "a runner.\nE1: first touches the bar\nE2: reaches peak height",
    top_k=100,
)

qa = search_online(
    query="Người phụ nữ mặc áo đỏ đang cầm vật gì?",
    task="qa",
    top_k=100,
)
print(qa["evidence"], qa["answer"])
```

Response TRAKE schema `1.0` là sequence-first; `candidates`, nếu có, chỉ là alias
của các complete sequence và không phải danh sách event/frame đã flatten. Ví dụ
dưới đây lược bớt một số nested retrieval fields để tập trung vào identity:

```json
{
  "schema_version": "1.0",
  "query": "a runner.\nE1: first touches the bar\nE2: reaches peak height",
  "requested_task": "trake",
  "task": "trake",
  "top_k": 100,
  "event_plan": {
    "original_query": "a runner.\nE1: first touches the bar\nE2: reaches peak height",
    "context": "a runner",
    "events": [
      {"index": 0, "original_text": "first touches the bar", "boundary_type": "first_contact"},
      {"index": 1, "original_text": "reaches peak height", "boundary_type": "peak"}
    ],
    "parser_source": "deterministic_list",
    "confidence": 1.0,
    "warnings": []
  },
  "hypotheses": [
    {
      "rank": 1,
      "video_id": "L10_V010",
      "frame_ids": [101, 203],
      "score": 0.82,
      "score_breakdown": {},
      "path_id": "TRP-...",
      "events": [
        {"event_index": 0, "result": {"video_id": "L10_V010", "frame_id": "KF_A", "frame_index": 101}},
        {"event_index": 1, "result": {"video_id": "L10_V010", "frame_id": "KF_B", "frame_index": 203}}
      ],
      "lineage": [
        {"event_index": 0, "video_id": "L10_V010", "original_frame_index": 101, "internal_frame_id": "KF_A", "source": "canonical_metadata"},
        {"event_index": 1, "video_id": "L10_V010", "original_frame_index": 203, "internal_frame_id": "KF_B", "source": "canonical_metadata"}
      ],
      "warnings": ["local_refinement_scorer_unavailable"]
    }
  ],
  "trace": {"refinement": {"scorer_available": false}},
  "latency_ms": 12.3
}
```

`frame_ids` trong response này là original zero-based `frame_index`; internal
`RetrievalResult.frame_id` chỉ xuất hiện trong lineage để audit và không được nộp.

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
yêu cầu BGE-M3 dense retrieval và answer model thực sự được áp
dụng. Vì vậy phải build `data/indexes/bge_m3/{bge_m3_flat_ip.faiss,
bge_m3_frame_map.json,bge_m3_manifest.json}` từ canonical selected-keyframe hoặc
segment metadata, rồi đặt các biến dưới đây trước khi chạy QA:

```powershell
$env:QA_BGE_DENSE_ENABLED = "true"
$env:QA_BGE_RERANKER_ENABLED = "false"
$env:QA_BGE_INDEX_ROOT = "data/indexes/bge_m3"
$env:QA_BGE_DEVICE = "cuda"
```

Không bỏ `--canonical-only` chỉ để smoke qua validation. Nếu build báo metadata
không phải `selected_keyframe` hoặc canonical segment, hãy tạo lại canonical
metadata trước. Có thể đặt các biến trong `.env` hoặc trong terminal đang chạy
lệnh; giá trị trong terminal được ưu tiên.

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
`/search`; TRAKE còn có `POST /retrieval/trake`. Các route cũ và modality-only
được giữ làm alias/diagnostic nhưng đều ủy quyền task public về cùng
`OnlinePipeline` hoặc cached `TrakePipeline` cùng corpus generation.

Contract đã triển khai trong router (nhưng chưa phục vụ HTTP) dùng các request sau:

| Task | Route | JSON body |
|---|---|---|
| KIS/KIST | `POST /retrieval/online` | `{"query":"người đang đi xe đạp","task":"kis","top_k":20,"expanded_queries":[]}` |
| TRAKE qua online wrapper | `POST /retrieval/online` | `{"query":"a jump.\nE1: first leaves ground\nE2: reaches peak","task":"trake","top_k":100}` |
| TRAKE core wrapper | `POST /retrieval/trake` | `{"query":"a jump.\nE1: first leaves ground\nE2: reaches peak","top_k":100}` |
| QA | `POST /retrieval/qa` | `{"query":"Người phụ nữ đang cầm vật gì?","top_k":100,"task_mode":"qa","expanded_queries":[]}` |

`POST /search` là wrapper tương đương dùng field `mode`, ví dụ KIS dùng
`{"query":"...","mode":"kis","top_k":20}` và TRAKE dùng
`{"query":"...","mode":"trake","top_k":100}`. Response HTTP được bọc bởi
`{"success":true,"data":...,"message":null}`. KIS trả ranked frame candidates;
TRAKE trả complete sequence hypotheses; QA trả answer status/citations và evidence
bundle. Score chỉ dùng để xếp hạng trong cùng query/path, không phải xác suất đã
hiệu chỉnh. Hiện không có host, port hay port-conflict policy.

### Xuất CSV KIS, QA và TRAKE

Router `backend/app/api/search.py` cung cấp contract `POST /search/export` để
mount vào FastAPI application sau này:

```json
{"query":"người mặc áo đỏ cầm điện thoại","task":"kis","top_k":100}
```

Response là `text/csv; charset=utf-8` với `Content-Disposition: attachment`.
Theo hướng dẫn nộp bài chính thức, mọi CSV đều **không có header**. KIS dùng mỗi
row `video_id,frame_id`; QA dùng `video_id,frame_id,answer`. Ranking được giữ nguyên, cặp frame trùng bị loại theo
lần xuất hiện đầu tiên và không tạo row giả. QA chỉ xuất khi grounded answer có
`status=answered`, nội dung không rỗng và citation hợp lệ; abstain hoặc thiếu dẫn
chứng trả lỗi rõ ràng. TRAKE dùng row
`video_id,frame_id_1,...,frame_id_N`, trong đó mỗi row là một complete sequence
cùng video và có đúng N event. Dedupe TRAKE dùng toàn identity
`(video_id, tuple(frame_ids))`, không loại hai sequence chỉ vì chúng dùng chung
một frame. `top_k` của mọi submission chỉ nhận từ 1 đến 100.

TRAKE export kiểm tra fail-closed từng lineage entry: `event_index` phải liên tục,
`video_id` phải cùng sequence, `original_frame_index` phải là số nguyên không âm
và đúng bằng `frame_ids` tương ứng, `source` phải có. Hypothesis thiếu hoặc lệch
lineage bị bỏ; serializer không thay bằng internal frame ID, timestamp, filename
hay FAISS row.

CLI dùng chung serializer/service với API:

```powershell
python -m backend.app.services.submission.export_query `
  --task kis --query "người mặc áo đỏ cầm điện thoại" --top-k 100 `
  --output data/submissions/kis_result.csv

python -m backend.app.services.submission.export_query `
  --task trake `
  --query "a jump. E1: first leaves ground E2: reaches peak" `
  --top-k 100 --output data/submissions/trake_result.csv
```

Nút **Export to CSV** trên UI gọi `POST /api/search/export-current`, không chạy lại
retrieval, hỏi official query ID (ví dụ `query-1-kis`) rồi ghi result set hiện tại
thành đúng tên `query-1-kis.csv` trong `data/submission/`. `video_id` là stem không
có `.mp4`.
Khi QA answer mode đang tắt, UI cho phép nhập answer thủ công (tối đa 100 ký tự);
video và frame vẫn lấy từ evidence đã retrieval, không chạy model trả lời.
Mọi cột mang tên `frame_id` trong file nộp luôn lấy từ original `frame_index`;
không dùng ordinal của keyframe, timestamp, tên file, internal frame ID hoặc
FAISS row.

### Đánh giá TRAKE

Pure evaluator nằm tại `backend/app/services/evaluation/trake_metrics.py`:

```python
from backend.app.services.evaluation.trake_metrics import trake_metrics_report

ground_truth = {
    "video_id": "L10_V010",
    "intervals": [[95, 105], [145, 155], [195, 205], [245, 255]],
}
report = trake_metrics_report(response["hypotheses"], ground_truth)
```

Sai `video_id` cho R-Score bằng 0. Đúng video thì mỗi event được hit khi submitted
original frame nằm trong inclusive interval `[s_j,e_j]`, và R-Score là số hit/N.
`R@k` là R-Score tốt nhất trong top-k với k thuộc `{1,5,20,50,100}`; Final Score
là trung bình năm giá trị đó. Report còn có `video_at_1/5/20`, per-event hit rate
và matched-event ratio. Evaluator từ chối event-count mismatch, interval đảo/
âm, frame âm, duplicate whole hypothesis và hơn 100 hypothesis. Xem protocol đầy
đủ tại [`docs/eval_protocol.md`](docs/eval_protocol.md).

Caption keyframe mặc định dùng `florence-community/Florence-2-base-ft` (~0.23B tham số),
task token `<MORE_DETAILED_CAPTION>` và revision bất biến
`0b03b6f15a4a211370fb204aee4e7dd48887ea37`. Đây là checkpoint đã chuyển đổi
cho Florence-2 native trong Transformers và không cần thực thi remote code.
Florence-2 sinh văn bản caption,
không sinh JSON instruction-following; adapter giữ schema JSONL cũ với
`structured_caption: null`. Quantization 4/8-bit chưa được kiểm thử cho checkpoint
này và bị từ chối rõ ràng.

Hai caption CLI đọc model, revision, cache và task prompt theo thứ tự: argument
được truyền rõ ràng, biến `.env`, rồi hằng số code. Report caption ghi effective
model/revision, task prompt, device và model cache directory; vì vậy một model
khai báo trong `.env` không thể bị runtime âm thầm thay bằng default khác.

Ví dụ sinh caption trên CPU và CUDA:

```powershell
.\.venv\Scripts\python.exe backend\app\services\ingestion\run_caption.py `
  --metadata-path data\metadata\keyframes_video7155.jsonl --device cpu --dtype float32

.\.venv\Scripts\python.exe backend\app\services\ingestion\run_caption.py `
  --metadata-path data\metadata\keyframes_video7155.jsonl --device cuda `
  --dtype auto --batch-size 4
```

Để chuẩn bị máy chạy offline, tải đúng revision vào cache trên máy có mạng:

```powershell
hf download florence-community/Florence-2-base-ft `
  --revision 0b03b6f15a4a211370fb204aee4e7dd48887ea37 `
  --cache-dir data/model_cache/caption
```

Sau đó chuyển nguyên `data/model_cache/caption` sang máy đích và đặt
`HF_HUB_OFFLINE=1`. Pipeline caption không thay đổi Qwen3.5 dùng cho grounded QA
hoặc query expansion.

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
manifest lineage; FPS/timestamp được giữ theo từng video. TRAKE chỉ nhận candidate
có original `frame_index`; output giữ per-event lineage tới internal retrieval
identity nhưng submission chỉ ghi original indexes.

## Test và trạng thái xác minh

```powershell
python -m pytest -q
python -m unittest discover -s backend/tests -v
python -m unittest -v backend.tests.test_trake_query_parser backend.tests.test_trake_pipeline backend.tests.test_trake_submission backend.tests.test_trake_metrics
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
- TRAKE trả ít hoặc không có hypothesis: xem `trace.event_retrieval`,
  `trace.video_gating` và warnings về missing original-frame lineage/full coverage;
  không sửa bằng cách suy frame từ timestamp hoặc filename.
- `local_refinement_scorer_unavailable`: behavior production mặc định hiện tại;
  coarse canonical frame vẫn hợp lệ và được giữ. Muốn refinement thật phải inject
  một `LocalFrameScorer` đã kiểm chứng và bảo đảm video ở canonical video root.

## Giới hạn hiện tại

- Chưa có web application entrypoint, health endpoint và frontend triển khai.
- Chưa có retry cho grounded answer generation; timeout/failure ở mode `required`
  được nâng thành lỗi nhưng evidence vẫn được giữ để chẩn đoán.
- BGE reranker revision mặc định `main` chưa reproducible tuyệt đối; dense revision
  lấy immutable commit từ manifest.
- Không có ASR nên câu hỏi chỉ xuất hiện trong lời nói có thể giảm recall; đây là
  trade-off tài nguyên có chủ đích, không phải lỗi audio.
- TRAKE parser/retrieval/alignment/ranking và fallback chạy được, nhưng local
  semantic-boundary accuracy chưa được chứng minh trên full corpus. Scorer
  production chưa được wire; heuristic `first_*`/`peak` chỉ được dùng khi caller
  inject score sequence, không phải pose/contact/VLM verification.
- Header TRAKE vẫn provisional vì chưa có official `sample_submission.csv` trong
  repository; đổi constant serializer khi format chính thức được cung cấp.
- Backend hiện dùng được ở mức library/CLI nhưng **chưa production/E2E-certified**
  cho đến khi chạy smoke thật trên máy có FFmpeg, Paddle, GPU/model cache và dataset.
