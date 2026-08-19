# Backend

Backend cung cấp ingestion metadata, indexing và retrieval API cho KIS, AVS,
temporal evidence, TRAKE và grounded QA.

## Ingestion

- `run_caption.py`: Florence-2-base-ft (~0.23B) image-to-text với task token
  `<MORE_DETAILED_CAPTION>`, stable JSONL retrieval schema, lazy loading, batch và dtype.
- `run_ocr.py`: PP-OCRv5 detection + Latin recognition cho tiếng Việt/Anh,
  polygon/confidence và Unicode normalization.
- `run_object_detection.py`: YOLOE open-vocabulary, configurable vocabulary và
  evidence-only output.

Các pipeline không ghi đè keyframe source. Mỗi modality có JSONL/report riêng,
cache dưới `data/model_cache/` và resume theo model/revision.

## Indexing và retrieval

- SigLIP2 + FAISS giữ nguyên visual contract.
- Text index v3 gồm caption, OCR và object labels/counts.
- API hỗ trợ visual, caption, OCR, object, hybrid, temporal, TRAKE và QA modes.
- Object evidence là soft signal trong reranking, không phải hard filter.
- `temporal` là route evidence phục vụ QA hiện hữu. `trake` là pipeline riêng tại
  `app/services/trake/`: parse N event có thứ tự, gọi hybrid retrieval theo từng
  event, gate video theo coverage, sinh K-best ordered paths trên original
  `frame_index`, rồi xếp hạng tối đa 100 complete sequence.

Ba nguồn frame không được đánh đồng:

- **Selected-keyframe index**: technical keyframes sparse do offline selector chọn;
  đây là coarse visual source cùng caption/OCR/object evidence cho KIS/AVS.
- **Full dense-candidate index**: toàn bộ materialized dense candidates trước
  selection, publish thành FAISS + metadata/map/manifest/report riêng; KIS/AVS dùng
  nó cho global rescue và per-clip CSES. Đây vẫn không phải mọi raw video frame.
- **TRAKE local refinement**: decode raw frames trong một bounded window quanh
  coarse frame rồi chấm bằng shared SigLIP2 text/image embeddings; nhánh này không
  search full dense-candidate FAISS.

### Task contract

| Task canonical | Tên thường gặp | Output public | Giới hạn |
|---|---|---|---|
| `kis` | KIS, KIST | Ranked `candidates` theo candidate schema: video/keyframe identity, timestamp, modality/fusion/rerank scores và canonical metadata | 200 |
| `avs` | Ad-hoc Video Search | Ranked `candidates` cùng schema KIS, profile AVS | 200 |
| `temporal` | Temporal evidence | Evidence candidates và `temporal_matches` cho contract QA hiện hữu | 200 |
| `trake` | Temporal Retrieval and Alignment of Key Events | Ranked `hypotheses`; mỗi item là một same-video sequence có đúng N original `frame_index` | 100 |
| `qa` | Grounded QA | `evidence`, eligibility/preflight, `answer` và citations/trace | 5 evidence |
| `auto` | Auto router | `requested_task="auto"` và task KIS/AVS/QA đã resolve | Theo task |

KIST là tên bài toán/tài liệu; identifier mà code thực sự nhận là `kis`.
`SUPPORTED_ONLINE_TASKS` không có alias `kist`. `auto` không tự route sang TRAKE,
vì TRAKE phải giữ chính xác event count/order do caller cung cấp.

## Chạy CLI

```powershell
.\.venv\Scripts\python.exe backend\app\services\ingestion\run_caption.py --help
.\.venv\Scripts\python.exe backend\app\services\ingestion\run_ocr.py --help
.\.venv\Scripts\python.exe backend\app\services\ingestion\run_object_detection.py --help

$env:RETRIEVAL_QUERY_EXPANSION_ENABLED = "false" # optional low-latency smoke
.\.venv\Scripts\python.exe -m backend.app.pipelines.online_pipeline `
  --task kis --top-k 20 --debug `
  --query "người đàn ông mặc áo đỏ đang mở cửa xe" `
  --output data/reports/kis_query.json

.\.venv\Scripts\python.exe -m backend.app.pipelines.online_pipeline `
  --task trake --top-k 100 `
  --query "Context: a runner. Events: 1. first touches the bar 2. reaches peak height" `
  --output data/reports/trake_query.json

$env:QA_ANSWER_MODE = "off"
.\.venv\Scripts\python.exe -m backend.app.pipelines.online_pipeline `
  --task qa --top-k 5 `
  --query "Người phụ nữ mặc áo đỏ đang cầm vật gì?" `
  --output data/reports/qa_query.json
```

Lệnh KIS trên là smoke command của chính production entrypoint:
`search_online -> get_online_pipeline -> OnlinePipeline.run`. Full coarse-to-dense
chỉ được xác nhận khi output có
`routing_trace.coarse_to_dense.executed=true` và `mode="coarse_to_dense"`.
`mode="selected_only_fallback"` nghĩa là request vẫn chạy selected-keyframe path
nhưng dense rescue/CSES chưa chạy; đừng lấy kết quả đó làm bằng chứng full path.

`QA_ANSWER_MODE=off` vẫn chạy QA evidence nhưng không load answer model;
`optional` giữ evidence và trả error/abstention status nếu generation lỗi;
`required` nâng answer/model failure thành lỗi. Thiếu evidence hoặc strict temporal
chain luôn trả `insufficient_evidence` thay vì dùng kiến thức ngoài video.

Caption mặc định dùng `florence-community/Florence-2-base-ft` (~0.23B tham số), task token
`<MORE_DETAILED_CAPTION>` và revision bất biến
`0b03b6f15a4a211370fb204aee4e7dd48887ea37`, cache dưới
`data/model_cache/caption`. CPU luôn dùng float32; CUDA hỗ trợ float16, bfloat16
và auto. Checkpoint này dùng Florence-2 native trong Transformers, không cần
remote code. Backend chưa kiểm thử quantization 4/8-bit nên từ chối các chế độ này.

```powershell
# CPU
.\.venv\Scripts\python.exe backend\app\services\ingestion\run_caption.py `
  --metadata-path data\metadata\keyframes_video7155.jsonl --device cpu --dtype float32

# CUDA
.\.venv\Scripts\python.exe backend\app\services\ingestion\run_caption.py `
  --metadata-path data\metadata\keyframes_video7155.jsonl --device cuda --dtype auto

# Chuẩn bị cache trên máy có mạng; sao chép nguyên cache và đặt HF_HUB_OFFLINE=1 ở máy đích.
hf download florence-community/Florence-2-base-ft `
  --revision 0b03b6f15a4a211370fb204aee4e7dd48887ea37 `
  --cache-dir data/model_cache/caption
```

Florence-2 không phải chat model và không được ép sinh JSON. Adapter ghi caption
text vào contract cũ với `structured_caption: null`. Qwen3.5 của grounded QA và
query expansion không bị thay đổi.

Gọi trực tiếp từ Python:

```python
from backend.app.services.retrieval.retrieval_manager import search_online, search_trake

kis = search_online(
    query="người đàn ông mặc áo đỏ đang mở cửa xe",
    task="kis",
    top_k=20,
)
trake = search_online(
    query="Context: a runner. Events: 1. first touches the bar 2. reaches peak height",
    task="trake",
    top_k=100,
)
trake_core = search_trake(
    "Context: a runner. Events: 1. first touches the bar 2. reaches peak height",
    top_k=100,
)
qa = search_online(
    query="Người phụ nữ mặc áo đỏ đang cầm vật gì?",
    task="qa",
    top_k=5,
)
```

KIS response có `query_plan` và ranked `candidates`; `keyframe_id`/`frame_id` là
technical retrieval identity, còn `frame_index` là canonical original-frame
lineage khi metadata có sẵn. Mỗi dense-path candidate giữ `score_breakdown`,
`modality_scores`, `cses_selection` và các identity/metadata field hiện hữu;
`routing_trace.latency` tách thời gian planning, expansion, selected visual, text,
fusion, dense global/rescue, CSES, deterministic rerank và context attachment.
QA response giữ evidence riêng trong `evidence` và
luôn có structured `answer` status; ở mode `off`, status cho biết answerer bị tắt,
không có câu trả lời được bịa từ ngoài evidence.

TRAKE response có `event_plan` và ranked `hypotheses`; mỗi hypothesis giữ
`video_id`, đúng N original `frame_ids`, score breakdown, events, lineage và
warnings. Alias `candidates`, nếu xuất hiện, vẫn chứa complete sequence và không
flatten event. Online wrapper bổ sung `requested_task="trake"`; core sequence
schema còn lại được giữ nguyên. `auto` không tự chọn TRAKE; truyền task rõ ràng.

Khi router được mount, các request chính xác là:

| Task | Route/body |
|---|---|
| KIS/KIST | `POST /retrieval/online` với `{"query":"...","task":"kis","top_k":20,"expanded_queries":[]}` |
| TRAKE | `POST /retrieval/online` với `{"query":"...","task":"trake","top_k":100,"expanded_queries":[]}` hoặc `POST /retrieval/trake` với `{"query":"...","top_k":100}` |
| QA | `POST /retrieval/qa` với `{"query":"...","top_k":5,"task_mode":"qa","expanded_queries":[]}` |

`POST /search` dùng `mode="kis"|"trake"|"qa"` thay cho field `task`. Router bọc
kết quả thành `{"success":true,"data":...,"message":null}`. Repository hiện
chưa có FastAPI app factory/uvicorn entrypoint, nên đây là service/router contract,
không phải URL đang lắng nghe trên một host/port mặc định.

Xuất CSV KIS, QA hoặc TRAKE:

```powershell
python -m backend.app.services.submission.export_query --task kis `
  --query "người mặc áo đỏ cầm điện thoại" --top-k 100 `
  --output data/submissions/kis_result.csv

python -m backend.app.services.submission.export_query --task trake `
  --query "Context: a jump. Events: 1. first leaves ground 2. reaches peak" `
  --top-k 100 --output data/submissions/trake_result.csv
```

TRAKE CSV dùng assumption `video_id,frame_id_1,...,frame_id_N` vì repository chưa
có official sample submission. Các giá trị `frame_id_N` là original zero-based
`frame_index`, không phải internal `RetrievalResult.frame_id`, timestamp, filename
hay FAISS row. Serializer yêu cầu lineage khớp và dedupe theo toàn sequence.

## KIS/KIST và QA config

KIS/KIST dùng canonical task `kis`; AVS dùng cùng implementation với profile
`avs`. Public call graph duy nhất là:

```text
search_online
  -> get_online_pipeline        # cache theo committed corpus generation
  -> OnlinePipeline.run
  -> query plan/optional expansion
  -> selected visual + caption/OCR/object retrieval
  -> intra-variant fusion + inter-modality weighted RRF
  -> coarse clip aggregation
  -> full dense-candidate FAISS global search + missed-clip rescue
  -> per-clip CSES
  -> deterministic evidence rerank + exact candidate dedup
  -> existing Candidate/public response schema
```

`configs/retrieval.yaml` sections `hybrid`, `weights`, `text_index`,
`query_expansion` và `online` điều khiển route này. Defaults quan trọng của
`online` là `coarse_top_n: 50`, `dense_global_top_k: 300`,
`dense_rescue_clips: 10`, `max_total_clips: 60`,
`dense_frames_per_clip: 12`, `rrf_k: 60`, `cses_enabled: true` và
`deterministic_rerank_enabled: true`. Rerank cuối là phép cộng có trọng số của
coarse RRF, dense visual, caption, OCR, objects, CSES gain, temporal consistency
và modality alignment. Hai cờ `learned_rerank_enabled` và `vlm_rerank_enabled`
mặc định `false`; canonical implementation hiện không load cross-encoder/VLM
retrieval reranker, kể cả trong request bình thường. Query-expansion Qwen là một
feature riêng của planner, không phải reranker này.

Mọi field `online` có env override prefix `RETRIEVAL_ONLINE_`; các switch chính
là `RETRIEVAL_ONLINE_COARSE_TO_DENSE_ENABLED`,
`RETRIEVAL_ONLINE_DENSE_ENABLED`,
`RETRIEVAL_ONLINE_DENSE_MISSING_BEHAVIOR`,
`RETRIEVAL_ONLINE_LEARNED_RERANK_ENABLED`,
`RETRIEVAL_ONLINE_VLM_RERANK_ENABLED` và `RETRIEVAL_ONLINE_DEBUG_ENABLED`.
CLI `--debug` override debug cho riêng request và mở detailed fusion trace.

Selected/coarse artifact paths vẫn lấy từ `RETRIEVAL_INDEX_PATH`,
`RETRIEVAL_FRAME_MAP_PATH`, `RETRIEVAL_MANIFEST_PATH` và
`RETRIEVAL_TEXT_INDEX_PATH`. Full dense-candidate bundle dùng riêng:

| Biến | Artifact |
|---|---|
| `RETRIEVAL_DENSE_INDEX_PATH` | Corpus-wide dense-candidate FAISS `IndexFlatIP` |
| `RETRIEVAL_DENSE_METADATA_PATH` | Enriched dense-candidate JSONL, cùng row order với FAISS |
| `RETRIEVAL_DENSE_FRAME_MAP_PATH` | Dense FAISS row-to-frame map |
| `RETRIEVAL_DENSE_MANIFEST_PATH` | Dense encoder/normalization/source/checksum contract |
| `RETRIEVAL_DENSE_REPORT_PATH` | Dense build/validation report |

Dense loader chạy lazy ở request KIS/AVS đầu tiên và cache theo corpus generation.
Với `online.dense_missing_behavior: fallback_sparse` (mặc định), **bundle bị
thiếu** (kể cả committed corpus cũ chưa khai báo dense roles) quay về
selected-only route và trace ghi `selected_only_fallback`;
`error` nâng thiếu artifact thành lỗi. Artifact đã có nhưng sai checksum, count,
dimension, row order, source lineage, khai báo bundle dở dang hoặc lệch encoder
contract với selected index luôn fail closed ở cả hai mode. Giới hạn public vẫn
là `hybrid.max_top_k: 200`.

QA dùng cùng canonical corpus nhưng có feature flags riêng:

| Biến | Mặc định | Hành vi |
|---|---|---|
| `QA_BGE_DENSE_ENABLED` | `false` | Bật BGE-M3 dense evidence retrieval |
| `QA_BGE_INDEX_ROOT` | `data/indexes/bge_m3` | Root chứa BGE FAISS/map/manifest |
| `QA_BGE_RERANKER_ENABLED` | `false` | Bật BGE cross-encoder candidate reranker |
| `QA_ANSWER_MODE` | `off` | `off`, `optional` hoặc `required` |
| `QA_MODELS_LOCAL_ONLY` | `false` | Chỉ dùng checkpoint đã có trong cache khi bật |

QA BGE flags không bật TRAKE BGE và ngược lại. Retrieval CLI entrypoint tự load
`.env`; biến trong process manager hoặc shell chạy CLI có độ ưu tiên cao hơn.

## TRAKE config và refinement

`configs/retrieval.yaml:trake` kiểm soát retrieval width, video weights, beam/DP,
soft gap penalty, bounded local window, diversity, `max_answers <= 100` và cutoffs
`[1,5,20,50,100]`. Có thể override bằng các biến `RETRIEVAL_TRAKE_*` tương ứng;
`RETRIEVAL_TRAKE_VIDEO_ROOT` mặc định là `data/raw/video`.

TRAKE BGE defaults:

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

Mỗi field có override cùng tên với prefix `RETRIEVAL_TRAKE_`, cụ thể:
`RETRIEVAL_TRAKE_BGE_DENSE_ENABLED`, `RETRIEVAL_TRAKE_BGE_DENSE_TOP_K`,
`RETRIEVAL_TRAKE_BGE_RERANKER_ENABLED`,
`RETRIEVAL_TRAKE_BGE_RERANKER_TOP_K`,
`RETRIEVAL_TRAKE_RETRIEVAL_FUSION`, `RETRIEVAL_TRAKE_RRF_K`,
`RETRIEVAL_TRAKE_HYBRID_RRF_WEIGHT`, `RETRIEVAL_TRAKE_BGE_RRF_WEIGHT` và
`RETRIEVAL_TRAKE_BGE_REQUIRED`.

Đó là policy/pool settings. Path/model/execution settings là env-only:

| Biến | Mặc định |
|---|---|
| `RETRIEVAL_TRAKE_BGE_INDEX_ROOT` | `data/indexes/bge_m3` |
| `RETRIEVAL_TRAKE_BGE_MODEL_NAME` / `RETRIEVAL_TRAKE_BGE_MODEL_REVISION` | `BAAI/bge-m3` / không ép revision |
| `RETRIEVAL_TRAKE_BGE_BATCH_SIZE` | `16` |
| `RETRIEVAL_TRAKE_BGE_RERANKER_MODEL` / `RETRIEVAL_TRAKE_BGE_RERANKER_REVISION` | `BAAI/bge-reranker-v2-m3` / `main` (optional/dev; required cần commit hash) |
| `RETRIEVAL_TRAKE_BGE_RERANKER_ALPHA` / `RETRIEVAL_TRAKE_BGE_RERANKER_BATCH_SIZE` | `0.5` / `16` |
| `RETRIEVAL_TRAKE_BGE_DEVICE` | `auto` |
| `RETRIEVAL_TRAKE_BGE_MODEL_CACHE_DIR` | `data/model_cache/bge_m3` |
| `RETRIEVAL_TRAKE_BGE_LOCAL_FILES_ONLY` | `false` |

Các flag này độc lập với `QA_BGE_DENSE_ENABLED` và
`QA_BGE_RERANKER_ENABLED`. Mỗi TRAKE event có canonical hybrid ranked list riêng;
khi bật dense BGE, pipeline tạo thêm BGE list cho cùng event và fusion bằng RRF
trước diversity, video gating và alignment. `rrf_k`, `hybrid_rrf_weight` và
`bge_rrf_weight` chỉ điều khiển rank fusion, không trộn raw score khác scale.
Optional BGE reranker chỉ xử lý bounded head `bge_reranker_top_k` mỗi event; tail
vẫn được giữ để bảo vệ recall. Context prior vẫn hybrid-only.

Với `bge_required: false`, lỗi init/artifact/search/rerank của nhánh BGE fail-open
về canonical hybrid và được ghi warning/trace. Với `true`, lỗi được nâng lên để
benchmark không chạy thiếu nhánh đã khai báo là bắt buộc. Policy này không thay
đổi QA BGE runtime. Per-event `sources`/`fusion`/`reranker` và aggregate `bge`
nằm dưới `trace.event_retrieval`; `trace.bge_contract` ghi corpus generation,
resolved dense revision/checksum, requested reranker revision và cờ
`revision_pinned`.
Trace không chứa local path, exception text hoặc query text.

TRAKE được khởi tạo lazy theo task nên `kis`/`qa` không load TRAKE index/model;
QA/temporal cũng chỉ khởi tạo dependency riêng khi route đó được chọn. QA và TRAKE
dùng chung BGE-M3 engine nếu toàn bộ artifact/model/revision/device/cache
contract giống nhau, nhưng cờ bật/tắt và fail policy của hai task vẫn độc lập.
`bge_required: true` là request-time fail-closed (API trả HTTP 503), không phải
startup readiness vì model được lazy-load. Hãy warm-up bằng một TRAKE query và
kiểm tra `trace.bge_contract` trước benchmark.

Mọi BGE dense runtime yêu cầu canonical offline corpus manifest đã publish và
khớp checksum. Thiếu manifest làm optional TRAKE fallback về hybrid; chế độ
`bge_required: true` luôn fail closed, kể cả khi chỉ bật reranker, để giữ corpus
lineage cho benchmark.

Optional/fail-open trên máy có cache:

```powershell
$env:RETRIEVAL_TRAKE_BGE_DENSE_ENABLED = "true"
$env:RETRIEVAL_TRAKE_BGE_RERANKER_ENABLED = "true"
$env:RETRIEVAL_TRAKE_BGE_REQUIRED = "false"
$env:RETRIEVAL_TRAKE_BGE_INDEX_ROOT = "data/indexes/bge_m3"
$env:RETRIEVAL_TRAKE_BGE_MODEL_CACHE_DIR = "data/model_cache/bge_m3"
$env:RETRIEVAL_TRAKE_BGE_LOCAL_FILES_ONLY = "true"
python -m backend.app.pipelines.online_pipeline `
  --task trake --top-k 100 `
  --query "Context: a jump. Events: 1. first leaves ground 2. reaches peak"
```

Required benchmark dùng cùng artifact/cache nhưng đổi policy; pin revision thay
cho `main` trước benchmark chính thức:

```powershell
$env:RETRIEVAL_TRAKE_BGE_DENSE_ENABLED = "true"
$env:RETRIEVAL_TRAKE_BGE_RERANKER_ENABLED = "true"
$env:RETRIEVAL_TRAKE_BGE_REQUIRED = "true"
$env:RETRIEVAL_TRAKE_BGE_INDEX_ROOT = "data/indexes/bge_m3"
$env:RETRIEVAL_TRAKE_BGE_RERANKER_REVISION = "<exact-40-hex-commit-from-cache>"
$env:RETRIEVAL_TRAKE_BGE_DEVICE = "cuda"
$env:RETRIEVAL_TRAKE_BGE_MODEL_CACHE_DIR = "data/model_cache/bge_m3"
$env:RETRIEVAL_TRAKE_BGE_LOCAL_FILES_ONLY = "true"
python -m backend.app.pipelines.online_pipeline `
  --task trake --top-k 100 `
  --query "Context: a jump. Events: 1. first leaves ground 2. reaches peak" `
  --output data/reports/trake_bge_required.json
```

Dense revision để trống để dùng immutable resolved revision trong BGE manifest;
chỉ override bằng đúng revision đó. Reranker `main` chỉ dành cho optional/dev;
`bge_required: true` yêu cầu hub model id và commit hash 40–64 ký tự hex.
Dense manifest trong required mode cũng phải chứa hub model id và resolved commit
hash bất biến; `main`, URL hoặc local model path đều bị từ chối.

Khi `refinement_enabled` bật, production `get_trake_pipeline()` inject
`Siglip2LocalFrameScorer` và reuse encoder/model đã nằm trong canonical visual
search engine. OpenCV chỉ decode bounded raw-frame windows dưới
`RETRIEVAL_TRAKE_VIDEO_ROOT`; scorer encode event text và RGB frames trong cùng
SigLIP2 vector space rồi đưa cosine similarity chuẩn hóa `[0,1]` cho
boundary-aware selection. Nhánh này không load thêm checkpoint, không search
full dense-candidate FAISS và không ảnh hưởng QA route. Nếu video/decode/scoring
không khả dụng, refiner giữ coarse canonical frame và ghi warning/fallback trace.

Sự thật cần giữ tỉnh táo: shared-SigLIP2 scorer là semantic similarity rẻ, không
phải pose/contact detector hay VLM verifier. Unit/integration contract có thể xác
nhận wiring và fallback, nhưng semantic-boundary accuracy vẫn cần full-corpus
run cùng ground truth/adjudication trước khi đưa claim chất lượng.

## Test

```powershell
python -m pytest -q
.\.venv\Scripts\python.exe -m unittest discover -s backend\tests -v
.\.venv\Scripts\python.exe -m unittest -v backend.tests.test_trake_query_parser backend.tests.test_trake_pipeline backend.tests.test_trake_submission backend.tests.test_trake_metrics
```
