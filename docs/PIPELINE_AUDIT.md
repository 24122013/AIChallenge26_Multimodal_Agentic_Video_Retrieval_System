# Pipeline Audit: KIS, QA và competition runtime

Ngày audit: 2026-08-15. Phạm vi: source, config, dependency, CLI, tests và tài liệu
đang được track trong repository. Audit không coi file cache/dataset bị ignore là
source of truth và không tuyên bố model-level E2E khi chưa chạy checkpoint thật.

## Kết luận điều hành

Kiến trúc chạy thật là visual-only và nhất quán ở các modality: SigLIP2,
caption, OCR, objects, temporal metadata, BM25/BGE. Không có ASR runtime, audio
preprocessing, transcript index hay ASR weight. `backend/` là implementation
canonical; `competition/` không phải bản fork độc lập mà là contract/orchestration
layer tái sử dụng phần lớn backend, cộng dense-candidate rescue, VKIS refinement,
run lineage và submission validation.

Readiness hiện tại:

| Mục tiêu | Đánh giá | Lý do |
|---|---|---|
| Competition development | Khá, có điều kiện | Contract/resume/schema/ranking được test; cần benchmark thật và pin model revisions |
| Backend library/CLI | Dùng được | Indexing/retrieval/QA callable; artifact contract rõ |
| Online web service | Chưa sẵn sàng | Có router nhưng không có `FastAPI()` app, router mounting hoặc health endpoint |
| Production | Chưa sẵn sàng | Chưa E2E thật, chưa observability/service lifecycle, vài revision chưa pin |

Trong môi trường audit, unit/integration suite có thể chạy nhưng FFmpeg/ffprobe và
Paddle không hiện diện, nên không thể chạy extraction/OCR/E2E checkpoint thật.

| Hạng mục | Trạng thái | Bằng chứng chính |
|---|---|---|
| Offline pipeline | Hoạt động nhưng còn rủi ro | Phase 3/orchestrator và 116 competition tests; chưa model E2E thật |
| KIS backend | Hoạt động nhưng còn rủi ro | `HybridSearchEngine`, advanced retrieval, query-plan tests; cần artifacts thật |
| QA backend | Chỉ hoàn chỉnh một phần | Evidence/answer/abstention đã test; answerer mặc định off, chưa service/E2E thật |
| Competition runtime | Hoạt động nhưng còn rủi ro | 10-stage runner, exact public contract, dry-run/schema/resume tests |
| Dependency/setup | Hoạt động nhưng còn rủi ro | Một manifest + ba install profile; chưa clean-install trên cả ba profile |
| Testing/observability | Chỉ hoàn chỉnh một phần | 244 backend + 116 competition tests; thiếu real-model/metrics/tracing |
| Production readiness | Chưa hoạt động end-to-end | Thiếu web app/health/lifecycle và release benchmark |
| Competition readiness | Hoạt động nhưng còn rủi ro | Cần pin BGE/VLM revisions và full 250-video run trên target GPU |

## 1. Bản đồ implementation canonical

### Offline indexing

| Stage | Source/function chính | Input | Output/contract |
|---|---|---|---|
| Video discovery/shot-aware extraction | `backend/app/services/indexing/extract_keyframes.py::extract_keyframes_for_video` | MP4, strategy/config | JPEG keyframes, `keyframes_<video>.jsonl`, extract report |
| Dense candidate pool | `keyframe_candidates.py`, `materialize_keyframe_candidates.py` | Video + shot boundaries | Stable candidate IDs, timestamps, anchors/endpoints |
| SigLIP2 features | `build_siglip2_index.py::encode_keyframes` | Candidate/keyframe JSONL + images | normalized embeddings, embedding metadata, skip log, benchmark |
| Caption | `ingestion/caption_pipeline.py`, `run_caption.py` | Frame metadata/images | Structured English caption JSONL; lazy Qwen load |
| OCR | `ingestion/ocr_pipeline.py`, `run_ocr.py` | Frame metadata/images | Vietnamese/English OCR JSONL |
| Objects | `ingestion/object_pipeline.py`, `run_object_detection.py` | Frame metadata/images | Prompt-dependent YOLOE labels/boxes JSONL |
| Feature join + selection | `keyframe_multimodal_pipeline.py::run_multimodal_keyframe_pipeline` | candidates + four feature streams | Canonical selected records, ledgers, hard-guarantee report |
| Selection algorithm | `keyframe_selection.py::select_keyframes` | importance/novelty/protected events | Shot/endpoint protection, temporal repair, dedup, MMR fill |
| Coarse visual index | `build_faiss_index.py::build_faiss_index` | SigLIP2 `.npy` + matching metadata | FAISS, compact frame map, encoder/index manifest, validation report |
| Segments/neighbors | backend wrappers + `src/indexing` | Canonical metadata | ordered neighbor and boundary-aware segment metadata |
| Sparse text index | `build_text_index.py` | caption/OCR/object or segments | versioned BM25-style JSON index |
| Dense text index | `build_bge_m3_index.py` | existing visual metadata only | normalized 1024-d BGE-M3 FAISS, map, manifest |

Hard guarantees đáng chú ý:

- Candidate identity, order, video ID, feature status và embedding alignment được
  validate trước selection/publish.
- Protected shot/endpoint events được chọn trước MMR; temporal gaps được repair;
  dedup có override audit khi hard constraints đòi giữ frame.
- Competition Phase 3 mặc định fail closed nếu caption/OCR/object không tiến triển
  hoặc có hard artifact failure; `--allow-partial-features` là override rõ ràng.
- Manifest/lineage fingerprint kiểm tra model, revision, config và hashes khi
  resume; artifact stale không được coi là current.

### Online KIS/AVS retrieval

```text
raw query
  -> query_plan.build_query_plan (typed mode, roles, constraints, temporal events)
  -> visual SigLIP2 FAISS
  -> caption/OCR/object sparse retrieval
  -> optional BGE-M3 dense retrieval
  -> merge: weighted scores (simple backend) hoặc weighted RRF (advanced)
  -> constraint/modality-aware deterministic rerank
  -> deduplicated ranked frame/segment evidence
```

Implementation:

- `hybrid_search.py::HybridSearchEngine` là đường đơn giản: visual + text
  channels, candidate merge và `HybridReranker`. Weights hiện tại trong
  `configs/retrieval.yaml` là visual 0.55, caption 0.20, OCR 0.10, objects 0.10,
  temporal 0.05; tổng bằng 1 và không có ASR.
- `advanced_search.py::advanced_text_search` và `advanced_vector_search` dùng
  weighted RRF để tránh so trực tiếp raw scores khác thang đo, coarse clip pool,
  dense rescue, CSES và deterministic rerank. Optional VLM chỉ rerank Top-M.
- Query parser ở `query_plan.py::build_query_plan` hỗ trợ tiếng Việt/Anh, KIS,
  AVS, QA và temporal; tối đa 5 event. Explicit KIS/AVS không tự biến thành
  temporal chain. Query expansion plan được gắn vào plan/trace.

### Online QA

```text
question
  -> plan_qa_question / typed QueryPlan
  -> shared hybrid/temporal retrieval
  -> optional BGE dense + cross-encoder
  -> evidence allowlist: image identity + caption + OCR + objects
  -> Qwen grounded JSON answer (optional/required) hoặc abstain
```

- `qa_evidence.py::QaEvidenceSearchEngine` điều phối query branches, constraint
  scores, temporal route, RRF và evidence construction.
- Non-temporal QA giới hạn Top-3 evidence. Temporal QA giữ strict ordered chain
  tối đa 5 event; `relaxed_gap`/`sparse_compat` chỉ có giá trị audit và không được
  đưa cho answerer như evidence đủ.
- `qa_answerer.py::_sanitize_evidence` dùng allowlist; transcript/ASR-shaped keys
  không được chuyển vào prompt. `build_grounded_prompt` yêu cầu JSON, citation ID
  hợp lệ và cấm outside knowledge.
- Empty/insufficient evidence dẫn tới abstain. Mode `required` bọc failure bằng
  `RequiredQaPipelineError`, giữ evidence/report để chẩn đoán; router dự kiến trả
  503. Có timeout và answer cache, chưa có retry.
- Model, index và runner được lazy-load/cached qua retrieval manager và các lazy
  runner, tránh reload mỗi query trong cùng process.

QA/error-path matrix:

| Trường hợp | Hành vi đã xác minh từ code/test |
|---|---|
| OCR rỗng hoặc caption thiếu ở một evidence | Backend QA vẫn dùng modality còn lại; field-tag document không tạo ASR fallback |
| Hard feature lỗi trong competition Phase 3 | Fail closed; chỉ tiếp tục khi operator dùng `--allow-partial-features` |
| Không có GPU | `auto` có thể về CPU; quantization/GPU-only model path có thể fail theo mode optional/required |
| Model answer không load/inference được | `optional` giữ evidence và abstain/fallback; `required` nâng lỗi có report/evidence |
| Index chưa tạo hoặc manifest không khớp | Loader fail với `FileNotFoundError`/validation error; router dự kiến 503 khi được mount |
| Query rỗng/không có token | `build_query_plan` fail fast bằng `ValueError` |
| Không có candidate/evidence đủ | Không gọi Qwen; trả `insufficient_evidence` |
| Artifact cũ sai dimension/normalization/identity | FAISS/BGE/Phase 3 contract validator từ chối trước search/publish |
| FPS/timestamp không hợp lệ | Input/metadata validators từ chối; temporal path không tự suy đoán FPS toàn corpus |

## 2. Competition runtime và divergence

`competition/run_retrieval_v2.py` chạy 10 subprocess stage để giải phóng model/GPU
memory giữa các stage và hỗ trợ resume:

```text
validate-input -> keyframes -> index -> neighbors -> segments -> text-index
-> bge-text-index -> dense-index -> predict -> validate-submission
```

Contract trong `competition/pipeline.py`:

- `corpus.csv`: đúng 250 video và exact header.
- `questions.csv`: đúng 100 query, 50 TKIS + 50 VKIS.
- `sample_submission.csv`: 100 answer columns; submission giữ query order và 100
  unique valid answers/query.

| Chức năng | Backend canonical | Competition implementation | Mức độ đồng bộ | Divergence có chủ đích? | Rủi ro |
|---|---|---|---|---|---|
| Query planning | `query_plan.build_query_plan` | Import trực tiếp cho TKIS; VKIS dùng image route | Cao | Có, do task contract | Thấp |
| Retrieval | Hybrid + advanced service | Import advanced backend, thêm public routing | Cao | Có | Trung bình |
| Multimodal scoring | Weighted fusion/RRF | Cùng config/backend, thêm competition profile | Cao | Có | Trung bình do hai score path |
| Reranking | deterministic/BGE/optional VLM | Dùng backend, BGE required mặc định cho TKIS | Cao | Có | Thấp |
| KIS | KIS/AVS/temporal callable | 50 TKIS + 50 VKIS submission | Trung bình | Có, exact public contract | Trung bình |
| QA | Evidence + optional Qwen answer | Chỉ smoke/evaluation, không vào submission | Cao về code | Có, public contract không có QA | Thấp |
| Index/schema | SigLIP2/BGE manifest + frame map | Cùng builders, thêm run-root lineage/dense candidates | Cao | Có | Thấp |
| Model/checkpoint | Defaults trong backend modules | Import defaults; thêm optional SmolVLM2 | Cao | Có | Cao khi revision `main` |
| Config | YAML + environment | YAML + CLI flags/run manifest | Trung bình | Có | Trung bình nếu override không lock |
| API contract | Router có code nhưng chưa mount | Không có competition HTTP API | Không áp dụng | Có, terminal submission runtime | Thấp |
| Timestamp mapping | Per-video FPS/frame/timestamp metadata | Validate corpus FPS/frame_count + canonical map | Cao | Không | Thấp, được contract-test |
| Dependency | Root `requirements.txt` + Torch/Paddle profile | Dùng cùng environment | Cao | Không | Trung bình cho wheel/driver |
| Error handling | Exceptions/fail-closed QA modes | Stage reports, retries, resume, hard-feature checks | Trung bình | Có, batch runtime cần durability | Trung bình |
| Không ASR | Allowlist caption/OCR/objects | Cùng backend và không có ASR stage | Cao | Không | Thấp; historical reports dễ gây nhầm |

Một boundary chưa sạch: `backend/app/services/retrieval/advanced_search.py` import
dense-index type từ namespace `competition`. Điều này tạo reverse dependency từ
canonical backend sang competition và nên được chuyển về backend/shared module.

## 3. So với kiến trúc mong muốn

| Khối kiến trúc mong muốn | Trạng thái | Phạm vi | Gap |
|---|---|---|---|
| Shot detection | Implemented | Shared/backend | Cần model smoke thật |
| Dense sampling | Implemented | Shared/backend | Cần corpus benchmark |
| Visual/OCR/caption extraction | Implemented | Shared/backend | Thiếu target-GPU validation |
| Importance scoring + protected events | Implemented | Shared/backend | Rule/threshold cần quality eval |
| Coverage/MMR selection + dedup + repair | Implemented | Shared/backend | Đã contract-test, chưa recall benchmark |
| Canonical keyframe index | Implemented | Shared/backend | Cần release artifact validation |
| Dense-frame index | Competition only | Competition | Backend advanced path phụ thuộc namespace competition |
| Query understanding | Implemented | Shared/backend | Rule parser cần multilingual eval thật |
| Coarse retrieval | Implemented | Shared/backend | Cần score/recall calibration |
| Load dense candidates + CSES | Implemented | Competition + advanced backend | Boundary package chưa sạch |
| Multimodal reranking | Implemented | Shared/backend | Simple/advanced fusion khác nhau |
| KIS result | Implemented | Backend; submission adapter ở competition | Chưa service E2E |
| QA result/evidence | Partial | Backend only | Answerer off mặc định, chưa real-model E2E |
| Runnable API + health | Missing | Backend | Cần app factory, mount, lifecycle/health |
| Frontend | Placeholder | Frontend | Chưa có UI |
| Docker/Compose | Missing | Toàn repo | Không có container/deployment manifest |
| Production observability | Partial | Toàn repo | Cần metrics, structured logs, tracing, SLO |

## 4. ASR/audio cleanup audit

| Vị trí | Phân loại | Kết luận/hành động |
|---|---|---|
| Tracked runtime source/config/dependencies | Architecture đã loại | Không có faster-whisper/Whisper/ctranslate2, audio reader, transcript index hoặc ASR weight |
| `bge_dense.py`, `bge_reranker.py`, `qa_answerer.py`, `qa_pipeline.py` | Compatibility cần giữ | Chỉ là allowlist/defensive exclusion để legacy ASR keys không lọt vào evidence |
| Backend tests | Test guard cần giữ | Cố ý chứng minh transcript/ASR bị loại; không phải dormant feature |
| `reports/index_size_latency.*`, `reports/Experiment.md` | Legacy/historical harmless nhưng dễ gây hiểu nhầm | Markdown được gắn nhãn historical; JSON được contextualize tại audit này; không dùng làm kiến trúc hiện tại |
| Ignored `data/model_cache/asr` | Dead local artifact | Không được source gọi; có thể xóa thủ công nếu không cần, audit không xóa dữ liệu user |
| `.venv` có package ASR cũ | Local environment artifact | Không có trong `requirements.txt`; tạo venv sạch sẽ không cài chúng |

Không-audio không phải error path: pipeline không probe/extract audio và video
không có audio không gây lỗi riêng. Trade-off là recall giảm với câu hỏi chỉ xuất
hiện trong lời nói.

## 5. Dependency audit

`requirements.txt` là manifest Python duy nhất. Torch và Paddle cố ý tách khỏi
file này vì CPU/CUDA wheels dùng index khác nhau; README cung cấp ba profile CPU,
CUDA 11.8 và CUDA 12.6. Các file `requirements-core.txt`,
`requirements-gpu-cu118.txt`, `requirements-gpu-cu126.txt` đã bị loại và notebook
được cập nhật.

Dependency được giữ vì có import/runtime path trực tiếp: NumPy, OpenCV, Pillow,
PyYAML, FAISS, tqdm, open-clip-torch, Transformers, Accelerate, bitsandbytes,
TransNetV2, PaddleOCR, Ultralytics, FastAPI/Pydantic/Uvicorn và python-docx cho
report scripts. Không có ASR dependency.

Lưu ý: FastAPI/Uvicorn giữ router importable và chuẩn bị runtime, nhưng việc có
package không đồng nghĩa service đã có entrypoint.

## 6. Issue/risk register

| Mức độ | Phạm vi | Thành phần | Vấn đề | Bằng chứng | Ảnh hưởng | Cách khắc phục |
|---|---|---|---|---|---|---|
| Critical | Toàn repo | — | Không có issue Critical còn mở đã chứng minh | Full test/compile xanh | Không áp dụng; không đồng nghĩa E2E-certified | Giữ release gate real-model E2E |
| High | Backend | Web service | Router chưa có app/mount/health | `backend/app/api/*.py`; không có `FastAPI()` | Không deploy/curl được | Thêm app factory, lifecycle, health/readiness |
| High | Shared | Package boundary | Backend import competition dense index | `retrieval/advanced_search.py` import `competition.dense_index` | Canonical backend không độc lập | Chuyển contract/type về backend/shared |
| High | Validation | Real-model E2E | Máy audit thiếu FFmpeg/ffprobe và Paddle | Runtime probes; unit tests dùng fakes | Chưa biết driver/VRAM/latency/recall thật | Chạy target-GPU acceptance checklist |
| High | Models | Revision pinning | BGE/reranker/VLM mặc định `main` | runner/parser defaults | Run khác thời điểm có thể khác model | Pin commit hash trong release config |
| Medium | Retrieval | Score fusion | Simple path trộn raw scores khác thang | `rerank.py::HybridReranker`; advanced dùng RRF | Rank có thể lệch theo modality calibration | Chuẩn hóa/calibrate hoặc chuẩn hóa về RRF |
| Medium | QA | Resilience | Answer timeout nhưng không retry | `qa_answerer.py::_call_with_timeout` | Transient failure thành 503 ở required mode | Bounded retry/backoff, giữ idempotent cache key |
| Medium | API | Error exposure | Generic exception detail đưa vào HTTP | `api/retrieval.py`, `api/search.py` | Lộ internal detail, contract không ổn định | Stable error codes + server-side logs |
| Medium | Setup | Environment config | `.env` không tự load | Chỉ `os.getenv`, không dotenv loader | Người mới copy `.env` nhưng config không đổi | Export qua shell/process manager hoặc thêm loader rõ ràng |
| Medium | Testing | Integration coverage | Không có fixture chạy model + FFmpeg thật | Test suite inject fake runners/backends | Wheel/checkpoint regression phát hiện muộn | Thêm bounded GPU smoke CI/manual gate |
| Medium | Architecture | Model lifecycle | API chưa có concurrency/lifecycle policy | Không có app entrypoint | OOM/race risk khi triển khai nhiều worker | Single-owner model service + queue/limits |
| Low | Docs | Historical ASR | Report cũ vẫn chứa ASR-era schema/count | `reports/index_size_latency.*`, `Experiment.md` | Dễ hiểu nhầm | Banner historical + audit classification |
| Low | Structure | Placeholders | Nhiều module agent/evaluation/core/db rỗng | Tracked zero-byte Python files | Dễ đánh giá quá mức implementation | Xóa hoặc gắn roadmap/status rõ |
| Low | Runtime | Runner overlap | Runner cũ và v2 cùng tồn tại | `run_end_to_end.py`, `run_retrieval_v2.py` | Chọn nhầm path bỏ stage mới | Gắn compatibility label/deprecation plan |

### Critical

Không phát hiện lỗi Critical còn mở trong source đã test. Tuy nhiên không có đủ
điều kiện để cấp chứng nhận real-model E2E trong máy audit.

### High

1. **Online service chưa runnable**: không có `FastAPI()` app, router mounting,
   `/health` hoặc lifecycle load/unload. Tác động: không thể deploy/curl backend
   theo tài liệu service thông thường.
2. **Reverse namespace dependency**: backend advanced retrieval phụ thuộc
   competition dense-index type. Tác động: canonical package khó tái sử dụng độc
   lập và dễ circular/drift khi competition thay đổi.
3. **Chưa real-model E2E**: FFmpeg/ffprobe và Paddle thiếu trong môi trường audit;
   public dataset/checkpoint run không được thực hiện. Unit tests không đo recall,
   latency, VRAM hoặc driver compatibility.
4. **Unpinned model revisions**: BGE-M3, BGE reranker và optional VLM mặc định
   `main`. Artifact manifest ghi revision nhưng cùng command ở hai thời điểm có
   thể tải nội dung khác.

### Medium

1. `.env.example` chỉ là reference vì không có dotenv loader; operator phải export
   biến qua shell/process manager.
2. Simple `HybridReranker` fusion raw modality scores phụ thuộc calibration;
   advanced weighted RRF tránh phần lớn vấn đề này nhưng hai path có thể khác rank.
3. Grounded QA có timeout/cache nhưng không retry/backoff; transient generation
   failure ở `required` trở thành 503 khi router được mount.
4. API router chuyển generic exception message ra HTTP detail; production nên
   trả error code ổn định và giữ internal detail trong structured logs.
5. `docs/architecture.md` mô tả nhiều module placeholder như một hệ thống hoàn
   chỉnh; phải đọc cùng audit này để phân biệt implemented và aspirational.
6. Không có integration fixture nhỏ chạy FFmpeg + model thật; regression về
   driver/wheel/checkpoint chỉ xuất hiện ở deployment.

### Low

1. Historical report JSON không thể chứa comment và vẫn có ASR-era fields; banner
   nằm ở Markdown/audit thay vì sửa bằng chứng benchmark cũ.
2. Nhiều file rỗng dưới `agent/`, `evaluation/`, `core/`, `db/` làm tăng cảm giác
   coverage giả dù không nằm trên call path chính.
3. Runner cũ và v2 cùng tồn tại; người dùng có thể chọn nhầm runner compatibility
   và bỏ qua BGE/dense/submission stages mới.

## 7. Lỗi đã sửa trong audit

- Khôi phục query expansion plan, typed task-mode behavior và modality
  decomposition trong `query_plan.py` sau merge regression.
- Khôi phục fail-closed caption/OCR/object progress checks và release Paddle/model
  memory trong competition Phase 3.
- Xóa khai báo/forwarding CLI trùng của caption batch/quantization trong runner v2.
- Hợp nhất dependency manifest, cập nhật notebook/report commands, sửa model table
  stale và viết lại hướng dẫn chạy theo implementation thật.

## 8. Tiêu chí chấp nhận trước production/competition release

1. Pin commit revision cho BGE/reranker/VLM và lưu lock cùng run manifest.
2. Chạy `validate-input`, một video Phase 3 thật, rồi full 250-video runner trên
   đúng GPU/driver; lưu peak VRAM, latency, failure/retry report và artifact hashes.
3. Chạy task smoke KIS/AVS/QA trên artifacts vừa tạo; kiểm tra citations và
   abstention bằng sampled manual review.
4. Chạy full backend + competition tests và submission validator ở cùng commit.
5. Nếu cần deploy online, bổ sung app factory, health/readiness, router mount,
   request limits, structured error/logging và concurrency policy trước khi mở API.
