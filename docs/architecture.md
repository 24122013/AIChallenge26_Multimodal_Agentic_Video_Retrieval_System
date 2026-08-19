# Architecture

# 1. System Overview

Mục tiêu của hệ thống là hỗ trợ truy xuất video đa phương thức (Multimodal Video Retrieval)

Hệ thống phải hỗ trợ:

* Text-based Retrieval
* Visual Retrieval
* OCR Retrieval
* Object Retrieval
* Temporal Retrieval
* TRAKE ordered-sequence retrieval
* Deterministic query planning và optional query expansion

Người dùng nhập truy vấn tự nhiên.

Hệ thống sẽ:

1. Phân tích truy vấn.
2. Chọn chiến lược tìm kiếm phù hợp.
3. Truy xuất dữ liệu từ nhiều nguồn.
4. Hợp nhất kết quả.
5. Trả về ranked candidates hoặc ranked TRAKE sequence hypotheses.

---

# 2. High-Level Architecture

```text
CLI / service caller / future HTTP host
    │
    ▼
`search_online` hoặc router contract
    │
    ▼
OnlinePipeline + typed QueryPlan
    │
    ▼
Retrieval Layer
    │
    ├── Visual Search
    ├── Caption Search
    ├── OCR Search
    ├── Object Search
    ├── Temporal Evidence Search
    └── TRAKE Sequence Pipeline
    │
    ▼
Index Layer
    │
    ├── Selected-keyframe visual FAISS (coarse)
    ├── Full dense-candidate visual FAISS (global rescue + CSES)
    ├── Sparse/optional dense text indexes
    ├── Metadata Store
    └── Canonical neighbor/segment indexes
    │
    ▼
Dataset
```

Đây là kiến trúc runtime đã triển khai. Repository hiện chỉ định nghĩa FastAPI
routers, chưa có `FastAPI()` app factory/mount, health endpoint hay frontend chạy
được; vì vậy sơ đồ không ngầm hứa một HTTP server sẵn có.

---

# 3. Offline Pipeline

Offline Pipeline chịu trách nhiệm chuẩn bị dữ liệu.

```text
Raw Videos
    │
    ▼
Shot detection + dense-candidate materialization
    │
    ▼
SigLIP2 encoding over the complete dense pool
    │
    ├── Full dense visual layer
    │     └── SigLIP2 FAISS + visual-only metadata + frame map + manifest
    │
    └── Visual-temporal selection
          └── Canonical selected keyframes + subsetted SigLIP2 vectors
                │
                ▼
          Caption + OCR + Object inference on selected IDs only
                ├── Selected-keyframe visual FAISS
                ├── Selected Caption/OCR/Object BM25 + optional BGE
                └── Selected neighbor/segment metadata
```

Output:

* Full dense-candidate visual bundle
* Canonical selected multimodal keyframes
* Segments
* Metadata
* Embeddings
* Search Index

Offline selected keyframes là **technical keyframes**: sparse frames được chọn
để lập coarse index và luôn giữ original zero-based `frame_index`. Selector chỉ
dùng shot/boundary coverage, temporal max-gap, visual dedup, full-dense SigLIP2
transition/novelty và deterministic diversity. Caption, OCR và object detection
không phải selector input.

Offline đồng thời publish một **full dense-candidate visual index** từ toàn bộ
materialized candidates. Bundle này rộng hơn selected index nhưng vẫn không chứa
mọi raw video frame. Dense row có identity/timestamp/path/shot-segment lineage và
SigLIP2 mapping; nó không có `caption`, `ocr_text` hay `objects`. Selected SigLIP2
vectors được lấy trực tiếp từ full-dense matrix, không encode lại. TRAKE tìm
**semantic keyframes** thỏa criterion của từng event bằng cách decode local
windows quanh technical frame và không phụ thuộc vào dense candidate FAISS.

Với `N` dense candidates và `M` selected keyframes, invariant canonical là:

```text
dense_siglip_count == N
selected_caption_count == M
selected_ocr_count == M
selected_object_count == M
```

`N == selected_caption_count` không còn là invariant. Empty OCR/object result có
thể là inference thành công; record `status="error"` là trạng thái khác và không
được giả làm empty evidence.

Temporal density là configuration-driven. Profile deadline có thể dùng
`max_gap_seconds=4.0` và soft density `0.25`, nhưng one-per-shot/boundary coverage
vẫn là hard constraint; shot-heavy video có thể chọn nhiều hơn soft target.

---

# 4. Online Pipeline

Online Pipeline chạy khi người dùng gửi truy vấn.

```text
User Query + explicit task
    │
    ▼
search_online -> get_online_pipeline -> OnlinePipeline.run
    ├── KIS / AVS
    │     └── query plan + optional expansion
    │         -> selected visual/caption/OCR/object retrieval
    │         -> weighted RRF -> coarse clips
    │         -> full dense FAISS global rescue -> per-clip CSES
    │         -> bounded neighbor/segment context scoring
    │         -> deterministic evidence rerank -> exact dedup -> Top-K
    │         -> response context attachment
    ├── temporal / QA
    │     └── existing evidence-oriented routes
    └── TRAKE
          └── event parser -> per-event retrieval -> video gating
              -> K-best frame alignment -> bounded shared-SigLIP2 refinement
              -> diverse ranked sequence hypotheses
    │
    ▼
Task-specific response
```

`temporal` is the existing QA/evidence-oriented task. `trake` is a separate
sequence-first task; it is selected explicitly and is not inferred by `auto`.
Both reuse canonical retrieval artifacts, but they keep different response
semantics.

The full dense-candidate loader is lazy and cached by committed corpus
generation, so QA and TRAKE are not blocked merely because the dense KIS/AVS
bundle is absent. KIS/AVS use `online.dense_missing_behavior` to choose explicit
`selected_only_fallback` or an error on missing files. A current dense manifest
must declare schema `2.0`, `layer="dense_visual"` and
`modalities=["siglip2"]`; dense rows containing Caption/OCR/Object are rejected
as an incompatible old architecture. Integrity/lineage/encoder contract failures
always fail closed. Canonical KIS/AVS has no heavy learned/VLM retrieval reranker;
legacy settings for those branches are ignored with a warning. The final rerank
is deterministic and exposes score breakdown, weighted contributions, context
evidence/cap diagnostics and per-stage latency.

Neighbor/segment scoring is an opt-in advanced path because loading the artifacts
is disabled by default. When requested and available, it runs after CSES and
before final dedup/Top-K. It resolves at most two neighbors on each side and a
12-keyframe canonical segment window, aggregates the segment Top-3, and caps the
combined context bonus at `0.08`. It reuses the normalized original-query vector,
full-dense visual vectors and selected neighbor/segment Caption/OCR/Object
evidence where available: no second encode and no additional global FAISS/BM25
search. It never expects semantic fields on an arbitrary dense row. After Top-K,
the same context index hydrates public `neighbors`/`segment_context`. Missing
optional files produce an explicit fallback trace; present but corrupt or
lineage-mismatched artifacts fail validation.

Offline checkpoints include architecture/schema and input lineage. Checkpoints
from the former full-dense multimodal architecture cannot mark selected semantic
inference complete. Compatible early artifacts may resume, but incompatible
selection/enrichment/corpus artifacts are regenerated or fail fast with an
actionable message. Corpus publication remains staged and atomic; the commit
manifest is written last.

---

# 5. Backend Architecture

Backend được chia thành 5 service chính.

## Ingestion

Nhiệm vụ:

* Caption Generation
* OCR Extraction
* Object Detection

Output:

Metadata chuẩn hóa.

---

## Indexing

Nhiệm vụ:

* Embedding Generation
* Vector Database
* Neighbor Frame Index

Output:

Searchable Index.

---

## Retrieval

Nhiệm vụ:

* Visual Retrieval
* Caption Retrieval
* OCR Retrieval
* Object Retrieval
* Hybrid Retrieval
* Temporal Retrieval
* TRAKE Event Retrieval / Video Gating / Alignment / Ranking

Output:

Candidate Results, or same-video ordered TRAKE hypotheses with explicit original
frame lineage.

---

## Query Planning

Nhiệm vụ:

* Typed task/profile planning
* Optional bounded query expansion
* Modality hints and query variants
* Deterministic retrieval/fusion orchestration

Output:

`QueryPlan` consumed directly by the canonical search pipeline.

---

## Evaluation

Nhiệm vụ:

* Benchmark
* Metrics
* Ablation
* Reporting

Output:

Performance Reports.

---

# 6. Future Frontend Architecture (chưa triển khai)

Khi được triển khai, frontend chỉ chịu trách nhiệm hiển thị.

```text
Search Page
    │
    ▼
Results Page
    │
    ▼
Timeline Viewer
    │
    ▼
Candidate Basket
    │
    ▼
Submission Workflow
```

Frontend không thực hiện:

* Inference
* Retrieval
* Metadata Processing

Frontend tương lai chỉ giao tiếp qua API sau khi có app/router mount thực tế.

---

# 7. Retrieval Architecture

Retrieval Layer gồm nhiều search engine độc lập.

```text
Query
 │
 ├── KIS / AVS
 │     ├── Selected-keyframe visual search
 │     ├── Caption / OCR / object search
 │     ├── Query-variant + weighted reciprocal-rank fusion
 │     ├── Coarse clip aggregation
 │     ├── Full dense-candidate global FAISS rescue
 │     ├── Per-clip CSES visual/temporal coverage selection
 │     ├── Bounded canonical neighbor/segment scoring
 │     ├── Deterministic evidence rerank + exact dedup + Top-K
 │     └── Bounded context payload attachment
 │
 ├── Temporal Evidence / QA
 │
 └── TRAKE (explicit task)
          │
          ├── Conservative deterministic event parser
          ├── Hybrid retrieval once per ordered event
          ├── Per-event rank normalization + shot diversity
          ├── Coverage-first candidate-video gating
          ├── Beam/DP K-best alignment on original frame_index
          ├── Bounded local refinement or canonical-frame fallback
          └── Sequence validation, NMS and top-100 diversity
           │
           ▼
      Candidate Pool
          │
          ▼
      Re-ranking
          │
          ▼
     Final Results
```

Mỗi search engine có thể phát triển độc lập, nhưng KIS/AVS production không gọi
`advanced_search` như một orphan utility: nó được route trực tiếp từ
`OnlinePipeline.run` qua canonical public manager entrypoint. Coarse RRF dùng
selected visual/Caption/OCR/Object evidence để chọn clip pool. Dense stage sau đó
dùng full-dense visual similarity, coarse selected evidence, CSES gain, temporal
consistency và bounded neighbor/segment evidence. Nó không đọc semantic evidence
trực tiếp từ arbitrary dense rows và không coi modality không tồn tại là zero
evidence để giữ nguyên một weight stale. Context lookup cho mỗi candidate bị chặn
bởi một cửa sổ nhỏ; nó không scan segment/corpus tùy ý và không gọi encoder/search
engine lần nữa.

TRAKE is orchestrated by `backend/app/services/trake/pipeline.py` and is cached
with the same corpus generation as the hybrid engine. Every complete hypothesis
contains exactly N non-negative original frame indexes from one video. Timestamp,
internal `frame_id`, filename and FAISS row are never converted into a submission
frame. Missing original-frame lineage is rejected before alignment/ranking.

Alignment has no hard maximum event gap. The configurable `none|linear|log` gap
penalty is soft; duplicate locations receive an additional soft penalty. Global
ranking preserves the best raw hypothesis, exact-deduplicates the whole sequence,
applies near-sequence NMS and distributes a first pass across videos/coarse paths.

Local refinement remains an injectable interface over canonical video files.
The production cached runtime now injects `Siglip2LocalFrameScorer`, reusing the
canonical visual engine's lazy encoder for both event text and decoded RGB frame
embeddings. It scores only bounded windows and owns no second model/index. Missing
video, decode failure or invalid scoring falls back to the coarse canonical frame
with warnings. Boundary-aware first-transition/first-leave/peak selection uses
the resulting cosine-similarity sequence; there is still no pose/contact model or
VLM verification branch, and no full-corpus semantic-boundary quality claim.

---

# 8. Query Planning Architecture

Runtime hiện tại không có generic agent loop tự chọn tool. `OnlinePipeline` tạo
một typed `QueryPlan`, có thể gọi optional bounded query-expansion provider, rồi
điều phối các retrieval branch cố định. Vì vậy “agentic” ở đây là query planning
có trace, không phải autonomous tool calling.

```text
User Query
    │
    ▼
Planner
    │
    ▼
Typed QueryPlan
    │
    ▼
Fixed retrieval branches
    │
    ▼
Result Fusion
    │
    ▼
Ranked response + routing trace
```

Ví dụ:

Query:

"Người đàn ông mặc áo đỏ bước vào xe bus"

Planner có thể:

1. Tách đối tượng.
2. Tạo nhiều query con.
3. Gắn modality hints cho visual/OCR/object.
4. Chạy các branch đã cấu hình.
5. Hợp nhất bằng weighted RRF rồi đi qua canonical dense/rerank path.

---

# 9. Data Storage

## Metadata Store

Lưu:

* selected-keyframe captions
* selected-keyframe OCR
* selected-keyframe objects
* dense visual-only identity/timestamp/path lineage

---

## Vector Database

Lưu:

* CLIP embeddings
* OpenCLIP embeddings
* selected-keyframe SigLIP2 embeddings
* full-dense SigLIP2 embeddings with a versioned dense-visual manifest

Có thể sử dụng:

* FAISS `IndexFlatIP` over L2-normalized float32 vectors
* Qdrant
* Milvus

---

## Experiment Store

Lưu:

* metrics
* benchmark results
* ablation results

---

# 10. Evaluation Metrics

Các metric chính:

## Retrieval

* Recall@10
* Recall@50
* MRR

## System

* Latency
* Throughput

## Competition

* Human Solve Time
* Success Rate

## TRAKE

For a prediction with the wrong `video_id`, R-Score is `0`. For the correct
video and N ordered events:

```text
R-Score = (1 / N) * sum_j I(frame_id_j in [s_j, e_j])
```

Intervals are inclusive and `frame_id_j` means original zero-based
`frame_index`. `R@k` is the best R-Score among the first k hypotheses for
`k = 1, 5, 20, 50, 100`; Final Score is their arithmetic mean. Diagnostic
reports also track `Video@1/@5/@20`, per-event hit rate and matched-event ratio.
Pure validation/metrics live in `backend/app/services/evaluation/trake_metrics.py`.

---

# 11. Team Responsibilities

## P1

System Integration

Benchmark

Evaluation

---

## P2

Keyframes

Embeddings

Vector Database

Indexing

---

## P3

Caption

OCR

Metadata

---

## P4

Retrieval

Re-ranking

Temporal Search

---

## P5

Query Planning

Future Frontend

Competition Workflow

---

# 12. Design Principles

1. Backend và frontend tương lai tách biệt hoàn toàn.

2. HTTP client chỉ giao tiếp qua API Contract sau khi router được mount.

3. Metadata Schema là nguồn dữ liệu chuẩn duy nhất.

4. Retrieval modules phải độc lập.

5. Query planner không truy cập dữ liệu trực tiếp; `OnlinePipeline` điều phối retrieval.

6. Mọi experiment phải tái lập được.

7. Benchmark trước khi tối ưu.

8. Ưu tiên hệ thống chạy được end-to-end trước khi nghiên cứu nâng cao.

---

# 13. Future Extensions

Kiến trúc hiện tại cho phép mở rộng:

* Multilingual Retrieval
* Video-Language Models
* Graph Retrieval
* Agentic Workflow
* Multi-Agent System
* Reinforcement-based Retrieval
* Competition Assistant

Mà không cần thay đổi kiến trúc lõi của hệ thống.
