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
* Agentic Retrieval

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
Frontend
    │
    ▼
FastAPI Backend
    │
    ▼
Agent Layer
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
    ├── Vector Database
    ├── Metadata Store
    └── Neighbor Index
    │
    ▼
Dataset
```

---

# 3. Offline Pipeline

Offline Pipeline chịu trách nhiệm chuẩn bị dữ liệu.

```text
Raw Videos
    │
    ▼
Keyframe Extraction
    │
    ▼
Segment Extraction
    │
    ▼
Metadata Extraction
    │
    ├── Caption
    ├── OCR
    └── Objects
    │
    ▼
Embedding Generation
    │
    ▼
Index Building
```

Output:

* Keyframes
* Segments
* Metadata
* Embeddings
* Search Index

Offline keyframes là **technical keyframes**: sparse frames được chọn để lập chỉ
mục và luôn giữ original zero-based `frame_index`. TRAKE tìm **semantic
keyframes** thỏa criterion của từng event. Semantic frame có thể được refinement
trong một local window quanh technical keyframe; kiến trúc không xây dense-frame
index cho toàn corpus.

---

# 4. Online Pipeline

Online Pipeline chạy khi người dùng gửi truy vấn.

```text
User Query + explicit task
    │
    ▼
OnlinePipeline
    ├── KIS / AVS / temporal / QA
    │     └── shared planning, retrieval, evidence and reranking
    └── TRAKE
          └── event parser -> per-event retrieval -> video gating
              -> K-best frame alignment -> optional local refinement
              -> diverse ranked sequence hypotheses
    │
    ▼
Task-specific response
```

`temporal` is the existing QA/evidence-oriented task. `trake` is a separate
sequence-first task; it is selected explicitly and is not inferred by `auto`.
Both reuse canonical retrieval artifacts, but they keep different response
semantics.

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

## Agent

Nhiệm vụ:

* Query Understanding
* Query Expansion
* Query Decomposition
* Tool Calling
* Result Fusion

Output:

Intelligent Search Pipeline.

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

# 6. Frontend Architecture

Frontend chỉ chịu trách nhiệm hiển thị.

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

Frontend chỉ giao tiếp qua API.

---

# 7. Retrieval Architecture

Retrieval Layer gồm nhiều search engine độc lập.

```text
Query
 │
 ├── Visual Search
 │
 ├── Caption Search
 │
 ├── OCR Search
 │
 ├── Object Search
 │
 ├── Temporal Evidence Search
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

Mỗi search engine có thể phát triển độc lập.

TRAKE is orchestrated by `backend/app/services/trake/pipeline.py` and is cached
with the same corpus generation as the hybrid engine. Every complete hypothesis
contains exactly N non-negative original frame indexes from one video. Timestamp,
internal `frame_id`, filename and FAISS row are never converted into a submission
frame. Missing original-frame lineage is rejected before alignment/ranking.

Alignment has no hard maximum event gap. The configurable `none|linear|log` gap
penalty is soft; duplicate locations receive an additional soft penalty. Global
ranking preserves the best raw hypothesis, exact-deduplicates the whole sequence,
applies near-sequence NMS and distributes a first pass across videos/coarse paths.

Local refinement is an injectable interface over canonical video files. The
default cached runtime has no semantic `LocalFrameScorer`, so it returns coarse
canonical frames with a warning. Boundary-aware first-transition/first-leave/
peak selection is exercised only when a scorer is injected. There is no active
pose/contact model or VLM verification branch.

---

# 8. Agent Architecture

Agent không trực tiếp truy cập database.

Agent chỉ gọi Retrieval Tools.

```text
User Query
    │
    ▼
Planner
    │
    ▼
Tool Selection
    │
    ▼
Retrieval APIs
    │
    ▼
Result Fusion
    │
    ▼
Explanation
```

Ví dụ:

Query:

"Người đàn ông mặc áo đỏ bước vào xe bus"

Agent có thể:

1. Tách đối tượng.
2. Tạo nhiều query con.
3. Gọi OCR Search.
4. Gọi Object Search.
5. Gọi Temporal Search.
6. Hợp nhất kết quả.

---

# 9. Data Storage

## Metadata Store

Lưu:

* captions
* OCR
* objects

---

## Vector Database

Lưu:

* CLIP embeddings
* OpenCLIP embeddings
* SigLIP2 embeddings with a versioned encoder manifest

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

Agent

Frontend

Competition Workflow

---

# 12. Design Principles

1. Backend và Frontend tách biệt hoàn toàn.

2. Mọi giao tiếp thông qua API Contract.

3. Metadata Schema là nguồn dữ liệu chuẩn duy nhất.

4. Retrieval modules phải độc lập.

5. Agent không truy cập dữ liệu trực tiếp.

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
