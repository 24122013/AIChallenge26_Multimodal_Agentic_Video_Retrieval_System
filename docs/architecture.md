# Architecture

# 1. System Overview

Mục tiêu của hệ thống là hỗ trợ truy xuất video đa phương thức (Multimodal Video Retrieval)

Hệ thống phải hỗ trợ:

* Text-based Retrieval
* Visual Retrieval
* OCR Retrieval
* Object Retrieval
* Temporal Retrieval
* Agentic Retrieval

Người dùng nhập truy vấn tự nhiên.

Hệ thống sẽ:

1. Phân tích truy vấn.
2. Chọn chiến lược tìm kiếm phù hợp.
3. Truy xuất dữ liệu từ nhiều nguồn.
4. Hợp nhất kết quả.
5. Trả về candidate tốt nhất.

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
    └── Temporal Search
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

---

# 4. Online Pipeline

Online Pipeline chạy khi người dùng gửi truy vấn.

```text
User Query
    │
    ▼
Query Understanding
    │
    ▼
Retrieval Strategy Selection
    │
    ▼
Search Execution
    │
    ▼
Result Fusion
    │
    ▼
Re-ranking
    │
    ▼
Response
```

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

Output:

Candidate Results.

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
 └── Temporal Search
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
