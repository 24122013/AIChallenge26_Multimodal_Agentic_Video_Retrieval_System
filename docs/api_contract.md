# API CONTRACT v1.0

## Project

AIChallenge26 - Multimodal Agentic Video Retrieval System

---

# Mục tiêu

API Contract định nghĩa giao tiếp giữa:

* Frontend
* Backend
* Retrieval Engine
* Agent Layer

Mọi thành viên phải tuân thủ contract này.

Frontend không được truy cập trực tiếp database hoặc vector DB.

Tất cả dữ liệu phải đi qua API.

---

# Base URL

```http
/api/v1
```

---

# Common Response Format

## Success

```json
{
  "success": true,
  "data": {},
  "message": null
}
```

## Error

```json
{
  "success": false,
  "error": {
    "code": "ERROR_CODE",
    "message": "Human readable message"
  }
}
```

---

# Core Data Models

## SearchResult

```json
{
  "video_id": "L01_V001",
  "segment_id": "SEG_0001",
  "frame_id": "FRAME_00123",

  "timestamp": 123.4,

  "score": 0.95,

  "thumbnail_url": "",

  "video_url": "",

  "caption": "",

  "ocr_text": "",

  "objects": [
    "person",
    "car"
  ]
}
```

---

## Candidate

```json
{
  "candidate_id": "cand_001",

  "video_id": "",

  "timestamp": 0,

  "score": 0.0
}
```

---

# SEARCH API

## POST /search

Unified Search Endpoint

### Request

```json
{
  "query": "a man cooking in a kitchen",

  "mode": "hybrid",

  "top_k": 50,

  "rerank": true,

  "agent_enabled": false
}
```

### Response

```json
{
  "success": true,
  "data": {
    "results": []
  }
}
```

### Owner

Backend Retrieval Team

---

# VISUAL SEARCH

## POST /search/visual

### Request

```json
{
  "query": "red bus",
  "top_k": 100
}
```

### Response

```json
{
  "results": []
}
```

### Owner

P4 Retrieval

---

# OCR SEARCH

## POST /search/ocr

### Request

```json
{
  "text": "McDonalds",
  "top_k": 100
}
```

### Owner

P4 Retrieval

---

# OBJECT SEARCH

## POST /search/object

### Request

```json
{
  "object_name": "bus",
  "top_k": 100
}
```

### Owner

P4 Retrieval

---

# TEMPORAL SEARCH

## POST /search/temporal

### Request

```json
{
  "events": [
    "man enters room",
    "man sits at table"
  ],

  "constraints": {
    "order": true
  },

  "top_k": 50
}
```

### Owner

P4 Retrieval

---

# AGENT SEARCH

## POST /agent/search

### Request

```json
{
  "query": "Tìm người đàn ông mặc áo đỏ đứng gần xe bus rồi đi vào tòa nhà"
}
```

### Response

```json
{
  "plan": {},
  "results": []
}
```

### Owner

P5 Agent

---

# VIDEO DETAILS

## GET /videos/{video_id}

### Response

```json
{
  "video_id": "",

  "duration": 0,

  "fps": 0,

  "segments": []
}
```

### Owner

P3 Metadata

---

# SEGMENT DETAILS

## GET /segments/{segment_id}

### Response

```json
{
  "segment_id": "",

  "start_time": 0,

  "end_time": 0,

  "caption": "",

  "ocr_text": "",

  "objects": []
}
```

### Owner

P3 Metadata

---

# TIMELINE API

## GET /timeline

### Query Params

```text
video_id
timestamp
window
```

### Response

```json
{
  "previous_frames": [],
  "current_frame": {},
  "next_frames": []
}
```

### Owner

P2 + P5

---

# CANDIDATE BASKET

## POST /basket/add

### Request

```json
{
  "candidate_id": ""
}
```

---

## DELETE /basket/remove

```json
{
  "candidate_id": ""
}
```

---

## GET /basket

Returns all saved candidates.

### Owner

P5 UI

---

# LOGGING API

## POST /logs/search

### Request

```json
{
  "query": "",

  "mode": "hybrid",

  "latency": 1.2,

  "results_returned": 50
}
```

### Owner

P1 Evaluation

---

# EVALUATION API

## POST /eval/run

### Request

```json
{
  "experiment_name": "clip_caption_ocr"
}
```

### Response

```json
{
  "recall_at_10": 0.0,

  "recall_at_50": 0.0,

  "mrr": 0.0,

  "latency": 0.0
}
```

### Owner

P1 Evaluation

---

# Metadata Schema

Every result must contain:

```json
{
  "video_id": "",

  "segment_id": "",

  "frame_id": "",

  "timestamp": 0,

  "caption": "",

  "ocr_text": "",

  "objects": [],

  "embedding_id": ""
}
```

Missing fields are not allowed.

---

# Service Ownership

| Module              | Owner |
| ------------------- | ----- |
| Evaluation          | P1    |
| Benchmark           | P1    |
| Keyframe Extraction | P2    |
| Embedding           | P2    |
| Vector DB           | P2    |
| Caption/OCR/Objects | P3    |
| Metadata            | P3    |
| Retrieval           | P4    |
| Re-ranking          | P4    |
| Temporal Search     | P4    |
| Agent               | P5    |
| UI                  | P5    |

---

# Golden Rule

Frontend only talks to API.

Agent only talks to Retrieval APIs.

Retrieval never talks to Frontend.

Metadata is the single source of truth.

Every experiment must be reproducible through API calls.
