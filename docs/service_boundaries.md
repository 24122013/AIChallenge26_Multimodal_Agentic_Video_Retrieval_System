# service_boundaries.md

## Service Boundaries v1.0

File này định nghĩa ranh giới giữa các nhóm.

Không ai được sửa code của service khác nếu không có pull request được review.

---

# Backend Architecture

```text
Ingestion
    ↓
Metadata
    ↓
Indexing
    ↓
Retrieval
    ↓
Agent
    ↓
API
    ↓
Frontend
```

Dependency chỉ được đi từ trên xuống.

Không được đi ngược.

---

# Team P2 - Indexing

Được phép sửa:

```text
backend/app/services/indexing/*
```

Được tạo:

```text
vector index
embedding index
neighbor index
```

Không được sửa:

```text
retrieval
agent
frontend
```

---

# Team P3 - Metadata

Được phép sửa:

```text
backend/app/services/ingestion/*
backend/app/models/*
```

Được tạo:

```text
caption
ocr
objects
metadata
```

Không được chứa:

```text
search logic
reranking
agent planning
```

---

# Team P4 - Retrieval

Được phép sửa:

```text
backend/app/services/retrieval/*
```

Chỉ được đọc:

```text
metadata
vector db
```

Không được:

```text
generate captions
generate OCR
modify metadata
```

---

# Team P5 - Agent

Được phép sửa:

```text
backend/app/services/agent/*
frontend/*
```

Agent chỉ gọi API.

Agent không được truy cập:

```text
vector db trực tiếp
database trực tiếp
metadata files trực tiếp
```

---

# Frontend

Frontend chỉ được gọi:

```http
/api/v1/*
```

Frontend không được:

```text
import backend modules
read vector db
read metadata files
```

---

# Database Rule

Database là nguồn dữ liệu duy nhất.

Không truyền file giữa các service.

Mọi service giao tiếp thông qua:

* API
* Database
* Message Queue (nếu có)

---

# Logging Rule

Mọi service phải log theo format:

```json
{
  "service": "",

  "event": "",

  "timestamp": "",

  "latency": 0.0
}
```

---

# Experiment Rule

Mọi experiment phải có:

```text
config file
run script
log
result table
```

Nếu thiếu một trong bốn thứ trên:

=> experiment chưa hoàn thành.

---

# Pull Request Rule

Không merge nếu:

* Không build được
* Không pass test
* Làm thay đổi API contract
* Làm thay đổi metadata schema

Mà không cập nhật tài liệu tương ứng.

---

# Architecture Principle

Correctness
>
End-to-End
>
Latency
>
Fancy Features

Baseline trước.

Tối ưu sau.

Không nghiên cứu thay cho triển khai.
