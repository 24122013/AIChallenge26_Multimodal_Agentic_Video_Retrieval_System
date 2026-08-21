# API contract đã triển khai

Tài liệu này mô tả đúng router/service contract đang có trong
`backend/app/api/search.py` và `backend/app/api/retrieval.py`.

> Repository hiện **chưa có** `FastAPI()` application factory, router mounting,
> health endpoint, host/port hay frontend HTTP client. Vì vậy các path bên dưới là
> router-relative; chưa có base URL `/api/v1` hoặc server sẵn để gọi bằng `curl`.

## Entrypoint canonical

KIS/AVS production phải đi qua đúng chuỗi:

```text
search_online
  -> get_online_pipeline
  -> OnlinePipeline.run
  -> typed QueryPlan + optional expansion
  -> selected visual/caption/OCR/object retrieval + weighted RRF
  -> coarse clips + corpus-wide full-dense rescue
  -> per-clip CSES
  -> bounded neighbor/segment scoring
  -> deterministic rerank + exact dedup + Top-K
  -> bounded context payload attachment
```

Modality-only routes vẫn tồn tại để diagnostic/compatibility, nhưng không thay thế
canonical KIS/AVS entrypoint trên.

## Response HTTP

Các retrieval endpoint thành công được bọc như sau:

```json
{
  "success": true,
  "data": {},
  "message": null
}
```

`RequiredQaPipelineError` và `RequiredTrakePipelineError` trả HTTP 503 với partial
structured response:

```json
{
  "success": false,
  "data": {},
  "message": "reason"
}
```

Các lỗi `ValueError`, thiếu artifact và lỗi runtime khác hiện dùng FastAPI
`HTTPException` (thường là 400, 503 hoặc 500) nên body là contract chuẩn
`{"detail":"..."}`, không phải một custom error envelope thống nhất. Pydantic
request validation dùng HTTP 422. `POST /search/export` trả trực tiếp `text/csv`,
không bọc JSON.

## `POST /retrieval/online`

Đây là router contract sát nhất với canonical manager entrypoint.

```json
{
  "query": "người đàn ông mặc áo đỏ đang mở cửa xe",
  "task": "kis",
  "top_k": 20,
  "expanded_queries": [],
  "include_context": true,
  "debug": true
}
```

| Field | Type/default | Contract |
|---|---|---|
| `query` | string, bắt buộc | Service trim whitespace và từ chối query rỗng |
| `task` | string, `auto` | `auto`, `kis`, `kis_visual`, `kis_temporal`, `avs`, `temporal`, `trake`, `qa`; hai `kis_*` route trả canonical task KIS, `auto` không tự chọn TRAKE |
| `top_k` | integer, `20` | Request model cho phép 1..200; TRAKE bị cap 100, QA evidence thực tế bị cap 5 |
| `expanded_queries` | string array, `[]` | Tối đa 20; caller-owned expansion cho route phù hợp |
| `include_context` | boolean hoặc null | Bỏ/null giữ runtime default; boolean override cả neighbor và segment cho request |
| `debug` | boolean hoặc null | Bỏ/null giữ `online.debug_enabled`; boolean override request |

`include_context=true` không tự tạo artifact. Hai biến
`ONLINE_NEIGHBOR_CONTEXT_ENABLED` và `ONLINE_SEGMENT_CONTEXT_ENABLED` quyết định
nguồn nào được load. Nếu file optional bị thiếu, request vẫn chạy và trace ghi
unavailable; artifact có mặt nhưng corrupt hoặc lệch committed lineage vẫn fail.
`include_context=false` tắt cả KIS/AVS context scoring và context payload attach.
TRAKE đi qua early sequence route và không dùng hai override generic
`include_context`/`debug`; context prior và trace nội bộ của TRAKE là contract
riêng.

Với KIS/AVS, `debug=true` giữ detailed intra/inter-modality fusion trace. Dù debug
tắt, response vẫn giữ routing summary, latency, coarse-to-dense state và
context-scoring summary.

### KIS/AVS response data

Ví dụ rút gọn; candidate object luôn giữ schema tương thích, nên các field không có
evidence có thể là `0`, chuỗi rỗng, array rỗng hoặc `null`.

```json
{
  "schema_version": "1.0",
  "query": "người đàn ông mặc áo đỏ đang mở cửa xe",
  "requested_task": "kis",
  "task": "kis",
  "top_k": 20,
  "query_plan": {},
  "candidates": [
    {
      "video_id": "L01_V001",
      "keyframe_id": "FRAME_00123",
      "frame_id": "FRAME_00123",
      "frame_index": 123,
      "timestamp": 4.1,
      "shot_id": "SHOT_0001",
      "segment_id": "SEG_0001",
      "visual_score": 0.83,
      "caption_score": 0.4,
      "ocr_score": 0.0,
      "object_score": 0.2,
      "fusion_score": 0.7,
      "rerank_score": 0.76,
      "score": 0.76,
      "modality_scores": {},
      "score_breakdown": {
        "dense_visual": 0.83,
        "neighbor_support": 0.55,
        "segment_support": 0.61
      },
      "score_contributions": {
        "context_bonus_before_cap": 0.06,
        "context_bonus_after_cap": 0.06,
        "final_score": 0.76
      },
      "context_scoring": {
        "neighbor_requested": true,
        "neighbor_used_for_scoring": true,
        "neighbor_evidence_count": 2,
        "neighbor_missing_dense_count": 0,
        "segment_requested": true,
        "segment_used_for_scoring": true,
        "segment_evidence_count": 3,
        "segment_missing_dense_count": 0,
        "context_bonus_cap": 0.08,
        "cap_applied": false
      },
      "cses_selection": {},
      "neighbors": [],
      "segment_context": null,
      "context_sources": []
    }
  ],
  "context": {
    "enabled": true,
    "neighbors_enabled": true,
    "segments_enabled": true,
    "neighbors_available": true,
    "segments_available": true,
    "fallback_reason": "",
    "index": {}
  },
  "latency_ms": 42.3,
  "routing_trace": {
    "coarse_to_dense": {
      "executed": true,
      "mode": "coarse_to_dense"
    },
    "context_scoring": {
      "neighbor": {
        "requested": true,
        "artifact_available": true,
        "executed": true,
        "fallback_reason": "",
        "results_with_evidence": 12
      },
      "segment": {
        "requested": true,
        "artifact_available": true,
        "executed": true,
        "fallback_reason": "",
        "results_with_evidence": 14
      },
      "max_neighbors_each_side": 2,
      "segment_candidate_limit": 12,
      "segment_top_k": 3,
      "context_bonus_cap": 0.08
    },
    "debug_enabled": true,
    "latency": {}
  }
}
```

Context scoring nằm **sau CSES, trước final rerank/dedup/Top-K**. Nó reuse vector
của original query, full-dense vectors và caption/OCR/object metadata đã load;
không gọi encoder và không chạy thêm global FAISS/BM25 search. Lookup bị chặn bởi
`max_neighbors_each_side=2`, `segment_context_candidate_limit=12`, segment
Top-3 và combined bonus cap `0.08`. Sau Top-K, cùng context index mới hydrate
`neighbors`, `segment_context` và `context_sources` cho response.
Neighbor/segment support weights mặc định đều `0.05`; neighbor aggregate dùng
`0.65 * max + 0.35 * mean`, còn segment dùng
`0.60 * max + 0.40 * mean(Top-3)`.

Score là tín hiệu xếp hạng trong cùng query/config, không phải xác suất calibrated.
Muốn chứng minh full path, phải thấy
`routing_trace.coarse_to_dense.executed=true`, `mode="coarse_to_dense"` và context
feature tương ứng có `executed=true`; `selected_only_fallback` không chứng minh
dense rescue/CSES/context scoring.

### Task-specific data

| Task | Output chính | Ghi chú |
|---|---|---|
| `kis`, `avs` | `candidates`, `query_plan`, `context`, `routing_trace` | Shared candidate schema; profile khác nhau |
| `kis_visual` request | `task="kis"`, `candidates`, visual-scoped `query_plan`, `routing_trace` | Không có `temporal_matches`/`hypotheses`; caption/OCR/object không tham gia coarse fusion |
| `kis_temporal` request | `task="kis"`, `candidates`, temporal `query_plan`, `routing_trace` | KIS profile; không trả `temporal_matches` hoặc TRAKE `hypotheses` |
| `temporal` | `candidates`, `temporal_matches` và trace | Evidence route, không phải TRAKE submission |
| `qa` | `candidates`/`evidence`, structured `answer` status và reports | Tối đa 5 evidence; answerer phụ thuộc `QA_ANSWER_MODE` |
| `trake` | `hypotheses`, `event_plan`, `trace` | Mỗi hypothesis là complete same-video sequence, tối đa 100 |

## `POST /search`

Unified wrapper dùng `mode` thay cho `task`:

```json
{
  "query": "người đang đi xe đạp",
  "mode": "kis",
  "top_k": 20,
  "task_mode": "auto",
  "expanded_queries": [],
  "include_context": true,
  "debug": true
}
```

`mode="online"|"auto"` chuyển `task_mode` vào canonical online pipeline.
Canonical task modes khác là `kis`, `kis_visual`, `kis_temporal`, `hybrid`, `avs`, `temporal`,
`trake`, `qa`. Request UI Temporal KIS dùng ví dụ:

```json
{"query":"Khoảnh khắc đầu tiên người dẫn xuất hiện trên xích lô","mode":"kis_temporal","top_k":100}
```

Response của request này có `requested_task="kis_temporal"`, `task="kis"` và
ordinary `candidates`. `routing_trace.cses` chỉ có `executed=true` khi dense path
thật sự gọi CSES; sparse fallback ghi `coarse_to_dense.executed=false` và
`cses.executed=false`.

KIST Visual gửi `{"query":"...","mode":"kis_visual","top_k":20}`. Response
có `requested_task="kis_visual"`, `task="kis"`,
`query_plan.modality_scope=["visual"]`, `routing_trace.route="kis_visual"` và
`retrieval_profile="visual"`. Khi dense path chạy, mỗi final candidate có
`cses_selection` cùng score breakdown `coarse_visual`, `dense_visual`,
`cses_gain`, `visual_coverage`. Khi dense bundle thiếu, route selected-only phải
ghi cả coarse-to-dense và CSES `executed=false` và không phát diagnostics giả.
Các compatibility alias QA và modality-only aliases vẫn được dispatch bởi code;
client mới nên dùng tên canonical. `expanded_queries` được dùng ở online/auto và
QA paths, không phải mọi modality diagnostic. Semantics nullable của
`include_context`/`debug` giống `/retrieval/online`.

## Các retrieval routes còn lại

| Route | Request body | Vai trò |
|---|---|---|
| `POST /retrieval/visual` | `{"query":"...","top_k":20}` | Visual-only diagnostic, 1..200 |
| `POST /retrieval/kis-visual` | `{"query":"...","top_k":20}` | Canonical Visual KIS wrapper; output task KIS |
| `POST /retrieval/hybrid` | `{"query":"...","top_k":20}` | KIS compatibility wrapper |
| `POST /retrieval/caption` | `{"query":"...","top_k":20}` | Caption-only diagnostic |
| `POST /retrieval/ocr` | `{"query":"...","top_k":20}` | OCR-only diagnostic |
| `POST /retrieval/object` | `{"query":"...","top_k":20}` | Object-only diagnostic |
| `POST /retrieval/temporal` | `{"query":"...","top_k":20}` | Temporal evidence wrapper |
| `POST /retrieval/kis-temporal` | `{"query":"...","top_k":100}` | Temporal KIS compatibility wrapper; output canonical KIS |
| `POST /retrieval/trake` | `{"query":"context...\nE1: ...\nE2: ...","top_k":100}` | Core TRAKE wrapper, 1..100 |
| `POST /retrieval/qa-evidence` | `{"query":"...","top_k":5}` | QA wrapper, effective Top-5 |
| `POST /retrieval/qa` | `{"query":"...","top_k":5,"expanded_queries":[]}` | QA answer/evidence, request 1..5 |

Hai advanced request fields `include_context` và `debug` chỉ nằm trên
`/retrieval/online` và `/search`. Dùng canonical route nếu cần per-request override;
đừng gửi hai field đó vào modality-only body.
`/retrieval/qa` còn nhận `task_mode` mặc định `qa` để tương thích schema, nhưng
endpoint luôn route task QA; field này không chọn task khác.

## `POST /search/export`

```json
{
  "query": "người mặc áo đỏ cầm điện thoại",
  "task": "kis",
  "top_k": 100
}
```

Request model cho phép `top_k` từ 1 đến 100. Serializer hỗ trợ task mà exporter
đã implement (KIS, QA và TRAKE); unsupported task trả lỗi. Thành công trả CSV với
`Content-Type: text/csv; charset=utf-8` và `Content-Disposition` filename.

## Artifact và config caveats

- Canonical selected-keyframe artifacts phục vụ coarse visual/text retrieval.
- Full dense-candidate FAISS + metadata/map/manifest/report phục vụ global rescue,
  CSES và deterministic dense rerank; đây không phải mọi raw video frame.
- `neighbors_all.jsonl` và `segments_all.jsonl` được build từ canonical selected
  keyframes. Context frame identity được resolve sang full-dense rows bằng unique
  composite key `(video_id, frame_id)`; reference không có dense row được đếm và
  bỏ qua, không tự encode/search bù.
- `online.dense_missing_behavior=fallback_sparse` chỉ cho phép thiếu dense bundle
  quay về selected-only. Bundle/artifact đã có nhưng sai checksum, row order,
  dimension, identity uniqueness hoặc corpus lineage phải fail closed.
- Canonical KIS/AVS final rerank là deterministic. Không có learned/VLM heavy
  retrieval reranker để bật; legacy settings chỉ được chấp nhận để cảnh báo rồi
  bỏ qua. Điều này không áp dụng cho các BGE reranker flag riêng của QA/TRAKE.

## Không phải contract hiện tại

Các path từng xuất hiện trong tài liệu thiết kế cũ nhưng chưa được implement gồm
`/api/v1`, `/agent/search`, `/videos/{video_id}`, `/segments/{segment_id}`,
`/timeline`, `/basket/*`, `/logs/search` và `/eval/run`. Chúng không được xem là API
cho đến khi có router, service implementation và app mount kiểm chứng được.
