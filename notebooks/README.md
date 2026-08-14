# Colab launcher cho Retrieval v2 + BGE + grounded QA

Branch `feature/new_qa_new_model` có hai launcher end-to-end cùng kiến trúc và
cùng contract output:

| Notebook | Cấu hình keyframes | GPU phù hợp | Ảnh hưởng dự kiến |
|---|---|---|---|
| [E2E.ipynb](E2E.ipynb) | Qwen caption 4-bit, batch 1; OCR/object batch 2; 0 worker | T4/L4 | Ít VRAM hơn; caption có thể khác nhẹ do quantization |
| [E2E_FULL_PRECISION.ipynb](E2E_FULL_PRECISION.ipynb) | Qwen caption không quantize, batch 2; OCR batch 4; object batch 8; 2 worker | Tối thiểu khoảng 24 GB, khuyến nghị A100 40 GB | Giữ cấu hình chất lượng cũ, tốn VRAM hơn |

Giảm batch và số worker chỉ đổi tốc độ/bộ nhớ, không chủ ý đổi chất lượng.
Khác biệt chất lượng có thể đến từ Qwen caption 4-bit trong profile low-memory.
Cả hai notebook vẫn bắt buộc BGE-M3 và BGE reranker như nhau.

## Output được giữ nguyên

Public `questions.csv` hiện có đúng 100 query: 50 TKIS text và 50 VKIS image,
không có QA. Notebook vẫn tạo:

```text
<run_root>/results/submission.csv
```

File giữ nguyên header, thứ tự `query_id`, 100 answer/query và strict validator
cũ. QA không được chèn vào submission.

## Model mới được test ở đâu

- `bge-text-index`: build BGE-M3 dense index từ caption + OCR + objects.
- `predict`: TKIS fusion thêm `dense_text`, sau đó BGE cross-encoder rerank.
- `task smoke`: KIS/AVS chạy parser → router → retrieval → evidence; QA chạy
  thêm Qwen3.5 grounded answerer trong process riêng.
- ASR không được cài, index hoặc score.

Hai BGE mode và QA answer mode mặc định là `required`. Nếu checkpoint/index
không load hoặc inference lỗi, notebook dừng. Đây là chủ ý để một lần chạy test
model mới không thể âm thầm biến thành baseline cũ.

## Trước khi chạy

1. Chọn Colab GPU runtime và mount Google Drive.
2. Sửa cell `parameters`: `GIT_URL`, source dataset, run/cache roots và GPU
   profile `cu118|cu126`.
3. Nếu clone Git, giữ `GIT_BRANCH = "feature/new_qa_new_model"`.
4. Chạy tuần tự toàn bộ cell.
5. Lần đầu cần mạng và đủ dung lượng Drive cho video artifacts/checkpoint.

Hai notebook cài `requirements-core.txt` cộng đúng một GPU profile; không cài
`requirements.txt` CPU. Sau install, cell preflight bắt buộc cả Torch và Paddle
nhìn thấy CUDA.

## Resume

Giữ nguyên `RUN_ID`, đọc `run_manifest.json`, rồi đặt `START_AT` thành stage đầu
tiên chưa `passed`. Kiến trúc mới có 10 stage:

```text
validate-input, keyframes, index, neighbors, segments, text-index,
bge-text-index, dense-index, predict, validate-submission
```

Không đổi `RUN_ID` giữa chừng nếu muốn dùng checkpoints cũ. Nếu đổi dataset,
source code fingerprint hoặc keyframe config, runner sẽ fail closed và yêu cầu
run root mới.

Khi so sánh hai profile, phải dùng hai `RUN_ID` riêng, ví dụ
`new-model-lowmem-001` và `new-model-full-001`. Không resume artifact đã tạo bằng
4-bit vào run full precision hoặc ngược lại; manifest cần giữ đúng lineage của
cấu hình sinh keyframe/caption.

## Artifact để kiểm tra

- `run_manifest.json`: cả 10 stage `passed`.
- `indexes/bge_m3/bge_m3_manifest.json`: model revision, dimension, ordering và
  source hashes.
- `results/query_traces.jsonl`: 50 TKIS phải có `dense_status=passed` và
  `reranker.status=passed`.
- `results/submission.csv`: 100 dòng, 100 answer/query.
- `results/task_smoke.json`: trace/evidence KIS, AVS và answer/evidence QA.

Smoke test chỉ chứng minh model, artifact và interface chạy được. Public data
không có QA label và notebook không được dùng để tuyên bố model mới tốt hơn.
