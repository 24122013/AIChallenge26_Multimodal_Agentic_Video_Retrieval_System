# Colab launcher cho Retrieval v2 + BGE + grounded QA

Notebook [colab_retrieval_v2_launcher.ipynb](colab_retrieval_v2_launcher.ipynb)
là launcher end-to-end cho branch `feature/new_qa_new_model`.

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

Notebook cài `requirements-core.txt` cộng đúng một GPU profile; không cài
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
