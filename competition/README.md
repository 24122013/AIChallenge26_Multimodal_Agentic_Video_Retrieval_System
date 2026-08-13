# Competition pipeline

Pipeline competition xử lý toàn bộ candidate pool bằng SigLIP2, caption
Qwen3.5, PP-OCRv5 và YOLOE trước khi chọn keyframe canonical. Không có bước xử
lý âm thanh.

## Model và artifact

| Modality | Mặc định | Artifact |
|---|---|---|
| Visual | SigLIP2 hiện có | `siglip2.npy`, embedding metadata |
| Caption | `Qwen/Qwen3.5-9B` @ `c202236` | `captions.jsonl` |
| OCR | `PP-OCRv5_server_det` + `latin_PP-OCRv5_mobile_rec` | `ocr.jsonl` |
| Objects | `yoloe-26l-seg.pt` | `objects.jsonl` |
| Dense text | `BAAI/bge-m3` | `bge_m3_flat_ip.faiss` + map + manifest |
| Text rerank | `BAAI/bge-reranker-v2-m3` | query trace/report |

Caption, OCR và object evidence được join theo `candidate_id`/`frame_id`. Object
evidence không phải điều kiện loại candidate. Feature manifest fingerprint đầy
đủ model, revision và tham số để resume fail closed khi cấu hình thay đổi.

## Chạy end-to-end

Runner v2 dưới đây giữ output submission cũ nhưng thêm stage BGE text index và
BGE rerank. Dùng `required` khi mục tiêu là kiểm tra model mới thật sự chạy:

```powershell
.\.venv\Scripts\python.exe -m competition.run_retrieval_v2 `
  --public-root data\public `
  --run-root competition\artifacts\new-model-run `
  --device cuda `
  --bge-dense-mode required `
  --bge-reranker-mode required `
  --bge-m3-model-revision main `
  --bge-reranker-model-revision main
```

Runner có 10 stage:

```text
validate-input -> keyframes -> index -> neighbors -> segments -> text-index
-> bge-text-index -> dense-index -> predict -> validate-submission
```

`bge-text-index` chỉ đọc metadata đã có, không extract video lại. BGE-M3 và
reranker chỉ áp dụng cho TKIS trong submission; VKIS vẫn đi qua visual query.
Kết quả vẫn là 100 query x 100 answer. Với benchmark chính thức, thay revision
`main` bằng commit hash đã khóa.

Runner cũ vẫn khả dụng nếu chỉ cần pipeline metadata/offline:

```powershell
.\.venv\Scripts\python.exe -m competition.run_end_to_end `
  --public-root data\public `
  --output-root competition\artifacts `
  --device cuda
```

Hoặc chạy từng stage:

```powershell
.\.venv\Scripts\python.exe -m competition.pipeline validate-input `
  --public-root data\public

.\.venv\Scripts\python.exe -m competition.pipeline keyframes `
  --public-root data\public `
  --output-root competition\artifacts `
  --device cuda --resume

.\.venv\Scripts\python.exe -m competition.pipeline index `
  --public-root data\public --output-root competition\artifacts

.\.venv\Scripts\python.exe -m competition.pipeline neighbors `
  --public-root data\public --output-root competition\artifacts

.\.venv\Scripts\python.exe -m competition.pipeline segments `
  --public-root data\public --output-root competition\artifacts

.\.venv\Scripts\python.exe -m competition.pipeline text-index `
  --public-root data\public --output-root competition\artifacts
```

## Tùy chỉnh metadata model

Các tùy chọn quan trọng của `keyframes` và `enrich`:

```text
--caption-model-name Qwen/Qwen3.5-9B
--caption-model-revision c202236
--caption-batch-size 2
--caption-max-new-tokens 384
--caption-dtype auto|bfloat16|float16|float32
--caption-quantization none|8bit|4bit

--ocr-detection-model PP-OCRv5_server_det
--ocr-recognition-model latin_PP-OCRv5_mobile_rec
--ocr-model-revision PP-OCRv5
--ocr-batch-size 4
--ocr-conf-threshold 0.3

--object-model-name yoloe-26l-seg.pt
--object-model-revision ultralytics-official
--object-prompt-mode text|internal
--object-vocabulary <class ...>
--object-batch-size 8
--object-conf-threshold 0.25
--object-iou-threshold 0.7
```

Ví dụ chỉ tạo lại metadata hiện hành:

```powershell
.\.venv\Scripts\python.exe -m competition.pipeline enrich `
  --public-root data\public `
  --output-root competition\artifacts `
  --modalities caption ocr objects `
  --device cuda --overwrite
```

## Resume và provenance

- Model chỉ load khi modality thực sự có work pending.
- Một backend được tái sử dụng cho toàn corpus và được giải phóng trước modality
  kế tiếp.
- Checkpoint chỉ hợp lệ khi identity/order, report, model/revision, config và
  hash artifact khớp.
- Artifact lỗi không được xem là hoàn tất đối với OCR/objects, trừ khi operator
  chủ động bật `--allow-partial-features`.
- Thay model/revision chỉ làm stale artifact tương ứng; pipeline không xóa các
  file dữ liệu ngoài workspace hiện hành.

## Retrieval

Text index v3 chỉ gồm `caption`, `ocr`, `objects`. Hybrid retrieval kết hợp các
modality này với visual SigLIP2; temporal retrieval vẫn giữ cùng kiến trúc
same-video ordered matching.

Query parser/router/evidence và grounded QA là online task path riêng. Có thể
kiểm tra từng task bằng:

```powershell
python -m backend.app.services.retrieval.run_task_smoke --task kis
python -m backend.app.services.retrieval.run_task_smoke --task avs
$env:QA_ANSWER_MODE = "required"
python -m backend.app.services.retrieval.run_task_smoke --task qa
```

Các lệnh này cần `RETRIEVAL_*` và `QA_BGE_*` trỏ tới artifacts của run; xem
biến môi trường đầy đủ ở README gốc. KIS/AVS không gọi Qwen answerer. QA dùng
Top-3 evidence và trả `answered` hoặc `insufficient_evidence`. Không có ASR,
query expansion chỉ được nhận qua input ngoài, temporal chỉ được parser đánh
dấu handoff.

Object labels là prompt-dependent evidence. Khi đổi vocabulary phải rebuild
object artifacts, segment metadata và text index để lineage nhất quán.

## Tài nguyên GPU

Checkpoint Qwen 9B ở BF16 cần khoảng 19 GB chỉ cho weights; tổng runtime thường
cần khoảng 22–28 GB tùy ảnh/output/batch. Khuyến nghị ban đầu:

- RTX 5090 32 GB: caption batch 1–2 BF16.
- A100 40 GB: caption batch 2–4 BF16.
- A100 80 GB: caption batch 4–8 BF16 sau profiling.

Nếu OOM, giảm caption batch trước; sau đó cân nhắc 4-bit. Quantization có thể làm
thay đổi chất lượng output và cần được đánh giá retrieval riêng. OCR/YOLOE có
batch độc lập và chạy sau khi Qwen đã được giải phóng.

## Kiểm thử

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s competition\tests -v
```

Test orchestration dùng fixture/fake model, kiểm tra exact resume, manifest,
schema, lỗi model, geometry và deterministic identity. Chúng không thay thế
smoke test checkpoint thật trên GPU mục tiêu.
