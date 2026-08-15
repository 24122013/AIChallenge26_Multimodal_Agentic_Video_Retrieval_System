# Competition pipeline: chạy từ input đến submission

Tài liệu này mô tả đường chạy được khuyến nghị cho toàn bộ pipeline, không chỉ riêng
query expansion. Runner chuẩn là `competition.run_retrieval_v2`; mỗi stage chạy trong
process riêng để giải phóng model/GPU và lưu checkpoint bền vững.

## 1. Phạm vi

<<<<<<< HEAD
| Modality | Mặc định | Artifact |
|---|---|---|
| Visual | SigLIP2 hiện có | `siglip2.npy`, embedding metadata |
| Caption | `Qwen/Qwen3.5-9B` @ `c202236` | `captions.jsonl` |
| OCR | `PP-OCRv5_server_det` + `latin_PP-OCRv5_mobile_rec` | `ocr.jsonl` |
| Objects | `yoloe-26l-seg.pt` | `objects.jsonl` |
| Dense text | `BAAI/bge-m3` | `bge_m3_flat_ip.faiss` + map + manifest |
| Text rerank | `BAAI/bge-reranker-v2-m3` | query trace/report |
=======
- TKIS: text-to-video KIS, query expansion bật mặc định.
- VKIS: image-to-video KIS, không tạo paraphrase.
- Temporal/TRAKE: chưa thực hiện trong chiến thuật này.
- Audio: không được đọc hoặc xử lý.
>>>>>>> origin/main

Pipeline offline tạo keyframe, visual embedding, caption, OCR và object evidence. Pipeline
online tạo query plan, truy hồi coarse+dense, rerank và ghi submission.

## 2. Input contract

<<<<<<< HEAD
Runner v2 dưới đây giữ output submission cũ nhưng thêm stage BGE text index và
BGE rerank. Competition profile mặc định bắt buộc cả hai BGE mode ở
`required`; chỉ hạ xuống `off`/`optional` khi chủ động chạy local/dev:

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

`bge-text-index` chỉ đọc metadata đã có, không extract video lại, và chỉ nhận
keyframe semantic/canonical đã được chọn. Dense candidate frames bị loại khỏi
source contract. BGE-M3 và reranker chỉ áp dụng cho TKIS trong submission; VKIS
vẫn đi qua visual query.
Kết quả vẫn là 100 query x 100 answer. Với benchmark chính thức, thay revision
`main` bằng commit hash đã khóa.

Runner cũ vẫn khả dụng nếu chỉ cần pipeline metadata/offline:

```powershell
.\.venv\Scripts\python.exe -m competition.run_end_to_end `
  --public-root data\public `
  --output-root competition\artifacts `
  --device cuda
=======
Thư mục `data\public` phải có:

```text
data/public/
  corpus.csv
  questions.csv
  sample_submission.csv
  <video files referenced by corpus.csv>
  <query images referenced by questions.csv>
>>>>>>> origin/main
```

Contract được kiểm tra fail-closed:

- `corpus.csv`: đúng 250 video; header
  `video,path,duration_seconds,fps,frame_count,width,height`.
- `questions.csv`: đúng 100 câu, gồm 50 TKIS và 50 VKIS; header
  `query_id,task,text,query_image`.
- TKIS bắt buộc có `text`; VKIS bắt buộc có `query_image`.
- `sample_submission.csv`: `query_id` và đúng 100 cột
  `answer_001` … `answer_100`.
- Mọi đường dẫn tương đối phải tồn tại dưới `--public-root`.

Có thể kiểm tra input trước khi cấp phát model:

```powershell
.\.venv\Scripts\python.exe -m competition.pipeline validate-input `
  --public-root data\public
```

## 3. Cài đặt runtime

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Cài FFmpeg và kiểm tra:

```powershell
ffmpeg -version
```

PaddlePaddle phải khớp môi trường CPU/CUDA. Repo đã được kiểm thử với phiên bản
`3.2.2`; không cài đồng thời wheel CPU và GPU. Ví dụ CPU:

```powershell
python -m pip install paddlepaddle==3.2.2
```

Với CUDA, cài `paddlepaddle-gpu==3.2.2` từ index PaddlePaddle tương ứng. Blackwell
`sm_120` (RTX 5090/5080/5070) cần index `cu129` hoặc mới hơn; wheel `cu118` không hỗ
trợ kiến trúc này. Không giữ đồng thời wheel CPU và GPU. `bitsandbytes` chỉ được dùng
cho quantization Qwen trên CUDA. Runner sẽ smoke-test cả Torch và Paddle GPU trước khi
khởi động stage keyframes để lỗi runtime dừng trước khi tải model lớn.

## 4. Chạy end-to-end — đường khuyến nghị

```powershell
.\.venv\Scripts\python.exe -m competition.run_retrieval_v2 `
  --public-root data\public `
  --run-root competition\runs\retrieval-v2-query-expansion `
  --device cuda
```

Runner thực hiện theo thứ tự:

1. `validate-input`: kiểm tra CSV, số lượng và file.
2. `keyframes`: lấy dense candidates, SigLIP2, Qwen caption, PP-OCRv5, YOLOE và chọn
   canonical keyframes.
3. `index`: tạo FAISS coarse visual index.
4. `neighbors`: tạo evidence lân cận.
5. `segments`: gom metadata theo segment.
6. `text-index`: tạo BM25 cho caption/OCR/objects.
7. `dense-index`: đóng gói toàn bộ candidate vectors cho dense rescue và CSES.
8. `predict`: chạy TKIS/VKIS, query expansion, fusion và rerank.
9. `validate-submission`: kiểm tra schema, thứ tự query và 100 answer/query.

Mặc định runner dùng advanced retrieval, `--tkis-routing hybrid` và
`--retrieval-profile kis`; vì vậy temporal/TRAKE không được kích hoạt.

Lần chạy đầu tải checkpoint vào `data\model_cache`. Các model lớn được chạy tuần tự và
giải phóng giữa stage/modality. Sau khi cache hoàn chỉnh có thể thêm:

```text
--offline-model-cache
```

### Chạy trên Google Colab

Mở [colab_retrieval_v2_launcher.ipynb](../notebooks/colab_retrieval_v2_launcher.ipynb)
trên Colab, chọn GPU runtime, sửa block cấu hình đầu tiên rồi **Run all**. Notebook:

- dùng caption batch 1 và 4-bit mặc định cho T4/L4;
- cài và kiểm tra riêng PaddlePaddle GPU;
- kiểm tra VRAM, local disk, source contract và public input contract;
- lưu model cache/run artifact trên Google Drive để resume;
- đọc `query_traces.jsonl` và fail nếu production query-expansion provider không chạy
  thành công hoặc không có paraphrase hợp lệ nào được giữ.

`SOURCE_MODE='drive'` yêu cầu checkout mới nhất nằm tại `DRIVE_REPO_PATH` và phù hợp khi
branch chưa được push. Chỉ dùng `SOURCE_MODE='git'` sau khi `GIT_BRANCH` chứa toàn bộ thay
đổi query expansion/notebook. Giữ nguyên `RUN_ID` khi resume; đổi tên khi bắt đầu một run
với code, dataset hoặc config khác.

### Resume và giới hạn stage

`--run-root` là một run có lineage bất biến. Nếu source code, dataset hoặc offline config
khác manifest, runner yêu cầu một run root mới thay vì trộn artifact.

```powershell
# Chỉ build artifact đến dense index
.\.venv\Scripts\python.exe -m competition.run_retrieval_v2 `
  --public-root data\public `
  --run-root competition\runs\retrieval-v2-query-expansion `
  --device cuda --stop-after dense-index

# Tiếp tục predict và validate
.\.venv\Scripts\python.exe -m competition.run_retrieval_v2 `
  --public-root data\public `
  --run-root competition\runs\retrieval-v2-query-expansion `
  --device cuda --start-at predict

# In toàn bộ stage command, không ghi artifact và không tải model
.\.venv\Scripts\python.exe -m competition.run_retrieval_v2 `
  --public-root data\public `
  --run-root competition\runs\dry-run --dry-run
```

`--fresh` bỏ chế độ resume của stage keyframes. Hãy dùng run root mới khi chủ động đổi
model/config thay vì ghi đè lineage cũ.

## 5. Query expansion TKIS

Provider production mặc định là local `Qwen/Qwen3.5-9B` revision `c202236`; đây là
provider sinh paraphrase thật, lazy-load một lần trước khi SigLIP được cấp phát. Provider
dùng chung thư mục cache với caption nhưng giữ checkpoint 9B riêng với checkpoint caption
4B. Response cache riêng nằm tại
`data\model_cache\query_expansion`.

Luồng xử lý cho mỗi TKIS query:

1. Giữ nguyên Original Query.
2. Bảo vệ literal trong dấu nháy, OCR, số/số lượng, màu, mã, proper name, phủ định và
   quan hệ không gian.
3. Yêu cầu JSON schema cố định gồm `paraphrases`, `objects`, `attributes`, `actions`,
   `relations`, `ocr_literals`, `scene_terms`.
4. Validate từng paraphrase độc lập theo thứ tự provider trả về; giữ hai câu hợp lệ đầu
   tiên. Paraphrase hợp lệ dư bị ghi `max_paraphrases_exceeded`, không làm fallback cả
   output.
5. Visual/caption search từng variant; OCR/object chỉ nhận term đáng tin cậy đúng modality.
6. Fuse variant trong từng modality, sau đó fuse giữa các modality.
7. Dense reranker và VLM (nếu bật) dùng Original Query làm semantic query. Caption, OCR,
   object metadata và candidate image vẫn là evidence của candidate.

Với WRRF `k`, trọng số Original `w_orig`, các paraphrase `w_para_i`:

```text
original_contribution = w_orig / (k + rank_orig)
raw_expansion          = sum(w_para_i / (k + rank_para_i))
max_budget             = max_expansion_contribution * w_orig / (k + 1)
expansion_contribution = min(raw_expansion, max_budget)
intra_modality_score   = original_contribution + expansion_contribution
```

Mặc định trong `configs\retrieval.yaml`: `w_orig=1.0`, `w_para=0.6`, `k=60`,
`max_expansion_contribution=1.0`. Score thô khác scale giữa engine không được cộng trực
tiếp vào công thức này.

Original-only chỉ xuất hiện trong ba trường hợp có chủ đích:

- provider lỗi/timeout/schema invalid → fallback có lý do trong trace;
- explicit ablation `--no-query-expansion`;
- provider trả output hợp lệ với `paraphrases: []` hoặc mọi paraphrase bị validator loại.

Ablation:

```powershell
.\.venv\Scripts\python.exe -m competition.run_retrieval_v2 `
  --public-root data\public `
  --run-root competition\runs\retrieval-v2-original-only `
  --device cuda --no-query-expansion
```

## 6. VLM reranker tùy chọn

Mặc định `--vlm-mode off`. Có thể bật fail-open:

```text
--vlm-mode optional --vlm-top-m 20
```

Hoặc fail-closed cho môi trường bắt buộc VLM:

```text
--vlm-mode required --vlm-top-m 20
```

VLM luôn nhận Original Query và ảnh candidate. `optional` quay về deterministic ranking
nếu model lỗi/timeout; `required` dừng pipeline.

## 7. Artifact và trace

Trong `competition\runs\<run>`:

```text
run_manifest.json
work/keyframe_v3/...
metadata/...
indexes/...
results/submission.csv
results/query_traces.jsonl
```

`query_traces.jsonl` ghi cho từng query:

- Original Query, protected literals, provider/model/prompt revision và cache hit;
- accepted/rejected paraphrase cùng rejection reason;
- structured decomposition và modality routing;
- original/raw/capped expansion contribution cho từng candidate;
- inter-modality contribution, canonical rerank query, VLM report và final results.

`submission.csv` chỉ được coi là hoàn tất sau khi stage `validate-submission` passed.

## 8. Chạy từng stage thủ công

Runner ở mục 4 nên được ưu tiên. Khi debug, dùng các command mà runner sẽ phát ra:

```powershell
.\.venv\Scripts\python.exe -m competition.pipeline keyframes `
  --public-root data\public --output-root competition\runs\manual `
  --device cuda --resume

.\.venv\Scripts\python.exe -m competition.pipeline index `
  --public-root data\public --output-root competition\runs\manual

.\.venv\Scripts\python.exe -m competition.pipeline neighbors `
  --public-root data\public --output-root competition\runs\manual

.\.venv\Scripts\python.exe -m competition.pipeline segments `
  --public-root data\public --output-root competition\runs\manual --strategy auto

.\.venv\Scripts\python.exe -m competition.pipeline text-index `
  --public-root data\public --output-root competition\runs\manual

.\.venv\Scripts\python.exe -m competition.pipeline dense-index `
  --run-root competition\runs\manual `
  --source-workspace competition\runs\manual\work\keyframe_v3 `
  --source-output-root competition\runs\manual `
  --public-root data\public
```

Để tránh bỏ sót flag/lineage ở `predict`, dùng `--dry-run` của runner rồi sao chép command
được in ra thay vì tự dựng một command cũ.

<<<<<<< HEAD
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
$env:QA_BGE_DENSE_ENABLED = "true"
$env:QA_BGE_RERANKER_ENABLED = "true"
python -m backend.app.services.retrieval.run_task_smoke --task qa
```

Các lệnh này cần `RETRIEVAL_*` và `QA_BGE_*` trỏ tới artifacts của run; xem
biến môi trường đầy đủ ở README gốc. KIS/AVS không gọi Qwen answerer. QA
non-temporal dùng tối đa Top-3 evidence. QA temporal chạy retrieval theo từng
event, giữ toàn bộ strict chain tối đa 5 evidence; chain `relaxed_gap` hoặc
`sparse_compat` chỉ phục vụ audit và trả `insufficient_evidence` mà không gọi
Qwen. Không có ASR; external whole-query expansion bị bỏ qua ở temporal route
và được ghi trong trace.

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
=======
## 9. Kiểm thử
>>>>>>> origin/main

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s backend\tests -v
.\.venv\Scripts\python.exe -m unittest discover -s competition\tests -v
```

Test query expansion chứng minh giới hạn hai paraphrase, công thức cap, Original Query ở
dense/VLM reranker, metadata candidate làm evidence, provider production mặc định và các
nhánh fallback/ablation/zero-paraphrase. Test dùng fake provider/runner nên không tải model.

## 10. Troubleshooting

- **OOM ở Qwen**: giảm batch caption, giữ quantization 4-bit, hoặc chạy lại trên GPU có
  nhiều VRAM; các model được runner tách process để tránh giữ VRAM giữa stage.
- **Không tìm thấy model khi offline**: chạy một lần không có `--offline-model-cache` để
  lấp cache đúng revision.
- **Paddle import/CUDA lỗi**: gỡ wheel Paddle CPU/GPU không phù hợp rồi cài đúng một wheel.
- **FFmpeg không tìm thấy**: thêm binary vào `PATH` trước stage keyframes.
- **Manifest/lineage mismatch**: tạo `--run-root` mới; không trộn artifact từ code hoặc
  dataset khác.
- **Provider expansion lỗi**: TKIS vẫn chạy original-only và lý do nằm trong
  `results\query_traces.jsonl`; không cần bật ablation để pipeline sống sót.
