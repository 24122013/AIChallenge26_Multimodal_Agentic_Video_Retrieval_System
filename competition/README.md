# Competition pipeline: chạy từ input đến submission

Tài liệu này mô tả đường chạy được khuyến nghị cho toàn bộ pipeline, không chỉ riêng
query expansion. Runner chuẩn là `competition.run_retrieval_v2`; mỗi stage chạy trong
process riêng để giải phóng model/GPU và lưu checkpoint bền vững.

## 1. Phạm vi

- TKIS: text-to-video KIS, query expansion bật mặc định.
- VKIS: image-to-video KIS, không tạo paraphrase.
- Temporal/TRAKE: chưa thực hiện trong chiến thuật này.
- Audio: không được đọc hoặc xử lý.

Pipeline offline tạo keyframe, visual embedding, caption, OCR và object evidence. Pipeline
online tạo query plan, truy hồi coarse+dense, rerank và ghi submission.

## 2. Input contract

Thư mục `data\public` phải có:

```text
data/public/
  corpus.csv
  questions.csv
  sample_submission.csv
  <video files referenced by corpus.csv>
  <query images referenced by questions.csv>
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

PaddlePaddle phải khớp môi trường CPU/CUDA. Ví dụ CPU:

```powershell
python -m pip install paddlepaddle
```

Với CUDA, cài wheel `paddlepaddle-gpu` tương ứng từ hướng dẫn PaddlePaddle. Không giữ
đồng thời wheel CPU và GPU. `bitsandbytes` chỉ được dùng cho quantization Qwen trên CUDA.

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
provider sinh paraphrase thật, lazy-load một lần trước khi SigLIP được cấp phát, và dùng
chung model cache với caption. Response cache riêng nằm tại
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

## 9. Kiểm thử

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
