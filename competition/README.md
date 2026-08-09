# Pipeline đầy đủ cho cuộc thi TKIS/VKIS

`competition/` là adapter của toàn bộ phần đã được triển khai trong hệ thống gốc cho
bộ dữ liệu `data/public/`. Adapter không viết lại thuật toán lõi mà gọi trực tiếp các
service hiện có:

- TransNetV2 + dense sampling + multimodal hard selector để trích keyframe;
- SigLIP2 để tạo image/text embedding;
- FAISS `IndexFlatIP` cho visual retrieval;
- BLIP caption, EasyOCR, YOLO objects và Whisper ASR;
- temporal-neighbor index và segment-level metadata;
- BM25 text index cho caption/OCR/ASR/object;
- `HybridSearchEngine`, `HybridReranker` và weights trong `configs/retrieval.yaml`;
- image-to-image SigLIP2 và local frame refinement cho VKIS.

Không có bước nào tự chạy. Đứng tại thư mục gốc repo và tự chạy lần lượt các lệnh
dưới đây.

## Luồng đầy đủ

```text
videos
  -> dense candidates + shot anchors
  -> SigLIP2 + caption + OCR + objects + ASR trên toàn candidate pool
  -> feature adapter -> protected events + shot/temporal coverage
  -> canonical keyframes + embeddings + multimodal metadata
  -> FAISS + frame map
  -> neighbor metadata
  -> multimodal segments -> BM25 text index
  -> TKIS: visual + caption + OCR + ASR + objects -> hybrid rerank
  -> VKIS: image FAISS -> frame-by-frame refinement
  -> submission.csv
```

## Cấu trúc artifact

```text
competition/
├── keyframes/     # ảnh keyframe theo video
├── metadata/      # keyframe, multimodal, segment, frame map, Phase 3/4 manifest
├── embeddings/    # vector SigLIP2 theo video
├── indexes/       # visual FAISS, text index và Phase 4 lineage sidecar
├── results/       # submission.csv cuối cùng
├── evaluation/    # split, config lock và report Phase 5
└── work/
    └── keyframe_v3/ # candidate pool, feature manifest và selection lineage
```

Artifact sinh ra đã được `.gitignore`; `.gitkeep` chỉ giữ cấu trúc thư mục.

Các module `backend/app/services/agent/` hiện là stub rỗng nên không có planner,
query expansion hay tool execution khả dụng để gọi lại. Adapter không tự phát minh
logic thay thế. QA/evaluation không tham gia submission vì public set không cung cấp
ground truth và output cuộc thi chỉ nhận `video,frame_idx`.

## 0. Chuẩn bị và kiểm tra input

Input phải có cấu trúc:

```text
data/public/
├── corpus.csv
├── questions.csv
├── sample_submission.csv
├── videos/*.mp4
└── vkis/frames/*.jpg
```

Cài dependency và kiểm tra cả `ffmpeg` lẫn `ffprobe`:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
ffmpeg -version
ffprobe -version
```

Kiểm tra CSV, 250 video và 50 ảnh VKIS mà không xử lý video:

```powershell
.\.venv\Scripts\python.exe -m competition.pipeline validate-input
```

Public set hợp lệ phải có 250 video, 100 query, gồm 50 TKIS và 50 VKIS.

## Chạy end-to-end bằng một runner

`competition/run_end_to_end.py` gọi tuần tự toàn bộ stage trong process riêng:

```text
validate-input -> keyframes -> index -> neighbors -> segments
               -> text-index -> predict -> validate-submission
```

GPU, dùng cấu hình keyframe khuyến nghị `0.5s candidate / 2s max gap`:

```powershell
.\.venv\Scripts\python.exe -m competition.run_end_to_end `
  --public-root data\public `
  --device cuda `
  --batch-size auto
```

Runner kiểm tra CUDA trước khi xử lý video. Nếu báo `CPU-only PyTorch build`, kiểm
tra môi trường đang kích hoạt:

```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

Với Python 3.12 và cặp version hiện tại `torch 2.13.0 / torchvision 0.28.0`, cài
CUDA 13.0 wheel chính thức bằng:

```powershell
.\.venv\Scripts\python.exe -m pip uninstall -y torch torchvision
.\.venv\Scripts\python.exe -m pip install torch==2.13.0 torchvision==0.28.0 `
  --index-url https://download.pytorch.org/whl/cu130
```

Sau đó chạy lại lệnh kiểm tra; kết quả cần có CUDA version và
`torch.cuda.is_available()` bằng `True` trước khi dùng `--device cuda`.

CPU:

```powershell
.\.venv\Scripts\python.exe -m competition.run_end_to_end `
  --public-root data\public `
  --device cpu `
  --batch-size 4 `
  --num-workers 0
```

Runner mặc định truyền `--resume` cho stage keyframe. Nếu bị dừng sau khi đã tạo
keyframe/index, có thể tiếp tục từ một stage cụ thể, ví dụ:

```powershell
.\.venv\Scripts\python.exe -m competition.run_end_to_end `
  --public-root data\public `
  --device cuda `
  --start-at index
```

Xem toàn bộ lệnh mà không chạy model:

```powershell
.\.venv\Scripts\python.exe -m competition.run_end_to_end --dry-run --device cuda
```

Output cuối mặc định là `competition/results/submission.csv`. Stage cuối đối chiếu
header, thứ tự 100 query và 100 answer column với chính
`data/public/sample_submission.csv`; runner dừng khác `0` nếu submission sai format.

Mỗi lần runner thực sự chạy (thành công, thất bại hoặc bị ngắt), nó append runtime
từng stage, config, Git revision, artifact counts, keyframe guarantees, index và
submission SHA256 vào `reports/Experiment.md`. Dry-run không ghi experiment. Điểm
scoreboard không thể lấy từ artifact local; sau khi submit có thể ghi bằng:

```powershell
.\.venv\Scripts\python.exe -m competition.experiment_tracker `
  --public-score 0.818 `
  --private-score 0.818 `
  --note "Tên hoặc mô tả submission"
```

Hoặc truyền `--public-score`, `--private-score` và `--experiment-note` trực tiếp
cho `competition.run_end_to_end`. Dùng `--no-experiment-log` nếu cố ý không muốn
ghi một run chẩn đoán.

## 1. Tạo keyframe Phase 3 (khuyến nghị)

Phase 3 chạy trọn workflow keyframe multimodal trước khi build index. GPU:

```powershell
.\.venv\Scripts\python.exe -m competition.pipeline keyframes --device cuda --resume
```

CPU:

```powershell
.\.venv\Scripts\python.exe -m competition.pipeline keyframes --device cpu --batch-size 4 --resume
```

Mỗi video đi qua một pipeline liền mạch:

1. TransNetV2 + dense sampling mặc định `0.5s` + shot boundary anchors tạo
   candidate pool cô lập;
2. SigLIP2, BLIP caption, EasyOCR, YOLO objects và Whisper ASR chạy trên
   toàn bộ candidate pool;
3. feature adapter tạo importance/novelty score và protected event cho rare/new
   OCR, rare/new object và strong semantic transition;
4. hard selector giữ detected protected events, ít nhất một representative mỗi
   detected shot và chèn frame cho đến khi cả head/tail lẫn gap giữa hai
   keyframe không vượt `--max-gap-seconds`;
5. chỉ khi independent audit xác nhận mọi hard constraint, pipeline mới
   publish canonical keyframe, embedding và multimodal artifact vào các đường
   dẫn chuẩn trong `competition/`.

Mỗi canonical file được ghi bằng atomic replace. Phase 3 manifest ghi checksum
của toàn bộ output; extraction report là commit marker được replace sau cùng.
Sau publish, pipeline còn chạy lại extraction/embedding contract hiện hành như
một kiểm tra độc lập và thoát khác `0` nếu validation thất bại.

`--target-keyframes` mặc định là `None`, nên selector dừng ngay khi detected
event coverage, shot coverage và temporal coverage đã đạt. Nếu đặt soft target,
target chỉ bổ sung diversity và không được làm yếu hard guarantee; nếu
`--hard-max-keyframes` làm constraint không thể thỏa, pipeline dừng và không
công bố kết quả như một run hợp lệ.

Guarantee về event là **exact recall trên detected protected events**: mỗi event
được feature adapter tạo từ artifact thành công sẽ có ít nhất một candidate
được chọn. Đây không phải ground-truth recall; text, object hay transition
mà model không phát hiện thì selector không thể bảo vệ.

Pipeline fail-closed theo mặc định:

- SigLIP2 phải có đúng một embedding hợp lệ cho mỗi candidate, đúng
  identity và thứ tự;
- OCR và object phải có record `success` cho mỗi candidate;
- caption và ASR là soft context, không tự tạo hard guarantee;
- manifest, checksum, source/config lineage hoặc independent hard audit không
  khớp thì canonical publish bị từ chối.

Chỉ dùng `--allow-partial-features` khi chấp nhận một run degraded có chủ
đích. Nếu OCR/object thực tế chưa đầy đủ, manifest ghi `degraded=true`,
và event guarantee chỉ áp dụng cho event thực sự phát hiện từ các
artifact thành công; không được báo cáo run này như full multimodal
coverage.

Workspace trung gian nằm tại
`<output-root>/work/keyframe_v3/<video_id>/<run_id>/`. `run_id` gắn candidate
pool với source video và extraction config; feature/selection manifest tiếp tục
gắn model config, resolved revision, checksum và selected identities. `--resume`
chỉ skip khi exact manifest và artifact checksum còn khớp; artifact stale, bị sửa
hoặc đổi config sẽ được tạo lại thay vì dùng nhầm cache.

Resume được kiểm tra riêng cho SigLIP2, caption, OCR, objects và ASR, nên một lỗi
muộn ở ASR không làm chạy lại các modality đã hoàn thành. ASR chạy trong process
riêng cho từng video: mặc định timeout GPU sau 90 giây, retry một lần rồi fallback
CPU với timeout 600 giây. Có thể chỉnh bằng `--asr-timeout-seconds`,
`--asr-retries` và `--asr-cpu-timeout-seconds`. Record ASR `status=error` không
được phép tạo feature manifest `passed`; video không có audio vẫn được chấp nhận
với `skipped/no_audio_stream`.

Sau lệnh `keyframes`, canonical embedding và caption/OCR/object/ASR đã có;
bước kế tiếp là `index`, không chạy lại `embed` hay `enrich`.

## Luồng Phase 2 rollback/chẩn đoán

Các lệnh `extract`, `embed` và `enrich` được giữ lại để rollback hoặc
chẩn đoán từng stage. Chúng không thay thế guarantee multimodal end-to-end
của lệnh `keyframes`.

### P2.1. Trích keyframe

```powershell
.\.venv\Scripts\python.exe -m competition.pipeline extract --device auto
```

Bước này gọi `extract_keyframes_for_video` của repo. Ảnh nằm trong
`competition/keyframes/<video_id>/`; metadata chứa `frame_index` 0-based,
`shot_start` và `shot_end` nằm trong `competition/metadata/`.

Competition mặc định dùng strategy `dense_coverage`: tạo candidate mỗi `0.5s`,
bảo vệ ít nhất một boundary representative cho mỗi shot, rồi chèn candidate cho
đến khi mọi temporal gap không vượt `5s`. `target_keyframes` mặc định không đặt,
nên selector dừng ngay khi hard constraint đã đạt. Có thể đổi ngưỡng bằng
`--candidate-interval-sec` và `--max-gap-seconds`; rollback bằng
`--keyframe-strategy legacy`.

Trong luồng rollback này, hard guarantee chỉ là temporal coverage và shot
representative; OCR/object/SigLIP chưa chạy trên dense pool trước selector.

Nếu bị gián đoạn:

```powershell
.\.venv\Scripts\python.exe -m competition.pipeline extract --device auto --resume
```

`--resume` chỉ skip khi report khớp toàn bộ extraction config và, với
`dense_coverage`, report xác nhận `constraints_satisfied=true`. Pipeline còn kiểm
tra fingerprint video nguồn, checksum metadata và cả nội dung từng ảnh. Nếu không
thể đạt hard constraint, lệnh dừng với report `status=partial`; không chạy bước
embedding cho đến khi sửa nguyên nhân hoặc điều chỉnh config có chủ đích.

### P2.2. Tạo embedding SigLIP2

GPU:

```powershell
.\.venv\Scripts\python.exe -m competition.pipeline embed --device cuda --batch-size auto
```

CPU:

```powershell
.\.venv\Scripts\python.exe -m competition.pipeline embed --device cpu --batch-size 4
```

Model được nạp một lần và dùng lại cho 250 video. Có thể thêm `--resume` khi chạy
lại. Model/revision/vector dimension được ghi vào metadata để kiểm tra contract.
Embedding lưu lineage của extraction; đổi strategy, temporal gap, candidate pool
hoặc video nguồn sẽ buộc encode lại thay vì dùng nhầm artifact cũ. Resume đối chiếu
commit model đã resolve, checksum `.npy`/JSONL, số lượng và identity frame. Bước
index từ chối embedding không khớp, còn `predict` từ chối FAISS manifest cũ nếu
extract/embedding đã thay đổi mà index chưa được build lại.

Sau khi extraction thay đổi, chạy lại `embed` và `index`. Với artifact caption/OCR/
object/ASR đã có, dùng `enrich --overwrite` để tránh giữ metadata của candidate cũ.

### P2.3. Sinh toàn bộ multimodal metadata

Có thể chạy cả bốn pipeline trong một lệnh:

```powershell
.\.venv\Scripts\python.exe -m competition.pipeline enrich --device cuda
```

Khuyến nghị chạy tách từng model để dễ theo dõi VRAM và tiếp tục khi gián đoạn:

```powershell
.\.venv\Scripts\python.exe -m competition.pipeline enrich --modalities caption --device cuda
.\.venv\Scripts\python.exe -m competition.pipeline enrich --modalities ocr --device cuda
.\.venv\Scripts\python.exe -m competition.pipeline enrich --modalities objects --device cuda
.\.venv\Scripts\python.exe -m competition.pipeline enrich --modalities asr --device cuda
```

Các service gốc tự bỏ qua record/video đã có output, nên bốn lệnh trên có thể chạy
lại an toàn. Chỉ dùng `--overwrite` khi muốn tạo lại artifact của modality đó.

Output cho mỗi video:

```text
competition/metadata/captions_<video_id>.jsonl
competition/metadata/ocr_<video_id>.jsonl
competition/metadata/objects_<video_id>.jsonl
competition/metadata/asr_<video_id>.jsonl
competition/metadata/asr_segments_<video_id>.jsonl
competition/metadata/<modality>_<video_id>_report.json
```

Mặc định tương ứng hệ thống gốc:

- caption: `Salesforce/blip-image-captioning-base`, batch 4, có segment caption;
- OCR: EasyOCR `vi` + `en`, threshold 0.3, batch 4;
- objects: `yolo11n.pt`, confidence 0.25, IoU 0.7, batch 8;
- ASR: faster-whisper/Whisper `small`, auto language và VAD.

Video không có audio được ASR ghi `skipped/no_audio_stream`, không phải lỗi.

Sau khi dùng luồng Phase 2, chỉ `enrich` canonical keyframe đã được
selector chọn; vì vậy nó không bổ sung protected-event guarantee của Phase 3.

Từ bước `index` trở đi là **Phase 4 — competition và downstream artifacts**.
Phase này không infer lại SigLIP2/OCR/object/ASR. Nó dùng trực tiếp embedding và
feature đã được Phase 3 subset theo keyframe cuối, truyền selection provenance
qua FAISS/segment/text index và khóa mỗi stage bằng source hash.

## 2. Tạo visual FAISS index

```powershell
.\.venv\Scripts\python.exe -m competition.pipeline index
```

Lệnh chỉ nhận embedding của đúng video trong `corpus.csv`, rồi gọi
`build_faiss_artifacts` để tạo FAISS, frame map và encoder manifest. Frame map giữ
`selection_phase`, reasons, protected event IDs, `importance_score`,
`semantic_novelty`, component scores và selection provenance; `MetadataStore`
đọc lại nguyên các field này để retrieval/UI/evaluation có thể audit quyết định
chọn frame.

## 3. Tạo temporal-neighbor metadata

```powershell
.\.venv\Scripts\python.exe -m competition.pipeline neighbors --window-seconds 5
```

Output là `competition/metadata/neighbors_all.jsonl`. Đây là artifact tùy chọn của
hệ thống gốc để tra ngữ cảnh trước/sau; visual search hiện cũng lấy same-shot
neighbors trực tiếp từ frame map. Sidecar
`neighbors_all_phase4_manifest.json` gắn output với đúng canonical keyframe run.

## 4. Aggregate multimodal segment

Chỉ chạy sau khi đủ caption, OCR, objects và ASR của 250 video:

```powershell
.\.venv\Scripts\python.exe -m competition.pipeline segments --strategy auto
```

Lệnh gọi `build_segment_metadata`, dùng boundary `segment_id`/`shot_id`, rồi gộp
caption, OCR, ASR và objects kèm provenance vào
`competition/metadata/segments_all.jsonl`. Mỗi segment còn giữ
`keyframe_selection`, union protected-event IDs và cờ protected. Sidecar
`segments_all_phase4_manifest.json` hash toàn bộ canonical keyframe cùng file
caption/OCR/object/ASR đầu vào.

## 5. Tạo BM25 text index

```powershell
.\.venv\Scripts\python.exe -m competition.pipeline text-index
```

Output `competition/indexes/retrieval_text_index.json` chứa bốn modality:
`caption`, `ocr`, `asr`, `objects`. Không có file này thì `predict` sẽ dừng thay vì
âm thầm fallback về visual-only; điều này bảo đảm submission thực sự dùng hybrid.
File được atomic replace và có sidecar
`retrieval_text_index_phase4_manifest.json` ràng buộc checksum của
`segments_all.jsonl`, canonical keyframe run và chính text index.

## 6. Chạy TKIS/VKIS và tạo submission

GPU:

```powershell
.\.venv\Scripts\python.exe -m competition.pipeline predict --device cuda
```

CPU:

```powershell
.\.venv\Scripts\python.exe -m competition.pipeline predict --device cpu --batch-size 4
```

Kết quả nằm tại `competition/results/submission.csv`.

Trước khi nạp model, `predict` kiểm tra cả FAISS lineage lẫn chuỗi Phase 4
`canonical → segments → text-index`. Nếu keyframe, multimodal metadata, segment
hoặc text index bị sửa/đổi run, lệnh dừng và yêu cầu build lại stage tương ứng;
không dùng lẫn artifact cũ và mới.

### TKIS

TKIS không còn là visual-only. Mỗi query chạy đồng thời qua:

1. SigLIP2 text-to-keyframe visual search;
2. BM25 caption search;
3. BM25 OCR search;
4. BM25 ASR search;
5. BM25 object-label search;
6. candidate merge và `HybridReranker`.

Pipeline đọc weights, same-shot dedupe và các giới hạn retrieval từ
`configs/retrieval.yaml`. Có thể truyền config khác bằng `--retrieval-config`, nhưng
không cần nếu muốn giống hệ thống gốc.

Mặc định `--tkis-routing auto-temporal`: query có cấu trúc thứ tự như `then`,
`after that`, `followed by` sẽ dùng `HybridSearchEngine.temporal_search`; mỗi event
vẫn được truy xuất bằng full hybrid trước khi ghép theo thứ tự thời gian. Query đơn
event dùng `HybridSearchEngine.search`. Dùng `--tkis-routing hybrid` nếu muốn tắt
nhận diện temporal tự động.

### VKIS

VKIS vẫn dùng nhánh phù hợp với truy vấn ảnh:

1. encode 50 query image theo batch bằng cùng model/revision SigLIP2 của corpus;
2. tìm keyframe gần nhất bằng cùng FAISS index;
3. với 20 ứng viên đầu, so ảnh query với từng frame trong ±75 frame quanh keyframe
   và trong biên shot bằng hàm `mse` sẵn có;
4. trả chỉ số frame 0-based khớp nhất.

Bước refine giúp vượt qua việc keyframe đúng video/shot nhưng cách frame gốc quá
dung sai VKIS ±12. Có thể tăng độ phủ:

```powershell
.\.venv\Scripts\python.exe -m competition.pipeline predict `
  --device cuda `
  --search-depth 300 `
  --vkis-refine-top-k 30 `
  --vkis-refine-radius-frames 100
```

`--search-depth` phải từ 100 trở lên. Tăng các giá trị VKIS làm chậm hơn nhưng không
cần build lại artifact.

## 7. Phase 5 — validation và rollout

Phase 5 đánh giá lại artifact canonical của Phase 3; không tin trực tiếp các cờ
`coverage_satisfied` đã ghi trong extraction report. Chọn đúng 16 video đại diện,
ghi mỗi `video_id` trên một dòng trong `phase5_video_ids.txt`, rồi tạo split bất biến:

```powershell
.\.venv\Scripts\python.exe -m competition.pipeline phase5-init `
  --video-ids-file phase5_video_ids.txt `
  --seed 42
```

Split luôn có `4 dev / 4 validation / 8 locked test`. Thứ tự được tạo bằng
SHA-256 từ seed và video ID, manifest có checksum assignment và không được ghi đè.
Chỉ tune threshold/config trên dev. Sau khi dev đạt yêu cầu, khóa đúng extraction
config đã tạo bốn artifact dev:

```powershell
.\.venv\Scripts\python.exe -m competition.pipeline phase5-evaluate --split dev
.\.venv\Scripts\python.exe -m competition.pipeline phase5-lock
.\.venv\Scripts\python.exe -m competition.pipeline phase5-evaluate --split validation
```

Chỉ mở test sau khi đã chốt config từ dev và đọc report validation. Test cần xác
nhận rõ ràng, chỉ được ghi một lần và không có tùy chọn overwrite:

```powershell
.\.venv\Scripts\python.exe -m competition.pipeline phase5-evaluate `
  --split test `
  --confirm-locked-test
```

Report nằm tại `competition/evaluation/keyframe_phase5_<split>_report.json` và có:

- `coverage_violation_count`, `max_gap_seconds`, `p95_gap_seconds`, gồm cả gap
  từ đầu video đến keyframe đầu và từ keyframe cuối đến cuối video;
- `effective_shot_recall` và `detected_protected_event_recall`, được tính lại từ
  canonical metadata/candidate/event ledgers;
- `manual_end_to_end_event_recall` và `manual_detector_event_recall`, chỉ có giá
  trị khi cung cấp human interval annotations;
- `false_protection_rate`, chỉ có giá trị trên detected event đã được người đánh giá;
- `keyframes_per_minute`, soft-budget overrun, runtime/RAM coverage và disk bytes;
- retrieval `Hit@1/5/10/100`, chỉ có giá trị khi cung cấp ranked retrieval evidence.

Evidence là JSONL tùy chọn. Mọi record manual/review/resource phải có `video_id`.
Ví dụ tối thiểu:

```json
{"video_id":"L27_V001","event_id":"MANUAL_001","start_time":12.2,"end_time":13.0}
{"video_id":"L27_V001","detected_event_id":"OCR_EVENT_001","is_true_event":true}
{"video_id":"L27_V001","runtime_sec":18.4,"peak_ram_mb":2310.0}
```

Retrieval evidence dùng interval frame ground truth và danh sách kết quả đã xếp hạng:

```json
{"query_id":"Q001","relevant":[{"video_id":"L27_V001","start_frame":300,"end_frame":325}],"ranked_results":[{"video_id":"L27_V001","frame_index":312}]}
```

Truyền các file tương ứng bằng `--manual-events`, `--protection-reviews`,
`--resource-usage`, `--retrieval-evidence`; có thể đặt dung sai annotation bằng
`--manual-tolerance-seconds`. Thiếu evidence thì metric tương ứng là `null`, không
được tự coi là 1.0. `detected_protected_event_recall` chỉ đo selector trên event mà
feature adapter đã phát hiện; claim end-to-end phải dựa vào manual annotation.

CI dùng fake/synthetic artifacts để gate logic metric, split và config lock. Model
thật/video thật vẫn là slow smoke/evaluation riêng vì cần model weights, dữ liệu và
manual visual inspection.

## 8. Kiểm tra submission

```powershell
.\.venv\Scripts\python.exe -m competition.pipeline validate-submission
```

Validator kiểm tra header, đúng thứ tự 100 query, đủ 100 answer/query, tên video,
giới hạn frame và số câu trả lời trùng chính xác. Report phải có `status: passed`;
lý tưởng là `exact_duplicate_answers: 0`.

## Lệnh đầy đủ theo thứ tự

```powershell
.\.venv\Scripts\python.exe -m competition.pipeline validate-input
.\.venv\Scripts\python.exe -m competition.pipeline keyframes --device cuda --resume
.\.venv\Scripts\python.exe -m competition.pipeline index
.\.venv\Scripts\python.exe -m competition.pipeline neighbors
.\.venv\Scripts\python.exe -m competition.pipeline segments
.\.venv\Scripts\python.exe -m competition.pipeline text-index
.\.venv\Scripts\python.exe -m competition.pipeline phase5-init --video-ids-file phase5_video_ids.txt --seed 42
.\.venv\Scripts\python.exe -m competition.pipeline phase5-evaluate --split dev
.\.venv\Scripts\python.exe -m competition.pipeline phase5-lock
.\.venv\Scripts\python.exe -m competition.pipeline phase5-evaluate --split validation
.\.venv\Scripts\python.exe -m competition.pipeline predict --device cuda
.\.venv\Scripts\python.exe -m competition.pipeline validate-submission
```

## Đường dẫn tùy chỉnh

Truyền cùng `--public-root` và `--output-root` cho mọi bước. `predict` mặc định ghi
vào `<output-root>/results/submission.csv`; có thể đặt `--submission-path` khác:

```powershell
.\.venv\Scripts\python.exe -m competition.pipeline predict `
  --public-root D:\dataset\public `
  --output-root D:\artifacts\competition `
  --submission-path D:\artifacts\competition\results\submission.csv `
  --device cuda
```

## 9. Chạy pipeline competition end-to-end (khuyến nghị)

Phần này là lệnh chạy thực tế cho pipeline hiện tại: kiểm tra input, trích xuất
keyframe đa phương thức, tạo index, retrieval, sinh submission và kiểm tra format.
Chạy các lệnh từ thư mục gốc của repository bằng PowerShell.

### Chuẩn bị

- Tạo môi trường `.venv` và cài dependency của dự án.
- Đảm bảo `ffmpeg` và `ffprobe` có trong `PATH`.
- Đặt dữ liệu ở `data/public`, gồm `corpus.csv`, `questions.csv`,
  `sample_submission.csv`, thư mục `videos` và thư mục `queries`.
- GPU CUDA được khuyến nghị. Có thể đổi `--device cuda` thành `--device cpu`,
  nhưng thời gian chạy sẽ lâu hơn đáng kể.

### Lệnh chạy đầy đủ

```powershell
& ".\.venv\Scripts\python.exe" -m competition.run_end_to_end `
  --public-root ".\data\public" `
  --output-root ".\competition" `
  --device cuda `
  --batch-size auto `
  --num-workers 0 `
  --candidate-interval-sec 0.5 `
  --max-gap-seconds 2.0 `
  --asr-timeout-seconds 90 `
  --asr-cpu-timeout-seconds 600 `
  --asr-retries 1 `
  --start-at validate-input
```

Pipeline chạy lần lượt:

```text
validate-input
  -> keyframes (candidate + SigLIP2 + caption + OCR + object + ASR + selection)
  -> index
  -> neighbors
  -> segments
  -> text-index
  -> predict
  -> validate-submission
```

Nếu chạy thành công, bước cuối phải báo submission hợp lệ. File nộp nằm tại:

```text
competition/results/submission.csv
```

### Resume khi lần chạy trước bị dừng

Runner mặc định truyền `--resume` vào bước keyframe. Các candidate và feature đã
hoàn thành được checkpoint theo từng video/modality, nên lần chạy sau sẽ bỏ qua
artifact hợp lệ và tiếp tục phần còn thiếu. Ví dụ tiếp tục từ keyframe:

```powershell
& ".\.venv\Scripts\python.exe" -m competition.run_end_to_end `
  --public-root ".\data\public" `
  --output-root ".\competition" `
  --device cuda `
  --batch-size auto `
  --num-workers 0 `
  --candidate-interval-sec 0.5 `
  --max-gap-seconds 2.0 `
  --asr-timeout-seconds 90 `
  --asr-cpu-timeout-seconds 600 `
  --asr-retries 1 `
  --start-at keyframes
```

ASR bị treo quá thời gian sẽ được retry; sau đó pipeline có thể thử CPU với timeout
dài hơn. Trên GPU khoảng 6 GB, nên giữ autocast mặc định và không thêm
`--no-autocast`. Chỉ dùng `--fresh` khi chủ động muốn không resume bước keyframe.

Để chạy đến một bước rồi dừng, dùng `--stop-after`. Ví dụ chỉ hoàn thành keyframe:

```powershell
& ".\.venv\Scripts\python.exe" -m competition.run_end_to_end `
  --public-root ".\data\public" `
  --output-root ".\competition" `
  --device cuda `
  --start-at keyframes `
  --stop-after keyframes
```

Để xem trước toàn bộ command mà không chạy model:

```powershell
& ".\.venv\Scripts\python.exe" -m competition.run_end_to_end --dry-run
```

### Artifact quan trọng

- `competition/work/keyframe_v3`: candidate, feature checkpoint, manifest và báo
  cáo chọn keyframe theo từng video.
- `competition/keyframes`: keyframe canonical dùng cho các bước downstream.
- `competition/metadata` và `competition/embeddings`: metadata/embedding đã chuẩn
  hóa cho retrieval.
- `competition/indexes`: FAISS index, text index và frame map.
- `competition/results/submission.csv`: submission cuối cùng.
- `reports/Experiment.md`: lịch sử cấu hình, artifact, metric nội bộ và leaderboard
  score của các lần chạy.

### Ghi Public/Private score vào báo cáo thí nghiệm

Runner tự thêm một snapshot vào `reports/Experiment.md` sau lần chạy thành công.
Khi đã có điểm leaderboard, cập nhật thêm một dòng bằng:

```powershell
& ".\.venv\Scripts\python.exe" -m competition.experiment_tracker `
  --public-root ".\data\public" `
  --output-root ".\competition" `
  --public-score 0.818 `
  --private-score 0.818 `
  --note "Baseline full multimodal, max_gap=2.0s"
```

### Kiểm tra riêng file submission

```powershell
& ".\.venv\Scripts\python.exe" -m competition.pipeline validate-submission `
  --public-root ".\data\public" `
  --submission-path ".\competition\results\submission.csv"
```

Chỉ dùng submission khi validator trả về `status: passed`. Nếu cần chế độ giảm tải
thiếu một số modality thì phải truyền rõ `--allow-partial-features`; không nên dùng
chế độ này cho lần chạy full multimodal cuối cùng.

## Retrieval/Leaderboard v2 (terminal-only)

Nhánh v2 giữ artifact 5 giây hiện tại làm baseline bất biến. Mọi submission mới,
query trace và manifest được ghi dưới `competition/runs/<run_id>`; API và frontend
không nằm trong luồng này.

### Runner kiến trúc v2 đầy đủ

Để build mới toàn bộ offline + coarse index + dense safety index + advanced
retrieval trong cùng một run root, dùng entrypoint sau. Đây là runner được dùng
bởi notebook Colab và không chạy Phase 5/ground-truth metric:

```powershell
& ".\.venv\Scripts\python.exe" -m competition.run_retrieval_v2 `
  --public-root ".\data\public" `
  --run-root ".\competition\runs\retrieval-v2-full" `
  --model-cache-root ".\data\model_cache" `
  --device cuda `
  --vlm-mode off
```

Runner chạy đúng chín stage:

```text
validate-input -> keyframes -> index -> neighbors -> segments -> text-index
  -> dense-index -> predict (advanced) -> validate-submission
```

Run dừng giữa chừng có thể tiếp tục bằng cùng `run_root` và `--start-at`. Runner
fail-closed nếu source code, public dataset hoặc offline config khác lineage đã
lưu. `reports/Experiment.md` chỉ được append sau khi manifest xác nhận đủ cả chín
stage; partial/failed/dry-run không được ghi thành một experiment hoàn tất.

Notebook launcher cho Colab Pro nằm tại
`notebooks/colab_retrieval_v2_launcher.ipynb`.

Ensemble hai submission đã validate bằng weighted RRF (không cần ground truth):

```powershell
& ".\.venv\Scripts\python.exe" -m competition.ensemble_submissions `
  --submission ".\competition\runs\run-a\results\submission.csv" `
  --weight 1.0 `
  --submission ".\competition\runs\run-b\results\submission.csv" `
  --weight 1.0 `
  --output ".\competition\runs\ensemble-ab\results\submission.csv" `
  --public-root ".\data\public"
```

Lệnh fail nếu union không đủ 100 answer duy nhất và ghi checksum/input lineage vào
`submission.csv.manifest.json` sau atomic replace.

Đăng ký baseline 5 giây và tạo `competition/runs/active_run.json` lần đầu:

```powershell
& ".\.venv\Scripts\python.exe" -m competition.pipeline init-run `
  --run-root competition\runs\baseline-5s-0818 `
  --baseline-source-root competition `
  --baseline-submission competition\results\submission.csv `
  --baseline-score 0.818 `
  --baseline-max-gap-seconds 5
```

Đóng gói 9.621 dense candidate đã cache, không chạy lại SigLIP2:

```powershell
& ".\.venv\Scripts\python.exe" -m competition.pipeline dense-index `
  --run-root competition\runs\retrieval-v2-baseline5s-dense `
  --source-workspace competition\work\keyframe_v3

& ".\.venv\Scripts\python.exe" -m competition.pipeline validate-dense-index `
  --run-root competition\runs\retrieval-v2-baseline5s-dense
```

Chạy retrieval nâng cao. `--dense-run-root` cho phép nhiều ablation dùng chung một
dense index bất biến, không nhân đôi vector/JPEG:

```powershell
& ".\.venv\Scripts\python.exe" -m competition.pipeline predict `
  --run-root competition\runs\retrieval-v2-deterministic `
  --dense-run-root competition\runs\retrieval-v2-baseline5s-dense `
  --retrieval-mode advanced `
  --coarse-top-n 50 `
  --dense-global-top-k 300 `
  --dense-frames-per-clip 12 `
  --vlm-mode off
```

Advanced mode dùng QueryPlan, weighted RRF, dense rescue, CSES và deterministic
multimodal rerank. Nó không padding bằng `video,frame 0`: nếu reserve không đủ 100
answer duy nhất, lệnh dừng trước khi atomic replace submission.

Chạy selector offline trên feature cache vào một run riêng, không ghi đè baseline
và không load lại model:

```powershell
& ".\.venv\Scripts\python.exe" -m competition.pipeline reselect-keyframes `
  --source-output-root competition `
  --run-root competition\runs\offline-gap2-dedup092 `
  --max-gap-seconds 2 `
  --dedup-similarity-threshold 0.92 `
  --asr-protection-threshold 0.80 `
  --endpoint-protection off
```

`--endpoint-protection on` fail-closed trước khi publish nếu cache nguồn chưa có
candidate first/last đã được encode. Khi đó phải tạo candidate cache có endpoint;
lệnh reselect không giả lập embedding hoặc silently dùng frame gần nhất.

Khởi tạo bộ 16 video để con người gán 80 event và 48 query:

```powershell
& ".\.venv\Scripts\python.exe" -m competition.optimize_retrieval `
  --init-labeling --seed 42
```

Sau khi hoàn tất `competition/evaluation/retrieval_labels.jsonl`, chạy ablation dev:

```powershell
& ".\.venv\Scripts\python.exe" -m competition.optimize_retrieval `
  --split dev `
  --experiment-config configs\retrieval_v2.yaml
```

Gắn score thật vào đúng SHA256 rồi chỉ promote khi score lớn hơn baseline:

```powershell
& ".\.venv\Scripts\python.exe" -m competition.pipeline record-score `
  --run-root competition\runs\retrieval-v2-deterministic `
  --score 0.825 --split public

& ".\.venv\Scripts\python.exe" -m competition.pipeline promote-run `
  --run-root competition\runs\retrieval-v2-deterministic `
  --minimum-score 0.818
```
