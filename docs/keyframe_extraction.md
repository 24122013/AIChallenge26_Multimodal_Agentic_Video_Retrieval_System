# Keyframe Extraction Cho Vòng Thi

## Chiến lược khuyến nghị

Pipeline chính thức nên chạy theo thứ tự:

1. Detect shot bằng TransNetV2.
2. Chọn keyframe theo độ dài shot:
   - Shot `duration <= 2s`: lấy 1 frame ở midpoint.
   - Shot `2s < duration <= 4s`: lấy 2 frame tại 1/3 và 2/3 shot.
   - Shot `duration > 4s`: lấy 1 frame mỗi 2 giây theo centered sampling (`start+1s`, `+3s`, ...).
3. Conservative dedup chỉ so sánh các frame cùng shot và gần nhau tối đa 2 giây.
4. pHash mặc định chỉ bỏ frame có Hamming distance <= 6; không dedup cross-shot.
5. Nếu bật CLIP dedup, điều kiện cùng shot và cửa sổ thời gian vẫn được giữ nguyên.
6. Lưu timestamp từ `frame_index / fps`, kèm `shot_start`, `shot_end`, `shot_id`.
7. Khi search, mỗi kết quả trả thêm các keyframe lân cận cùng `shot_id`.

Đây là cấu hình ưu tiên recall: chấp nhận giữ nhiều keyframe hơn để giảm nguy cơ bỏ sót moment trước bước retrieval.

## Tham số ablation

- `--short-shot-max-sec`: ngưỡng shot ngắn, mặc định `2.0`.
- `--regular-shot-max-sec`: ngưỡng shot thường, mặc định `4.0`.
- `--long-shot-interval-sec`: chu kỳ centered sampling, mặc định `2.0`.
- `--phash-threshold`: Hamming distance tối đa để coi là gần như trùng, mặc định `6`.
- `--phash-window-sec`: cửa sổ thời gian within-shot, mặc định `2.0`.
- `--clip-similarity-threshold`: ngưỡng CLIP tùy chọn, mặc định `0.985`.
- `--clip-window-sec`: cửa sổ thời gian CLIP within-shot, mặc định `2.0`.

## Lệnh chạy một video

TransNetV2 PyTorch cần `ffmpeg.exe` trong `PATH`. Kiểm tra trước bằng:

```powershell
ffmpeg -version
```

```powershell
.\.venv\Scripts\python.exe -B backend\app\services\indexing\extract_keyframes.py `
  --video-path data\raw\video\L27_V001.mp4 `
  --output-dir data\keyframes
```

Output mặc định:

- Ảnh keyframe: `data/keyframes/<video_id>/`
- Metadata JSONL: `data/metadata/keyframes_<video_id>.jsonl`
- Report: `data/metadata/keyframes_<video_id>_extract_report.json`

## Lệnh chạy toàn bộ folder video

```powershell
.\.venv\Scripts\python.exe -B backend\app\services\indexing\extract_keyframes.py `
  --video-dir data\raw\video `
  --video-glob *.mp4 `
  --output-dir data\keyframes
```

## Bật CLIP dedup nếu có thời gian

```powershell
.\.venv\Scripts\python.exe -B backend\app\services\indexing\extract_keyframes.py `
  --video-dir data\raw\video `
  --enable-clip-dedup `
  --clip-similarity-threshold 0.985 `
  --clip-window-sec 2
```

Chỉ nên bật sau khi đã có bản pHash chạy ổn, vì CLIP dedup tốn thời gian hơn. Nếu dataset nhiều cảnh tĩnh hoặc camera ít đổi, CLIP dedup giúp giảm nhiễu kết quả.

## Metadata quan trọng

Mỗi dòng JSONL có các field cần cho retrieval và UI:

- `frame_id`, `video_id`, `shot_id`, `segment_id`
- `timestamp`, `timestamp_source`, `timestamp_confidence`
- `frame_index`, `shot_start`, `shot_end`, `shot_index`
- `keyframe_path`, `frame_path`, `thumbnail_path`
- `keyframe_strategy`, `selection_reason`, `phash`, `shot_detector`

Sau khi build FAISS/frame_map từ metadata này, search response sẽ có thêm `neighbors`: các frame cùng shot để UI hiển thị ngữ cảnh trước/sau kết quả chính.
