# Keyframe Extraction Cho Vòng Thi

## Chiến lược khuyến nghị

Pipeline chính thức nên chạy theo thứ tự:

1. Detect shot bằng TransNetV2.
2. Chọn keyframe theo độ dài shot:
   - Shot < 4 giây: lấy 1 frame ở midpoint.
   - Shot 4-8 giây: lấy 2 frame tại khoảng 1/3 và 2/3 shot.
   - Shot > 8 giây: lấy frame mỗi 4 giây, bắt đầu gần giây thứ 2 của shot.
3. Tránh frame sát biên shot bằng margin nhỏ để giảm blur/chuyển cảnh.
4. Dedup trong cùng video bằng pHash, mặc định bỏ frame có Hamming distance <= 6.
5. Nếu có thời gian hoặc GPU, bật CLIP dedup gần nhau theo thời gian để bỏ các frame giống ngữ nghĩa.
6. Lưu timestamp từ `frame_index / fps`, kèm `shot_start`, `shot_end`, `shot_id`.
7. Khi search, mỗi kết quả trả thêm các keyframe lân cận cùng `shot_id`.

Đây là cấu hình cân bằng tốt cho bài thi retrieval: không lấy quá thưa làm mất recall, cũng không lấy quá dày làm FAISS/index phình và nhiều kết quả trùng.

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
  --clip-window-sec 12
```

Chỉ nên bật sau khi đã có bản pHash chạy ổn, vì CLIP dedup tốn thời gian hơn. Nếu dataset nhiều cảnh tĩnh hoặc camera ít đổi, CLIP dedup giúp giảm nhiễu kết quả.

## Metadata quan trọng

Mỗi dòng JSONL có các field cần cho retrieval và UI:

- `frame_id`, `video_id`, `shot_id`, `segment_id`
- `timestamp`, `timestamp_source`, `timestamp_confidence`
- `frame_index`, `shot_start`, `shot_end`, `shot_index`
- `keyframe_path`, `frame_path`, `thumbnail_path`
- `selection_reason`, `phash`, `shot_detector`

Sau khi build FAISS/frame_map từ metadata này, search response sẽ có thêm `neighbors`: các frame cùng shot để UI hiển thị ngữ cảnh trước/sau kết quả chính.
