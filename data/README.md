# Dữ liệu và artifact runtime

Mọi dataset, model cache và file sinh trong lúc chạy đều nằm dưới `data/`:

```text
data/
  raw/video/       video gốc
  keyframes/       ảnh keyframe đã trích xuất
  metadata/        frame map, caption, OCR, object và segment metadata
  embeddings/      vector trung gian
  indexes/         FAISS, BM25 và manifest chỉ mục
  model_cache/     checkpoint tải từ model hub
  cache/           cache suy luận, gồm grounded QA
  submissions/     CSV KIS/QA đã xuất
  reports/         báo cáo chạy và kiểm tra lineage
```

Luồng dữ liệu canonical:

```text
data/raw/video
  -> data/keyframes + data/metadata
  -> data/embeddings + data/indexes
  -> retrieval KIS hoặc grounded QA
  -> data/submissions (chỉ khi chủ động lưu CSV)
```

`frame_id` trong CSV submission là `frame_index` nguyên, không âm của frame trong
video gốc. Nó không phải thứ tự keyframe, timestamp, tên JPEG hay hàng FAISS.

Các thư mục artifact được giữ bằng `.gitkeep`; dataset, checkpoint, cache, index,
report và submission sinh tự động đều bị bỏ qua bởi Git. Nếu ban tổ chức cung cấp
`data/sample_submission.csv`, header và quy tắc định danh của file đó phải được cập
nhật thành nguồn sự thật cao nhất trước khi nộp bài.
