# Colab launcher cho Retrieval v2

Notebook [colab_retrieval_v2_launcher.ipynb](colab_retrieval_v2_launcher.ipynb)
chỉ làm nhiệm vụ orchestration. Notebook không sao chép hoặc định nghĩa lại thuật
toán retrieval.

Trước khi chạy:

1. Chọn Colab Pro GPU runtime.
2. Đưa checkout hiện tại và public dataset lên Google Drive, hoặc push branch
   `contest/RRF` rồi chọn `SOURCE_MODE = "git"`.
3. Sửa các biến trong cell `parameters`, đặc biệt là `DRIVE_REPO_PATH`,
   `PUBLIC_DATA_SOURCE`, `DRIVE_RUNS_ROOT`, `MODEL_CACHE_ROOT`, `RUN_ID` và
   `FUSION_MODE`. Giữ nguyên `RUN_ID` khi resume; đổi ID khi code/dataset/config
   thay đổi.
4. Chạy tuần tự toàn bộ cell.

Notebook gọi đúng một public entrypoint:

```bash
python -m competition.run_retrieval_v2 \
  --public-root /content/public_data \
  --run-root /content/drive/MyDrive/AIChallenge26/retrieval_runs/<run_id> \
  --device cuda \
  --fusion-mode adaptive_rrf \
  --retrieval-modalities visual,caption,ocr,objects,asr \
  --coarse-top-n 100 \
  --rrf-k 60 \
  --dense-expansion-before-sec 1 \
  --dense-expansion-after-sec 1 \
  --rerank-top-n 300 \
  --final-top-k 100 \
  --vlm-mode off
```

Khi thành công, run root có `run_manifest.json`, offline/coarse/dense artifacts,
query trace và `results/submission.csv`. `Experiment.md` chỉ được append sau khi
cả chín stage và strict submission validator đều pass. Ground-truth/quality
metric không được chạy hoặc giả lập trong notebook này.

Notebook mặc định chạy `adaptive_rrf`. Có thể đổi `FUSION_MODE` thành `legacy`,
`standard_rrf` hoặc `weighted_rrf` để chạy ablation mà không thay indexing hay
keyframe extraction. Query trace được kiểm tra để bảo đảm modality weights, RRF
contribution và dense recovery thực sự xuất hiện trong run Colab.
