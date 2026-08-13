# Colab launcher cho Retrieval v2

Notebook [colab_retrieval_v2_launcher.ipynb](colab_retrieval_v2_launcher.ipynb)
chỉ làm nhiệm vụ orchestration. Notebook không sao chép hoặc định nghĩa lại thuật
toán retrieval.

Trước khi chạy:

1. Chọn Colab Pro GPU runtime.
2. Đưa checkout hiện tại và public dataset lên Google Drive, hoặc push branch
   `codex/retrieval-leaderboard-v2` rồi chọn `SOURCE_MODE = "git"`.
3. Sửa các biến trong cell `parameters`, đặc biệt là `DRIVE_REPO_PATH`,
   `PUBLIC_DATA_SOURCE`, `DRIVE_RUNS_ROOT` và `MODEL_CACHE_ROOT`.
4. Chạy tuần tự toàn bộ cell.

Notebook gọi đúng một public entrypoint:

```bash
python -m competition.run_retrieval_v2 \
  --public-root /content/public_data \
  --run-root /content/drive/MyDrive/AIChallenge26/retrieval_runs/<run_id> \
  --device cuda \
  --vlm-mode off
```

Khi thành công, run root có `run_manifest.json`, offline/coarse/dense artifacts,
query trace và `results/submission.csv`. `Experiment.md` chỉ được append sau khi
cả chín stage và strict submission validator đều pass. Ground-truth/quality
metric không được chạy hoặc giả lập trong notebook này.
