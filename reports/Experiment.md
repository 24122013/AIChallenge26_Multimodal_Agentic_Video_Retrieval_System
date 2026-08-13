# Competition Experiments

Lịch sử benchmark append-only cho pipeline TKIS/VKIS. Scoreboard chỉ được ghi
khi có số liệu được cung cấp rõ ràng; metric local được thu từ artifact và không
được xem là ground-truth retrieval quality.

## Best scoreboard

<!-- scoreboard-summary:start -->
| Scoreboard metric | Best score | Experiment |
|---|---:|---|
| Public | 0.818000 | EXP-20260809-SCOREBOARD-001 |
| Private | 0.818000 | EXP-20260809-SCOREBOARD-001 |
<!-- scoreboard-summary:end -->

## Metric definitions

- Public/Private: điểm do nền tảng chấm, phải được nhập rõ ràng sau khi submit.
- Selection ratio: số keyframe cuối chia cho số dense candidate.
- Hard-constraint pass rate: tỷ lệ video thỏa event, shot và temporal guarantees.
- Detected-event coverage: recall trên event mà feature adapter thực sự phát hiện,
  không phải recall ground-truth do người gán nhãn.
- Submission SHA256: liên kết một scoreboard result với đúng file submission local.

## Experiment log

<!-- experiment-log:append-below -->

## EXP-20260809-SCOREBOARD-001

- Recorded at: `2026-08-09T00:00:00+00:00`
- Source: `user_reported_scoreboard_baseline`
- Status: `completed`
- Public score: `0.818000`
- Private score: `0.818000`
- Note: Điểm do user cung cấp. Snapshot local được gắn bằng submission SHA256;
  chưa có API của nền tảng để xác minh tự động quan hệ giữa score và file này.

### Artifact-backed metrics

| Metric | Value |
|---|---:|
| Dataset videos | 250 |
| Dataset queries | 100 |
| Query tasks | TKIS=50, VKIS=50 |
| Candidate-pool videos | 250 |
| Dense candidates | 9621 |
| SigLIP2 videos | 250 |
| Caption videos | 250 |
| OCR videos | 250 |
| Object videos | 250 |
| ASR videos | 250 |
| Complete feature manifests | 250 |
| Published videos | 250 |
| Selected keyframes | 1533 |
| Selection ratio | 0.159339 |
| Hard-constraint pass rate | 1.000000 |
| Detected-event coverage pass rate | 1.000000 |
| Observed max temporal gap (s) | 5.000000 |
| Submission SHA256 | `b9f4bd41cf51745def824e9fd6d4665b8c96663a36b50f1e33e9d234f331d8c6` |
| Submission query count | 100 |
| Answers per query | 100 |
| FAISS vectors | 1533 |
| FAISS index size (MiB) | 6.737 |
| FAISS build runtime (s) | 0.336 |

<!-- experiment-json:{"experiment_id":"EXP-20260809-SCOREBOARD-001","recorded_at":"2026-08-09T00:00:00+00:00","source":"user_reported_scoreboard_baseline","status":"completed","public_score":0.818,"private_score":0.818,"note":"User-reported score; platform API verification unavailable.","local_metrics":{"dataset":{"video_count":250,"query_count":100,"tasks":{"TKIS":50,"VKIS":50}},"workspace":{"candidate_pool_video_count":250,"candidate_count":9621,"siglip2_video_count":250,"caption_video_count":250,"ocr_video_count":250,"object_video_count":250,"asr_video_count":250,"feature_manifest_count":250},"canonical":{"published_video_count":250,"candidate_count":9621,"selected_keyframe_count":1533,"selection_ratio":0.15933894605550358,"constraint_pass_rate":1.0,"detected_event_coverage_pass_rate":1.0,"observed_max_gap_seconds":5.000000000000002},"submission":{"exists":true,"sha256":"b9f4bd41cf51745def824e9fd6d4665b8c96663a36b50f1e33e9d234f331d8c6","size_bytes":198787,"query_count":100,"answers_per_query":100},"index":{"exists":true,"index_type":"IndexFlatIP","metric":"ip","vector_count":1533,"runtime_sec":0.336,"index_file_size_mb":6.737},"phase5":{}}} -->
