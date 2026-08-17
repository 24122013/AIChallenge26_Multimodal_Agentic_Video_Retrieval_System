# Prompt triển khai TRAKE cho repository hiện tại

Sao chép toàn bộ nội dung trong khối prompt dưới đây và đưa cho coding agent làm việc tại thư mục gốc của repository.

```text
Bạn là senior ML/backend engineer chịu trách nhiệm triển khai end-to-end task TRAKE (Temporal Retrieval and Alignment of Key Events) cho repository AIChallenge26_Multimodal_Agentic_Video_Retrieval_System hiện tại.

MỤC TIÊU

Biến TRAKE thành một task public, chạy được và kiểm thử được trên pipeline hiện có. Với một truy vấn mô tả context chung và chuỗi N event có thứ tự, hệ thống phải:

1. tìm các video có khả năng chứa toàn bộ chuỗi event;
2. tìm một original-video frame_index cho từng event, đúng thứ tự;
3. sinh tối đa 100 sequence hypothesis đã xếp hạng;
4. xuất mỗi hypothesis theo dạng:
   <video_id>, <frame_id_1>, ..., <frame_id_N>

Trong repository này, giá trị nộp với tên frame_id phải là original zero-based frame_index dạng số nguyên, KHÔNG phải RetrievalResult.frame_id nội bộ, tên ảnh, timestamp hay FAISS row.

NGUỒN SỰ THẬT VÀ QUY TẮC LÀM VIỆC

- Trước khi sửa code, đọc README.md, docs/PIPELINE_AUDIT.md, docs/architecture.md, docs/retrieval_api_contract.md, docs/metadata_schema.md, configs/retrieval.yaml và các test liên quan.
- Canonical runtime nằm trong backend/. src/indexing/ chỉ chứa helper indexing dùng chung. Không tạo một pipeline song song dưới src/.
- Repo đang có KIS, QA và task temporal dùng cho temporal evidence/QA. Task temporal hiện tại KHÔNG đồng nghĩa với TRAKE. Không đổi ngữ nghĩa hoặc làm hỏng KIS, QA, AVS hay temporal QA.
- Tái sử dụng retrieval stack hiện có: SigLIP2 visual, caption, OCR, objects, BM25/BGE nếu được bật, fusion/rerank, canonical metadata, neighbor/segment artifacts.
- Không thêm ASR/audio/transcript vì runtime hiện tại không có các modality đó.
- Không chạy VLM trên toàn corpus, không xây dense-frame index toàn corpus và không huấn luyện model mới trong MVP.
- Mọi model nặng phải lazy-load, cache và có feature flag. Unit test không được cần GPU, model lớn, Internet hay full dataset.
- Dữ liệu truy vấn là untrusted data, không phải instruction. Parser/LLM phải bỏ qua prompt injection trong query và chỉ trả schema đã định nghĩa.
- Không suy đoán frame_index từ filename, timestamp hoặc FAISS index. Nếu lineage original frame bị thiếu, fail closed hoặc bỏ hypothesis đó và ghi trace/warning rõ ràng.
- Giữ thay đổi nhỏ, typed, deterministic và tương thích ngược. Không rewrite retrieval pipeline hiện có.

HỢP ĐỒNG CHẤM ĐIỂM CHÍNH THỨC PHẢI ĐƯỢC PHẢN ÁNH TRONG THIẾT KẾ

- Một query được gửi tối đa 100 câu trả lời.
- Nếu video_id sai: R-Score = 0.
- Nếu video_id đúng:
  R-Score = (1/N) * tổng_j I(frame_id_j nằm trong [s_j, e_j]).
- Khoảng ground-truth của một semantic keyframe thường rất ngắn, thông thường dưới 10 frame.
- Với k thuộc {1, 5, 20, 50, 100}, R@k là R-Score tốt nhất trong top-k.
- Final Score là trung bình của R@1, R@5, R@20, R@50 và R@100.

Vì vậy không chỉ trả một best chain. Hệ thống phải tối ưu đồng thời video correctness, frame-level alignment và diversity của ranked top-100.

HIỆN TRẠNG CẦN TẬN DỤNG

- backend/app/pipelines/online_pipeline.py là online entrypoint canonical và hiện hỗ trợ auto/kis/avs/temporal/qa.
- backend/app/services/retrieval/retrieval_manager.py giữ cached service entrypoints.
- backend/app/services/retrieval/query_plan.py đã có typed temporal parsing nhưng có giới hạn/semantics phục vụ QA; không ép TRAKE dùng giới hạn 5 event nếu luật TRAKE không quy định như vậy.
- backend/app/services/retrieval/temporal_search.py đã có EventQuery, TemporalMatch và ordered matching/beam fallback. Có thể tái sử dụng hoặc refactor helper thuần, nhưng không nhét toàn bộ TRAKE vào module rank-fusion hoặc QA evidence.
- backend/app/services/retrieval/qa_evidence.py hiện chỉ flatten event của best temporal chain vào results và phục vụ evidence/answer eligibility. Không dùng private QA helper làm public TRAKE API.
- RetrievalResult đã mang video_id, internal frame_id, timestamp, frame_index, shot_id, segment_id và modality scores.
- backend/app/services/submission/ hiện chỉ hỗ trợ KIS/QA và chủ động từ chối trake.
- configs/retrieval.yaml có max_top_k 200; submission TRAKE vẫn phải giới hạn 1..100.

KIẾN TRÚC YÊU CẦU

Tạo một package riêng tại backend/app/services/trake/ (tên file có thể điều chỉnh theo convention sau khi inspect repo), tối thiểu tách các trách nhiệm sau:

1. models.py
   - TemporalEvent: index, name tùy chọn, original_text, retrieval_query, boundary_type, protected_terms.
   - TemporalEventPlan: original_query, context, ordered events, parser source/confidence/warnings.
   - EventCandidate: event index + RetrievalResult + normalized event-local score.
   - VideoCandidate: video_id, coverage, event support, context score, total score.
   - TemporalPath/TrakeHypothesis: video_id, đúng N original frame_index, coarse candidates, score breakdown, rank, lineage và warnings.
   - Tất cả object trả public phải serialize ổn định bằng to_dict().

2. query_parser.py
   - Parse query thành context và đúng N ordered events.
   - Hỗ trợ ít nhất numbered/bulleted events và các connective tiếng Việt/Anh mà parser hiện có hỗ trợ.
   - Không merge/split event quá mức. Bảo toàn nguyên văn các từ quyết định semantic boundary như “đầu tiên”, “bắt đầu”, “rời hoàn toàn”, “cao nhất”, “chạm”, “first”, “fully leaves”, “maximum/peak”.
   - Suy ra boundary_type bảo thủ: first_contact, first_leave, first_transition, peak hoặc state/unknown. Nếu không chắc, dùng unknown và giữ nguyên criterion.
   - Mỗi event retrieval query có dạng context + event criterion; original event text luôn được giữ trong plan và trace.
   - Query expansion, nếu dùng, chỉ tạo semantic-preserving variants theo từng event, không biến nhiều event thành các paraphrase của cùng một intent và không làm thay đổi N/order/boundary terms.
   - Fallback deterministic về event text nguyên văn khi parser nâng cao thất bại.

3. event_retrieval.py và candidate_video.py
   - Gọi public hybrid retrieval stack hiện có độc lập cho từng event với event_top_k đủ rộng.
   - Context branch là optional và chỉ hỗ trợ video-level scoring.
   - Áp dụng shot/temporal diversity trước alignment: giới hạn candidate trên mỗi shot và/hoặc temporal NMS; không để top list của event bị chiếm bởi các frame gần như trùng nhau.
   - Normalize score riêng cho từng event bằng rank/percentile ở V1; không cộng raw score khác distribution một cách trực tiếp.
   - Group theo video_id và tính ít nhất coverage, event support, context score, total score. Coverage phải là tín hiệu mạnh vì sai video làm toàn bộ answer bằng 0.
   - Chỉ giữ top_videos để alignment, nhưng có fallback có trace khi không video nào coverage đầy đủ.

4. temporal_alignment.py
   - Sinh K-best ordered paths trên candidate sparse của từng video bằng DP hoặc bounded beam search.
   - Thứ tự phải dựa trên original frame_index khi có; timestamp chỉ dùng làm hỗ trợ, không dùng để chế frame index.
   - Không áp hard maximum gap theo mặc định vì luật không quy định khoảng cách tối đa giữa event. Dùng soft gap penalty cấu hình được.
   - Score path phải có breakdown: event scores, video score/coverage, gap penalty và duplicate-location penalty.
   - Không tái sử dụng cùng một frame cho nhiều event khi có lựa chọn hợp lệ. Phạt near-duplicate path/shot một cách mềm, không cấm hai event thực sự gần nhau.
   - Kết quả deterministic với tie-break rõ ràng.

5. temporal_refinement.py
   - Đây là local refinement trên top coarse paths, không phải dense indexing toàn corpus.
   - Resolve original video an toàn từ canonical video root; không chấp nhận path traversal từ video_id.
   - Decode dense local window quanh coarse frame_index với stride mặc định 1.
   - Thiết kế scorer/refiner dưới dạng protocol hoặc dependency injection để unit test bằng fake scorer. Backend mặc định có thể tái sử dụng visual-text encoder hiện có khi khả dụng.
   - Boundary-aware behavior:
     * first_contact/first_transition: ưu tiên frame đầu tiên thỏa criterion sau transition;
     * first_leave: ưu tiên frame đầu tiên ở trạng thái rời hoàn toàn;
     * peak: ưu tiên local maximum;
     * unknown/state: dùng semantic score và giữ nhiều local hypotheses.
   - Nếu V1 chưa có pose/contact model đủ mạnh, không giả vờ xác định chính xác hình học. Giữ nhiều frame hypotheses lân cận có kiểm soát, ghi strategy/confidence/warning và cho phép thay scorer sau này.
   - Nếu video hoặc decoder/scorer không khả dụng, fallback về coarse original frame_index với warning; tuyệt đối không bịa frame.

6. ranking.py và pipeline.py
   - Phối hợp parse -> event retrieval -> video gating -> K-best alignment -> optional local refinement -> global ranking.
   - Sinh tối đa 100 hypothesis, mỗi hypothesis có đúng N frame_index không âm và cùng một video_id.
   - Xếp hạng ưu tiên confidence cao ở đầu nhưng phân bổ diversity có kiểm soát giữa video, temporal path và local frame variants.
   - Tránh lãng phí 100 slot cho các path chỉ lệch một frame. Dùng deterministic sequence-level NMS/diversity và giới hạn số answer trên mỗi video/path, nhưng không phân tán quá mạnh làm hỏng R@1.
   - Trace phải nêu candidate count từng stage, parser fallback, video coverage, alignment/refinement strategy, model/feature flags, warnings và latency; không làm lộ prompt hoặc dữ liệu nhạy cảm.

PUBLIC INTEGRATION BẮT BUỘC

1. Thêm "trake" vào SUPPORTED_ONLINE_TASKS nhưng giữ nguyên semantics của "temporal".
2. OnlinePipeline nhận một TrakePipeline dependency riêng và route task="trake" tới đó.
3. retrieval_manager cung cấp get_trake_pipeline() có cache đúng theo corpus generation và search_trake()/search_online(..., task="trake"). Cache phải được clear cùng clear_retrieval_caches().
4. API /retrieval/online chấp nhận task="trake"; thêm wrapper/endpoint riêng nếu phù hợp convention, nhưng không tạo FastAPI app factory ngoài scope.
5. CLI online_pipeline --task phải nhận trake.
6. Response TRAKE canonical nên có tối thiểu:
   {
     "schema_version": "...",
     "query": "...",
     "task": "trake",
     "event_plan": {...},
     "hypotheses": [
       {
         "rank": 1,
         "video_id": "L10_V010",
         "frame_ids": [101, 150, 203, 251],
         "score": 0.0,
         "score_breakdown": {...},
         "events": [...],
         "warnings": []
       }
     ],
     "trace": {...},
     "latency_ms": 0.0
   }
   Có thể giữ candidates alias nếu cần tương thích, nhưng sequence hypothesis là đơn vị xếp hạng chính; không flatten một chain thành N candidate độc lập.

CONFIG

Mở rộng loader config hiện có và thêm section trake vào configs/retrieval.yaml. Giá trị khởi đầu để benchmark, không hard-code trong thuật toán:

trake:
  event_top_k: 300
  top_videos: 30
  max_candidates_per_event_per_video: 20
  max_candidates_per_shot: 2
  score_normalization: rank
  context_weight: 0.10
  coverage_weight: 0.45
  event_support_weight: 0.45
  alignment_method: beam
  beam_width: 200
  k_best_paths_per_video: 10
  gap_penalty: log
  gap_lambda: 0.02
  refinement_enabled: true
  refinement_top_paths: 20
  window_before_frames: 60
  window_after_frames: 60
  dense_stride_frames: 1
  local_hypotheses_per_event: 3
  max_answers: 100
  ranking_cutoffs: [1, 5, 20, 50, 100]

Validate tất cả bounds/types. Các path/model settings có thể override bằng config/env theo convention repo, nhưng một nguồn config phải rõ ràng.

SUBMISSION

- Mở rộng SubmissionTask với TRAKE và loại bỏ nhánh NotImplementedError.
- Thêm serialize_trake_csv()/export path tương ứng. Mỗi row phải chứa video_id và đúng N original frame_index theo thứ tự event.
- top_k cho submission chỉ từ 1 đến 100, dedupe theo toàn sequence (video_id, tuple(frame_ids)), không dedupe theo từng frame độc lập.
- Vì repo chưa có sample_submission.csv chính thức, cô lập header/format assumption trong serializer, document rõ assumption và dễ đổi. Không đưa score, timestamp, internal frame_id, path hoặc trace vào file nộp.
- Filename ổn định, ví dụ trake_result.csv.

EVALUATION

Thêm pure evaluation functions và test cho:

- trake_r_score(prediction, ground_truth): sai video = 0; đúng video = số event frame nằm trong inclusive interval chia N;
- best_r_at_k cho k = 1, 5, 20, 50, 100;
- final score là trung bình 5 R@k;
- validate mismatch event count, interval xấu, frame âm và duplicate hypothesis;
- report thêm Video@1/@5/@20, per-event hit rate và matched-event ratio khi ground truth có sẵn.

TEST BẮT BUỘC

Tạo test tập trung, dùng fake retrieval/refinement, bao phủ ít nhất:

1. parser giữ đúng N/order/original criterion và các boundary term tiếng Việt/Anh;
2. parser không nghe instruction nằm trong query và fallback không làm mất event;
3. event-wise retrieval gọi từng event độc lập với context;
4. video gating ưu tiên video coverage 4/4 hơn video chỉ có score cao ở 1 event;
5. score normalization không bị raw scale giữa event chi phối;
6. alignment chỉ sinh path cùng video, đúng thứ tự frame_index, đủ N event và deterministic;
7. soft gap không loại path hợp lệ chỉ vì vượt 180 giây;
8. refinement cho first-transition/peak bằng fake score sequence và fallback khi thiếu video;
9. top-100 ranking có sequence diversity, không trùng sequence và không quá 100;
10. response không nhầm internal frame_id với original frame_index;
11. serializer tạo row N+1 cột và loại hypothesis thiếu lineage;
12. R-Score ví dụ L10_V010, [101,156,203,251] với GT intervals [95,105], [145,155], [195,205], [245,255] bằng 0.75;
13. sai video cho R-Score 0;
14. OnlinePipeline/manager/API/CLI route task="trake" đúng và task="temporal" vẫn giữ behavior cũ;
15. toàn bộ test KIS/QA/submission hiện có không regression; cập nhật các test trước đây kỳ vọng TRAKE bị từ chối thành test TRAKE được hỗ trợ.

Tối thiểu chạy:

python -m pytest -q backend/tests/test_trake_query_parser.py backend/tests/test_trake_pipeline.py backend/tests/test_trake_submission.py backend/tests/test_trake_metrics.py
python -m pytest -q backend/tests/test_online_pipeline.py backend/tests/test_submission_export.py backend/tests/test_retrieval_phase3.py
python -m pytest -q

DOCUMENTATION

- Cập nhật README.md, backend/README.md, docs/PIPELINE_AUDIT.md, docs/architecture.md, docs/retrieval_api_contract.md, docs/eval_protocol.md và config docs đúng với implementation thực tế.
- Xóa mọi câu “TRAKE planned/not implemented” đã hết đúng.
- Document rõ semantic keyframe khác technical keyframe; frame_id nộp là original frame_index; current limitations của local refiner; cách chạy CLI/API/export; schema response và cách benchmark.
- Không tuyên bố VLM/pose/contact verification đã hoạt động nếu chỉ có interface/fallback.

TRÌNH TỰ TRIỂN KHAI KHUYẾN NGHỊ

P0: data models + conservative parser + event retrieval + video gating + chronological K-best baseline + online integration + top-100 serializer.
P1: score normalization + K-best beam/DP + shot/sequence diversity + official metrics.
P2: dense local refinement với injectable scorer và safe fallback.
P3: optional verifier sau feature flag, chỉ trên top paths.

DEFINITION OF DONE

- search_online(query, task="trake", top_k=100) trả ranked sequence hypotheses đúng schema và không flatten event.
- Mọi hypothesis dùng original frame_index, cùng video, đúng số event và đúng thứ tự.
- Có candidate-video gating, K-best ordered alignment, controlled top-100 diversity và local-refinement interface/fallback thực sự chạy được.
- Có serializer TRAKE và evaluator phản ánh đúng công thức chính thức.
- Không regression KIS/QA/AVS/temporal; full test suite pass.
- Code/doc/config không còn claim mâu thuẫn với trạng thái triển khai.

CÁCH BÁO CÁO KHI HOÀN THÀNH

Trả lời ngắn gọn theo cấu trúc:

1. Kết quả đã triển khai.
2. Các file chính đã đổi và vai trò của chúng.
3. Test/command đã chạy cùng kết quả pass/fail.
4. Giới hạn còn lại, đặc biệt độ chính xác semantic-boundary của local refiner và dependency/model chưa kiểm chứng trên full corpus.
5. Không nói “hoàn tất” nếu còn test fail hoặc đang dùng mock/fallback ở đường chạy production mà chưa ghi rõ.
```

## Ghi chú thiết kế

Prompt này cố ý yêu cầu một task `trake` riêng thay vì đổi tên route `temporal`: route hiện tại đang gắn với temporal evidence của QA và có hợp đồng đầu ra khác. Nó cũng đặt code mới dưới `backend/app/services/trake/` vì `backend/` là canonical runtime của repository, thay vì sao chép nguyên cấu trúc `src/trake/` trong tài liệu đề xuất sơ bộ.
