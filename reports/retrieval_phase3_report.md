# Bao cao Retrieval Phase 3

## 1. Ket luan nhanh

Retrieval Phase 3 khong can doi cac thanh vien khac de lam core service.
Phan co the lam doc lap da duoc trien khai tren nhanh `phase3`:

- Hybrid reranking cho top visual candidates.
- Temporal search cho query nhieu su kien theo thu tu thoi gian.
- API endpoint cho hybrid search va temporal search.
- Unit test pure-Python cho logic Phase 3.

Tuy nhien, chat luong retrieval that van phu thuoc cac artifact tu pipeline khac:

- FAISS index that: `data/indexes/openclip_vit_b16_flat_ip.faiss`.
- Frame map that: `data/metadata/openclip_vit_b16_frame_map.json`.
- Metadata caption, OCR, object, ASR neu muon hybrid score day du.

## 2. Pham vi da lam

### Hybrid rerank

File chinh:

- `backend/app/services/retrieval/rerank.py`

Logic:

```text
hybrid_score =
  0.45 * visual_score
+ 0.25 * caption_match
+ 0.15 * ocr_match
+ 0.10 * object_match
+ 0.05 * temporal_confidence
```

Reranker co the chay ngay ca khi chua co caption/OCR/object metadata. Modality nao thieu thi diem modality do bang 0, khong lam vo pipeline.

### Temporal search

File chinh:

- `backend/app/services/retrieval/temporal_search.py`

Da co:

- Tach query theo cac tu khoa `then`, `after that`, `next`, `followed by`.
- Xu ly dang query co `after`, vi du `talks to cashier after man enters shop`.
- Ghep candidates cung video theo timestamp tang dan.
- Gioi han khoang cach thoi gian bang `max_gap_seconds`.

### Hybrid orchestration

File chinh:

- `backend/app/services/retrieval/hybrid_search.py`

Pipeline:

```text
Stage 1: visual search top-500
Stage 2: hybrid rerank pool top-100
Stage 3: tra ve top-k hoac ghep temporal match
```

### API va entrypoint

File da cap nhat:

- `backend/app/services/retrieval/retrieval_manager.py`
- `backend/app/api/retrieval.py`

Endpoint moi:

- `POST /retrieval/hybrid`
- `POST /retrieval/temporal`

Wrapper Python moi:

- `search_hybrid(query, top_k)`
- `search_temporal(query, top_k)`

### Metadata support

File da cap nhat:

- `backend/app/services/metadata/metadata_store.py`
- `backend/app/services/retrieval/search_visual.py`

Frame map bay gio co the doc va propagate cac field:

- `caption`
- `ocr` hoac `ocr_text`
- `objects`

## 3. Test da chay

Da pass:

```bash
python -m unittest backend.tests.test_retrieval_phase3
python -m py_compile backend\app\services\metadata\metadata_store.py backend\app\services\retrieval\rerank.py backend\app\services\retrieval\temporal_search.py backend\app\services\retrieval\hybrid_search.py backend\app\services\retrieval\retrieval_manager.py backend\app\api\retrieval.py backend\tests\test_retrieval_phase3.py
```

Chua chay duoc test baseline cu trong moi truong hien tai:

```bash
python -m unittest backend.tests.test_search_visual
```

Ly do: Python environment hien tai thieu dependency `cv2`. Day la dependency moi truong, khong phai loi logic Phase 3.

## 4. Phan con can doi team khac

Retrieval Phase 3 code da co the merge/push, nhung de demo chat luong that can cac dau vao sau:

| Dau vao | Team/pham vi phu thuoc | Neu thieu thi sao |
|---|---|---|
| FAISS index | Indexing/Embedding | Khong chay visual search that |
| Frame map | Indexing/Metadata | Khong map duoc faiss index sang frame/video/timestamp |
| Caption | Caption pipeline | Caption score bang 0 |
| OCR | OCR pipeline | OCR score bang 0 |
| Objects | Object detection pipeline | Object score bang 0 |
| ASR | ASR pipeline | Chua duoc dua vao score Phase 3 hien tai |

## 5. Huong tiep theo

1. Khi co artifact that, chay `/retrieval/hybrid` tren subset 100-500 video.
2. Do Recall@10, Recall@50, latency cho CLIP-only va hybrid rerank.
3. Them ASR/text-index score vao reranker neu metadata ASR da on dinh.
4. Neu co VLM/LLM runtime, them expensive rerank cho top-20 hoac top-50.
