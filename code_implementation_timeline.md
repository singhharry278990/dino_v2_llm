# DINOv2 Code Implementation Timeline

This timeline is based on current file modification timestamps in `dinov2/*.py`.

## Timestamp-Ordered Changes

1. `2026-05-04 14:07:37` - `predict_dino.py`
   - Inference utility for single and batch prediction.
   - Reconstructs model from checkpoint payload and returns top-k species probabilities.

2. `2026-05-05 11:00:22` - `build_dinov2_faiss_index.py`
   - Added embedding extraction pipeline from DINOv2 backbone.
   - Builds cosine-similarity FAISS index and aligned metadata/manifest files.

3. `2026-05-05 11:05:13` - `dinov2_faiss_server.py`
   - Added Flask API for similarity search over FAISS vectors.
   - Serves `/query` endpoint and `/ui` page integration.

4. `2026-05-05 13:32:25` - `remove_bg.py`
   - Utility script update (image preprocessing support script).

5. `2026-05-05 13:32:25` - `save_code.py`
   - Utility script update (project support script).

6. `2026-05-06 00:09:15` - `train_dino_model.py`
   - Latest training implementation update.
   - Two-phase fine-tuning with balanced per-class batching and test/confusion-matrix reporting.

7. `2026-05-06 23:34:00` - `train_dino_model.py`, `../efficient_v2/train_efficientnetv2.py`
   - Added hard negative mining via batch-hard triplet loss on normalized embeddings.
   - Training loss updated to `CrossEntropy + HNM_WEIGHT * TripletLoss` (train only).
   - Replaced `torch.cdist` pairwise distance with cosine-distance matrix (`1 - emb @ emb.T`) for MPS-safe backward pass.

## Current Implementation Snapshot

Primary production flow:
- Train classifier: `train_dino_model.py`
- Predict classes: `predict_dino.py`
- Build retrieval index: `build_dinov2_faiss_index.py`
- Serve retrieval API/UI: `dinov2_faiss_server.py`

Supporting utilities:
- `remove_bg.py`
- `save_code.py`
