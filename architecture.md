# DINOv2 System Architecture

## Overview

This project uses a transfer-learning pipeline built around `dinov2_vits14` for herbarium species classification, then reuses the same backbone embeddings for FAISS similarity search.

Core code modules:
- `train_dino_model.py`: model training and evaluation.
- `predict_dino.py`: inference for single image or batch.
- `build_dinov2_faiss_index.py`: embedding extraction + FAISS index build.
- `dinov2_faiss_server.py`: HTTP API and UI serving for image similarity search.

## Model Architecture

Training and inference share the same `Model` definition:
- Backbone: DINOv2 ViT-S/14 loaded from `facebookresearch/dinov2`.
- Head: `Dropout(0.3) -> Linear(embed_dim=384, num_classes)`.
- Forward output: `(feature_embedding, logits)`.

Fine-tuning strategy:
- Phase 1: freeze all backbone parameters, train only the classification head.
- Phase 2: unfreeze only the last `N_UNFROZEN_BLOCKS` transformer blocks + head.

## Training Architecture

`train_dino_model.py` pipeline:
1. Load ImageFolder dataset from `../noiseless_datasets`.
2. Per-class split into train/val/test with approximate 80/10/10 split.
3. Wrap raw subsets with transform-specific dataset wrapper (`TransformSubset`).
4. Use `BalancedPerClassBatchSampler` to enforce fixed images/class in each train batch.
5. Train with `AdamW + CosineAnnealingLR` in two phases.
6. Track best validation checkpoint and save:
   - `best_dino_model.pth` (state dict)
   - `herbarium_dinov2_final.pth` (payload: model state + metadata)
7. Evaluate on test split and export confusion matrix image.

## Inference Architecture

`predict_dino.py`:
- Loads checkpoint payload (`herbarium_dinov2_final.pth`).
- Rebuilds identical model architecture.
- Applies validation transform (`Resize(224) + Normalize`).
- Runs softmax and returns top-k class probabilities.

Inference entry modes:
- `predict_single_image(...)`
- `predict_images(single_image=..., images_dir=...)`

## Retrieval Architecture (FAISS)

`build_dinov2_faiss_index.py`:
- Reuses model loading from `predict_dino.py`.
- Encodes all dataset images using `model.backbone(...)`.
- L2-normalizes embedding vectors.
- Writes FAISS `IndexFlatIP` index (`vectors.faiss`) for cosine similarity via inner product.
- Writes aligned metadata (`metadata.json`) and run config manifest (`manifest.json`).

`dinov2_faiss_server.py`:
- Loads index + metadata + model once at startup.
- Endpoint `POST /query`:
  1. Decode uploaded image.
  2. Transform and encode with DINOv2 backbone.
  3. L2-normalize query embedding.
  4. Search top-k nearest vectors in FAISS.
  5. Return ranked JSON results with species/path metadata.
- Endpoint `GET /ui` serves browser search UI.

## Runtime and Device Design

Device selection order:
- CUDA -> MPS -> CPU (training uses CUDA-first; prediction scripts include MPS-first helper).

Compatibility behavior:
- `PYTORCH_ENABLE_MPS_FALLBACK=1` is enabled to reduce MPS operator compatibility failures.
- `num_workers` defaults to `0` on Darwin for safer multiprocessing behavior.
