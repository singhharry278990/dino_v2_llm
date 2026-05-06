# DINOv2 Data Pipeline

## Dataset Layout

Expected input structure:

`../noiseless_datasets/<species_name>/<image_file>`

The project uses `torchvision.datasets.ImageFolder`, so:
- folder names become class labels,
- class index order is lexicographic by folder name.

## Split Strategy

Implemented in `train_dino_model.py`:
- Iterate class-by-class.
- Shuffle class sample indices (seeded).
- Allocate per class:
  - test: `max(1, 10%)`
  - val: `max(1, 10%)`
  - train: remaining samples

This keeps each split stratified per species.

## Transform Pipeline

### Train Transform
- Resize to `224x224`
- Random horizontal + vertical flip
- Random rotation (`30 deg`)
- Color jitter (brightness/contrast/saturation/hue)
- Random grayscale
- Tensor conversion
- ImageNet normalization
- Random erasing

### Validation/Test Transform
- Resize to `224x224`
- Tensor conversion
- ImageNet normalization

## Batch Construction

`BalancedPerClassBatchSampler` ensures each train batch has:
- fixed images per class (`TRAIN_IMAGES_PER_CLASS=2`),
- all classes represented in each batch,
- shuffled indices per epoch.

Effective train batch size:
- `num_classes * TRAIN_IMAGES_PER_CLASS`

## Training Outputs

Generated artifacts:
- `best_dino_model.pth`
- `herbarium_dinov2_final.pth`
- `encoders.pkl`
- `confusion_matrix_dinov2.png`

## Inference Data Flow

`predict_dino.py`:
1. Load image(s) from path or directory.
2. Apply validation transform.
3. Forward through model.
4. Compute softmax.
5. Return top-k (`species`, `confidence`).

## Retrieval Data Flow

### Index Build (`build_dinov2_faiss_index.py`)
1. Walk all ImageFolder samples with paths.
2. Transform each image.
3. Extract backbone embedding (`dim=384`).
4. L2-normalize vectors.
5. Add vectors to FAISS `IndexFlatIP`.
6. Save:
   - `vectors.faiss`
   - `metadata.json`
   - `manifest.json`

### Query Serving (`dinov2_faiss_server.py`)
1. Receive uploaded query image.
2. Apply same transform.
3. Compute normalized query embedding.
4. Search FAISS top-k.
5. Return ranked matches with species and file path.
