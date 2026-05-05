# -----------------------------------------------------------------------------
# Herbarium classification: DINOv2 vits14 (small) + ArcFace + Triplet
# Optimized for: Apple Silicon Mac (MPS backend)
# Model: dinov2_vits14 (embed_dim=384, lighter / faster than vitb14)
# -----------------------------------------------------------------------------

import os
# DINOv2 interpolates positional encodings with bicubic upsample; MPS lacks this op.
# Enable CPU fallback for those ops only (set before `import torch`).
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import sys
import types

# Pyenv/macOS builds without liblzma omit the _lzma extension; torchvision's
# top-level import loads torchvision.datasets, which imports lzma. Stub the
# datasets package so we only pull in transforms (this script does not use datasets).
if "torchvision.datasets" not in sys.modules:
    _ds = types.ModuleType("torchvision.datasets")
    _ds.__path__ = []
    sys.modules["torchvision.datasets"] = _ds

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
import pickle
import torch.nn.functional as F
import math
from collections import Counter
# from google.colab import drive

# drive.mount('/content/drive')
# # ---------------- CONFIG ----------------
# IMG_DIR    = "/content/drive/MyDrive/noiseless_datasets/"
IMG_DIR        = "noiseless_datasets/"
TRAIN_CSV      = "Training_Data/training-data-train.csv"
TEST_CSV       = "Training_Data/training-data-test.csv"


def canonical_species(s):
    """Unify CSV casing (e.g. Conjugatum vs conjugatum) with on-disk folder names (lowercase)."""
    return str(s).strip().lower()


def resolve_specimen_image(base_dir, species, barcode):
    """Find image on disk: base_dir/<species>/<barcode>.png (your layout), else flat base_dir."""
    base_dir = os.path.expanduser(base_dir)
    sp = canonical_species(species)
    bc = str(barcode).strip()
    exts = (".png", ".PNG", ".jpg", ".jpeg", ".JPEG", ".JPG")
    for root in (os.path.join(base_dir, sp), base_dir):
        for ext in exts:
            p = os.path.join(root, bc + ext)
            if os.path.isfile(p):
                return p
    return None


# ---------------- DEVICE: MPS > CUDA > CPU ----------------
if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

# ---------------- CONFIG (~500–1k images: small data) ----------------
BATCH_SIZE          = 16     # more steps/epoch than 32; stable grads on small N
EPOCHS              = 60
PATIENCE            = 15     # val F1 is noisy on small val splits
MIXUP_ALPHA         = 0.2    # lighter mixup — easier to fit scarce classes
TRIPLET_WEIGHT      = 0.12   # triplets are noisy in small batches
ARCFACE_WEIGHT      = 0.35
HARD_TRIPLET_WEIGHT = 0.12
# Fewer unfrozen ViT blocks → less overfit on tiny finetune sets
N_UNFROZEN_BLOCKS   = 4

HARD_CLASSES = ["acuminata", "paniculata", "nepalensis", "indicum"]

# DINOv2: dinov2_vits14 (small) | dinov2_vitb14 (base) | dinov2_vitl14 (large)
DINO_BACKBONE = "dinov2_vits14"


# ---------------- DATASET ----------------
class CSVImageDataset(Dataset):
    def __init__(self, csv_file, species_encoder, is_train=True):
        self.df = pd.read_csv(csv_file).dropna(subset=["species", "barcode_number"])
        self.df["species"] = self.df["species"].map(canonical_species)
        known_mask   = self.df["species"].astype(str).isin(species_encoder.classes_)
        self.df      = self.df[known_mask].reset_index(drop=True)

        before = len(self.df)
        paths, keep = [], []
        for i in range(before):
            row = self.df.iloc[i]
            p = resolve_specimen_image(IMG_DIR, row["species"], row["barcode_number"])
            if p:
                paths.append(p)
                keep.append(i)
        self.df = self.df.iloc[keep].reset_index(drop=True)
        self.image_paths = paths
        dropped = before - len(self.df)
        if dropped:
            print(
                f"  CSVImageDataset({csv_file}): skipped {dropped} row(s) — "
                f"no file under {IMG_DIR}<species>/ or flat layout"
            )
        if len(self.df) == 0:
            raise FileNotFoundError(
                f"No images found for {csv_file} under {os.path.abspath(IMG_DIR)}. "
                "Expected paths like noiseless_datasets/<species>/<barcode>.png"
            )

        self.species = species_encoder.transform(self.df["species"].astype(str))
        self.is_train = is_train

        self.base_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(0.3, 0.3, 0.3, 0.1),
            transforms.RandomGrayscale(p=0.05),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        self.hard_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(35),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.15),
            transforms.RandomGrayscale(p=0.08),
            transforms.RandomAffine(degrees=0, translate=(0.08, 0.08)),
            transforms.RandomPerspective(distortion_scale=0.3, p=0.4),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        self.val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(self.image_paths[idx]).convert("RGB")

        if not self.is_train:
            return self.val_transform(img), self.species[idx]
        if row["species"] in HARD_CLASSES:
            return self.hard_transform(img), self.species[idx]
        return self.base_transform(img), self.species[idx]


# ---------------- ARCFACE LOSS ----------------
class ArcFaceLoss(nn.Module):
    def __init__(self, in_features, num_classes, s=32.0, m=0.6):
        # m=0.6 (was 0.5) — tighter boundary for confused species
        super().__init__()
        self.s      = s
        self.m      = m
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th    = math.cos(math.pi - m)
        self.mm    = math.sin(math.pi - m) * m

    def forward(self, embeddings, labels):
        cosine  = F.linear(embeddings, F.normalize(self.weight))
        sine    = torch.sqrt(1.0 - torch.clamp(cosine ** 2, 0, 1))
        phi     = cosine * self.cos_m - sine * self.sin_m
        phi     = torch.where(cosine > self.th, phi, cosine - self.mm)
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)
        output  = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        return F.cross_entropy(output * self.s, labels)


# ---------------- MODEL ----------------
class Model(nn.Module):
    def __init__(self, num_species):
        super().__init__()

        # vits14: embed_dim=384, 12 blocks — smallest DINOv2 /14 variant
        self.backbone = torch.hub.load("facebookresearch/dinov2", DINO_BACKBONE)

        for p in self.backbone.parameters():
            p.requires_grad = False

        n_unfreeze = min(N_UNFROZEN_BLOCKS, len(self.backbone.blocks))
        for block in self.backbone.blocks[-n_unfreeze:]:
            for p in block.parameters():
                p.requires_grad = True

        dim = self.backbone.embed_dim  # 384 for vits14

        self.embedding = nn.Sequential(
            nn.Linear(dim, 512),       # backbone dim → 512
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.45),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.35),
            nn.Linear(256, 256)
        )

        self.classifier = nn.Linear(256, num_species)

    def forward(self, x):
        feat   = self.backbone(x)
        emb    = self.embedding(feat)
        emb    = F.normalize(emb, dim=1)
        logits = self.classifier(emb)
        return emb, logits


# ---------------- MIXUP ----------------
def mixup_data(imgs, labels, alpha=0.4):
    if alpha <= 0:
        return imgs, labels, labels, 1.0
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(imgs.size(0), device=imgs.device)
    return lam * imgs + (1 - lam) * imgs[idx], labels, labels[idx], lam

def mixup_criterion(criterion, logits, y_a, y_b, lam):
    return lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)


# ---------------- TRIPLET LOSS ----------------
def compute_triplet_loss(emb, labels, margin=0.3):
    loss_fn = nn.TripletMarginLoss(margin=margin)
    anchors, positives, negatives = [], [], []

    for i in range(len(labels)):
        same = (labels == labels[i]).nonzero(as_tuple=True)[0]
        diff = (labels != labels[i]).nonzero(as_tuple=True)[0]
        if len(same) < 2 or len(diff) == 0:
            continue
        same_no_i = same[same != i]
        pos_sims  = F.cosine_similarity(emb[i].unsqueeze(0), emb[same_no_i])
        pos_idx   = same_no_i[torch.argmin(pos_sims)]
        neg_sims  = F.cosine_similarity(emb[i].unsqueeze(0), emb[diff])
        neg_idx   = diff[torch.argmax(neg_sims)]
        anchors.append(emb[i])
        positives.append(emb[pos_idx])
        negatives.append(emb[neg_idx])

    if len(anchors) == 0:
        return torch.tensor(0.0, requires_grad=True).to(emb.device)
    return loss_fn(torch.stack(anchors), torch.stack(positives), torch.stack(negatives))


# ---------------- HARD CLASS TRIPLET ----------------
def compute_hard_class_triplet(emb, labels, hard_class_ids, margin=0.6):
    loss_fn  = nn.TripletMarginLoss(margin=margin)
    anchors, positives, negatives = [], [], []

    hard_mask = torch.zeros(len(labels), dtype=torch.bool, device=labels.device)
    for hid in hard_class_ids:
        hard_mask |= (labels == hid)

    hard_indices = hard_mask.nonzero(as_tuple=True)[0]
    for i in hard_indices.tolist():
        same = (labels == labels[i]).nonzero(as_tuple=True)[0]
        diff = hard_indices[labels[hard_indices] != labels[i]]
        if len(same) < 2 or len(diff) == 0:
            continue
        same_no_i = same[same != i]
        pos_sims  = F.cosine_similarity(emb[i].unsqueeze(0), emb[same_no_i])
        pos_idx   = same_no_i[torch.argmin(pos_sims)]
        neg_sims  = F.cosine_similarity(emb[i].unsqueeze(0), emb[diff])
        neg_idx   = diff[torch.argmax(neg_sims)]
        anchors.append(emb[i])
        positives.append(emb[pos_idx])
        negatives.append(emb[neg_idx])

    if len(anchors) == 0:
        return torch.tensor(0.0, requires_grad=True).to(emb.device)
    return loss_fn(torch.stack(anchors), torch.stack(positives), torch.stack(negatives))


# ---------------- TTA VALIDATION ----------------
def validate_with_tta(model, loader, device):
    model.eval()
    val_preds, val_true = [], []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            _, logits_orig    = model(imgs)
            _, logits_flipped = model(torch.flip(imgs, dims=[3]))
            logits = (logits_orig + logits_flipped) / 2
            preds  = torch.argmax(logits, dim=1)
            val_preds.append(preds.cpu())
            val_true.append(labels)

    return torch.cat(val_true), torch.cat(val_preds)


# ---------------- MAIN ----------------
if __name__ == '__main__':

    print(f"Using device: {DEVICE}")
    if DEVICE == "mps":
        print("Apple Silicon MPS backend active — ~3-4x faster than CPU")
        if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1":
            print("  PYTORCH_ENABLE_MPS_FALLBACK=1 — rare ops (e.g. DINOv2 pos-embed bicubic) run on CPU")

    train_df_raw = pd.read_csv(TRAIN_CSV).dropna(
        subset=["species", "barcode_number"]
    )
    train_df_raw["species"] = train_df_raw["species"].map(canonical_species)

    species_encoder = LabelEncoder()
    species_encoder.fit(train_df_raw["species"].astype(str))
    num_species = len(species_encoder.classes_)
    print(f"Total species (canonical): {num_species}  |  classes: {list(species_encoder.classes_)}")

    # Hard class sample counts
    print("\nHard class sample counts (train):")
    counts = Counter(train_df_raw["species"])
    for cls in HARD_CLASSES:
        n    = counts.get(cls, 0)
        flag = "  ⚠ LOW" if n < 30 else ""
        print(f"  {cls:<15}: {n} samples{flag}")
    print()

    hard_class_ids = [
        int(species_encoder.transform([c])[0])
        for c in HARD_CLASSES if c in species_encoder.classes_
    ]
    print(f"Hard classes → IDs: {hard_class_ids}")

    # Class weights — no label_smoothing (conflicts with ArcFace)
    train_species  = species_encoder.transform(train_df_raw["species"].astype(str))
    weights        = compute_class_weight(
        "balanced", classes=np.unique(train_species), y=train_species
    )
    weights_tensor = torch.tensor(weights, dtype=torch.float).to(DEVICE)
    ce_loss_fn     = nn.CrossEntropyLoss(weight=weights_tensor)

    # Datasets
    train_dataset = CSVImageDataset(TRAIN_CSV, species_encoder, True)
    test_dataset  = CSVImageDataset(TEST_CSV,  species_encoder, False)

    n_train, n_val = len(train_dataset), len(test_dataset)
    print(f"Samples with images — train: {n_train}  |  val: {n_val}\n")

    class_counts = np.bincount(train_dataset.species, minlength=num_species)
    sample_weights = torch.tensor(
        1.0 / np.maximum(class_counts[train_dataset.species], 1).astype(np.float64),
        dtype=torch.float,
    )
    sampler = WeightedRandomSampler(sample_weights, len(train_dataset), replacement=True)

    # MPS pe num_workers=0 zaroori hai — multiprocessing MPS ke saath crash karta hai
    nw = 0 if DEVICE == "mps" else 4
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, sampler=sampler,
        num_workers=nw, pin_memory=False   # pin_memory=False for MPS
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE,
        num_workers=nw, pin_memory=False
    )

    # Model
    model   = Model(num_species).to(DEVICE)
    arcface = ArcFaceLoss(256, num_species, s=32.0, m=0.6).to(DEVICE)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)\n")

    optimizer = optim.AdamW([
        {"params": model.backbone.parameters(),   "lr": 8e-6,  "weight_decay": 1e-5},
        {"params": model.embedding.parameters(),  "lr": 2e-4,  "weight_decay": 4e-3},
        {"params": model.classifier.parameters(), "lr": 3e-4,  "weight_decay": 4e-3},
        {"params": arcface.parameters(),          "lr": 2e-4,  "weight_decay": 1e-3},
    ])

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    # MPS pe GradScaler nahi chalta — sirf CUDA ke liye
    use_amp = (DEVICE == "cuda")
    scaler  = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_val_f1       = 0
    epochs_no_improve = 0

    # ---------------- TRAIN LOOP ----------------
    for epoch in range(EPOCHS):

        model.train()
        arcface.train()
        all_preds, all_true = [], []
        total_loss = 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

            mixed_imgs, labels_a, labels_b, lam = mixup_data(imgs, labels, MIXUP_ALPHA)

            optimizer.zero_grad()

            # MPS pe autocast supported nahi — sirf CUDA ke liye
            with torch.cuda.amp.autocast(enabled=use_amp):
                _, logits_mixed       = model(mixed_imgs)
                ce_loss               = mixup_criterion(
                    ce_loss_fn, logits_mixed, labels_a, labels_b, lam
                )
                emb_orig, logits_orig = model(imgs)
                arc_loss              = arcface(emb_orig, labels)
                triplet_loss          = compute_triplet_loss(emb_orig, labels, margin=0.3)
                hard_triplet_loss     = compute_hard_class_triplet(
                    emb_orig, labels, hard_class_ids, margin=0.6
                )
                loss = (ce_loss
                        + ARCFACE_WEIGHT * arc_loss
                        + TRIPLET_WEIGHT * triplet_loss
                        + HARD_TRIPLET_WEIGHT * hard_triplet_loss)

            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    list(model.parameters()) + list(arcface.parameters()), max_norm=1.0
                )
                scaler.step(optimizer)
                scaler.update()
            else:
                # MPS / CPU path
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(model.parameters()) + list(arcface.parameters()), max_norm=1.0
                )
                optimizer.step()

            total_loss += loss.item() * imgs.size(0)
            preds = torch.argmax(logits_orig, dim=1)
            all_preds.append(preds.cpu())
            all_true.append(labels.cpu())

        scheduler.step()
        label_idx = np.arange(num_species)
        train_f1 = f1_score(
            torch.cat(all_true), torch.cat(all_preds),
            labels=label_idx, average="macro", zero_division=0,
        )
        avg_loss = total_loss / len(train_dataset)

        # Validation with TTA
        val_true_cat, val_preds_cat = validate_with_tta(model, test_loader, DEVICE)
        val_f1 = f1_score(
            val_true_cat, val_preds_cat,
            labels=label_idx, average="macro", zero_division=0,
        )

        lrs = scheduler.get_last_lr()
        print(f"\nEpoch {epoch+1}/{EPOCHS}  |  backbone LR: {lrs[0]:.2e}  |  head LR: {lrs[1]:.2e}  |  Loss: {avg_loss:.4f}")
        print(f"Train F1: {train_f1:.4f}  |  Val F1 (TTA): {val_f1:.4f}")

        if (epoch + 1) % 5 == 0:
            report = classification_report(
                val_true_cat.numpy(), val_preds_cat.numpy(),
                labels=label_idx,
                target_names=species_encoder.classes_,
                output_dict=True, zero_division=0,
            )
            print("  Per-class F1 (val):")
            for cls in species_encoder.classes_:
                f1  = report[cls]["f1-score"]
                tag = " <-- improve karo" if f1 < 0.7 else ""
                print(f"    {cls:<15} F1: {f1:.3f}{tag}")

        if val_f1 > best_val_f1:
            best_val_f1       = val_f1
            epochs_no_improve = 0
            torch.save({
                "model_state":   model.state_dict(),
                "arcface_state": arcface.state_dict(),
            }, "best_model.pth")
            print("  Saved best model")
        else:
            epochs_no_improve += 1
            print(f"  No improvement ({epochs_no_improve}/{PATIENCE})")
            if epochs_no_improve >= PATIENCE:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break

    print(f"\nBest Val F1: {best_val_f1:.4f}")

    # ---------------- SAVE ----------------
    best_ckpt = torch.load("best_model.pth", map_location=DEVICE)
    model.load_state_dict(best_ckpt["model_state"])
    arcface.load_state_dict(best_ckpt["arcface_state"])

    torch.save({
        "model_state":     model.state_dict(),
        "arcface_state":   arcface.state_dict(),
        "species_classes": species_encoder.classes_,
        "num_species":     num_species,
        "best_val_f1":     best_val_f1,
    }, "herbarium_dinov2_final.pth")

    with open("encoders.pkl", "wb") as f:
        pickle.dump({"species": species_encoder}, f)

    print("Final model saved: herbarium_dinov2_final.pth")