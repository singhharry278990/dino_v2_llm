# -----------------------------------------------------------------------------
# Herbarium classification: DINOv2 ViT-S/14 — training approach aligned with
# train_efficientnetv2.py (two-phase fine-tuning, stratified splits, augmentations).
# Model: dinov2_vits14 (embed_dim=384)
# -----------------------------------------------------------------------------

import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

# ─────────────────────────────────────────
# CONFIG (mirror train_efficientnetv2.py where sensible)
# ─────────────────────────────────────────
DATA_DIR = "./noiseless_datasets"

DINO_BACKBONE = "dinov2_vits14"
N_UNFROZEN_BLOCKS = 4  # phase 2: last K transformer blocks unfrozen

IMG_SIZE = 224  # ViT-S/14 standard finetuning size (EN uses 384)
BATCH_SIZE = 16
PHASE1_EPOCHS = 10
PHASE2_EPOCHS = 30
LR_PHASE1 = 1e-3
LR_PHASE2 = 1e-4
WEIGHT_DECAY = 1e-4
SEED = 42

# CE with label smoothing (same as EfficientNet script)
LABEL_SMOOTHING = 0.1

OUT_CKPT_FINAL = "herbarium_dinov2_final.pth"
OUT_CKPT_BEST = "best_dino_model.pth"
OUT_CONFUSION = "confusion_matrix_dinov2.png"


class TransformSubset(torch.utils.data.Dataset):
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        img, lbl = self.subset[idx]
        return self.transform(img), lbl


def freeze_backbone(model: nn.Module) -> None:
    for p in model.backbone.parameters():
        p.requires_grad = False


def unfreeze_backbone_tail(model: nn.Module, n_blocks: int) -> None:
    freeze_backbone(model)
    n = min(n_blocks, len(model.backbone.blocks))
    for block in model.backbone.blocks[-n:]:
        for p in block.parameters():
            p.requires_grad = True


class Model(nn.Module):
    """DINOv2 ViT-S/14 + Linear head (same role as EfficientNet classifier)."""

    def __init__(self, num_classes: int):
        super().__init__()
        self.backbone = torch.hub.load("facebookresearch/dinov2", DINO_BACKBONE)

        for p in self.backbone.parameters():
            p.requires_grad = False

        dim = self.backbone.embed_dim
        self.head = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(dim, num_classes),
        )

    def forward(self, x):
        feat = self.backbone(x)
        logits = self.head(feat)
        return feat, logits


def run_epoch(model, loader, criterion, device, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.set_grad_enabled(is_train):
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            _, logits = model(imgs)
            loss = criterion(logits, labels)
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * imgs.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            total += imgs.size(0)
    return total_loss / max(total, 1), correct / max(total, 1)


def main():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    torch.manual_seed(SEED)
    print(f"Using device: {device}")
    pin_memory = device.type == "cuda"
    num_workers = int(os.environ.get("NUM_WORKERS", "0" if sys.platform == "darwin" else "2"))

    train_tf = transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(30),
            transforms.ColorJitter(
                brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05
            ),
            transforms.RandomGrayscale(p=0.05),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.15)),
        ]
    )

    val_tf = transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    full_dataset = datasets.ImageFolder(DATA_DIR)
    num_classes = len(full_dataset.classes)
    if num_classes == 0:
        raise RuntimeError(f"No classes under {DATA_DIR}")

    train_idx, val_idx, test_idx = [], [], []
    for cls_idx in range(num_classes):
        idxs = [i for i, (_, lbl) in enumerate(full_dataset.samples) if lbl == cls_idx]
        rng = np.random.RandomState(SEED)
        rng.shuffle(idxs)
        n = len(idxs)
        n_val = max(1, int(n * 0.10))
        n_test = max(1, int(n * 0.10))
        test_idx += idxs[:n_test]
        val_idx += idxs[n_test : n_test + n_val]
        train_idx += idxs[n_test + n_val :]

    full_raw = datasets.ImageFolder(DATA_DIR, transform=None)
    train_set = TransformSubset(Subset(full_raw, train_idx), train_tf)
    val_set = TransformSubset(Subset(full_raw, val_idx), val_tf)
    test_set = TransformSubset(Subset(full_raw, test_idx), val_tf)

    train_loader = DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    class_names = full_dataset.classes
    print(f"Backbone: {DINO_BACKBONE}  |  Classes: {class_names}")
    print(f"Train: {len(train_set)} | Val: {len(val_set)} | Test: {len(test_set)}")

    model = Model(num_classes).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

    print("\n=== Phase 1: Head only (frozen DINO backbone) ===")
    freeze_backbone(model)
    for p in model.head.parameters():
        p.requires_grad = True

    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR_PHASE1,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=PHASE1_EPOCHS)

    best_val_acc, best_state = 0.0, None
    for epoch in range(1, PHASE1_EPOCHS + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, device, optimizer)
        vl_loss, vl_acc = run_epoch(model, val_loader, criterion, device)
        scheduler.step()
        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        print(
            f"Epoch {epoch:2d}/{PHASE1_EPOCHS} | "
            f"train loss {tr_loss:.4f} acc {tr_acc:.3f} | "
            f"val loss {vl_loss:.4f} acc {vl_acc:.3f}"
        )

    model.load_state_dict(best_state)

    print("\n=== Phase 2: Fine-tune (last backbone blocks + head) ===")
    unfreeze_backbone_tail(model, N_UNFROZEN_BLOCKS)
    for p in model.head.parameters():
        p.requires_grad = True

    optimizer = AdamW(model.parameters(), lr=LR_PHASE2, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=PHASE2_EPOCHS)

    best_val_acc, best_state = 0.0, None
    for epoch in range(1, PHASE2_EPOCHS + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, device, optimizer)
        vl_loss, vl_acc = run_epoch(model, val_loader, criterion, device)
        scheduler.step()
        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        print(
            f"Epoch {epoch:2d}/{PHASE2_EPOCHS} | "
            f"train loss {tr_loss:.4f} acc {tr_acc:.3f} | "
            f"val loss {vl_loss:.4f} acc {vl_acc:.3f}"
        )

    model.load_state_dict(best_state)
    torch.save(best_state, OUT_CKPT_BEST)
    print(f"\nBest val accuracy (phase 2): {best_val_acc:.3f} — saved {OUT_CKPT_BEST}")

    payload = {
        "model_state": best_state,
        "species_classes": class_names,
        "num_species": num_classes,
        "backbone": DINO_BACKBONE,
        "img_size": IMG_SIZE,
        "best_val_acc": best_val_acc,
    }
    torch.save(payload, OUT_CKPT_FINAL)

    from sklearn.preprocessing import LabelEncoder

    le = LabelEncoder()
    le.fit(class_names)
    with open("encoders.pkl", "wb") as f:
        pickle.dump({"species": le}, f)

    print("\n=== Test set evaluation ===")
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device)
            _, logits = model(imgs)
            all_preds.extend(logits.argmax(1).cpu().numpy())
            all_labels.extend(labels.numpy())

    print(classification_report(all_labels, all_preds, target_names=class_names))

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title("Confusion matrix — DINOv2 ViT-S/14 test set")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(OUT_CONFUSION, dpi=150)
    print(f"Confusion matrix saved to {OUT_CONFUSION}")
    print(f"Final checkpoint: {OUT_CKPT_FINAL}")


if __name__ == "__main__":
    main()
