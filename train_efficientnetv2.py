"""
EfficientNetV2-S training script for herbarium species classification
Dataset: 5 species × 125 images = 625 total
"""

import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
from torchvision.models import EfficientNet_V2_S_Weights
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
DATA_DIR       = "./noiseless_datasets"   # folder structure: DATA_DIR/<species_name>/<image>.jpg
IMG_SIZE       = 384                     # EfficientNetV2-S native resolution
BATCH_SIZE     = 16                      # small dataset → small batch
PHASE1_EPOCHS  = 10                      # head-only warm-up
PHASE2_EPOCHS  = 30                      # full fine-tune
LR_PHASE1      = 1e-3
LR_PHASE2      = 1e-4
WEIGHT_DECAY   = 1e-4
SEED           = 42


class TransformSubset(torch.utils.data.Dataset):
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        img, lbl = self.subset[idx]
        return self.transform(img), lbl


def freeze_backbone(model, freeze=True):
    for name, param in model.named_parameters():
        if "classifier" not in name:
            param.requires_grad = not freeze


def run_epoch(model, loader, criterion, device, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.set_grad_enabled(is_train):
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * imgs.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += imgs.size(0)
    return total_loss / total, correct / total


def predict(image_path, model, class_names, device, transform):
    """Predict species for a single herbarium image."""
    from PIL import Image

    img = Image.open(image_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
    top_idx = probs.argmax()
    return class_names[top_idx], float(probs[top_idx]), dict(zip(class_names, probs.tolist()))


def main():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    torch.manual_seed(SEED)
    print(f"Using device: {device}")

    # macOS default spawn: workers re-import this file — only `main()` must run training.
    # MPS + workers is fragile; keep 0 on Darwin unless overridden.
    num_workers = int(os.environ.get("NUM_WORKERS", "0" if sys.platform == "darwin" else "2"))
    pin_memory = device.type == "cuda"

    train_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
        transforms.RandomGrayscale(p=0.05),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.15)),
    ])

    val_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    full_dataset = datasets.ImageFolder(DATA_DIR)
    class_names = full_dataset.classes
    num_classes = len(class_names)

    train_idx, val_idx, test_idx = [], [], []
    for cls_idx in range(num_classes):
        idxs = [i for i, (_, lbl) in enumerate(full_dataset.samples) if lbl == cls_idx]
        np.random.shuffle(idxs)
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

    print(f"Classes: {class_names}")
    print(f"Train: {len(train_set)} | Val: {len(val_set)} | Test: {len(test_set)}")

    weights = EfficientNet_V2_S_Weights.IMAGENET1K_V1
    model = models.efficientnet_v2_s(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features, num_classes),
    )
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    print("\n=== Phase 1: Head only ===")
    freeze_backbone(model, freeze=True)
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

    print("\n=== Phase 2: Fine-tune ===")
    freeze_backbone(model, freeze=False)
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
    torch.save(best_state, "efficientnetv2s_herbarium_best.pth")
    print(f"\nBest val accuracy: {best_val_acc:.3f} — model saved.")

    print("\n=== Test set evaluation ===")
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device)
            preds = model(imgs).argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    print(classification_report(all_labels, all_preds, target_names=class_names))

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion matrix — test set")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)
    print("Confusion matrix saved to confusion_matrix.png")


if __name__ == "__main__":
    main()
