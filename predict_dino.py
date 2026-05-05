import os

# Match train_dino_model.py — DINOv2 pos-embed bicubic on MPS uses CPU fallback.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

# Must match checkpoint (train_dino_model.py)
DINO_BACKBONE = "dinov2_vits14"
N_UNFROZEN_BLOCKS = 4


def pick_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class Model(nn.Module):
    """Same architecture as train_dino_model.Model (ViT-S/14 + dropout + linear head)."""

    def __init__(self, num_species):
        super().__init__()

        self.backbone = torch.hub.load("facebookresearch/dinov2", DINO_BACKBONE)

        for p in self.backbone.parameters():
            p.requires_grad = False

        n_unfreeze = min(N_UNFROZEN_BLOCKS, len(self.backbone.blocks))
        for block in self.backbone.blocks[-n_unfreeze:]:
            for p in block.parameters():
                p.requires_grad = True

        dim = self.backbone.embed_dim  # 384 for vits14
        self.head = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(dim, num_species),
        )

    def forward(self, x):
        feat = self.backbone(x)
        logits = self.head(feat)
        return feat, logits


def build_transform():
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


def list_images(single_image: str | None, images_dir: str | None):
    paths = []
    if single_image:
        paths.append(Path(single_image))
    if images_dir:
        folder = Path(images_dir)
        exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
        paths.extend(sorted(p for p in folder.iterdir() if p.suffix.lower() in exts))
    return paths


def load_model(model_path: str = "herbarium_dinov2_final.pth"):
    device = pick_device()
    checkpoint = torch.load(model_path, map_location=device)

    num_species = int(checkpoint["num_species"])
    species_classes = checkpoint["species_classes"]

    model = Model(num_species=num_species).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    transform = build_transform()
    return model, transform, species_classes, device


def predict_single_image(
    image_path: str,
    model_path: str = "herbarium_dinov2_final.pth",
    topk: int = 3,
):
    model, transform, species_classes, device = load_model(model_path=model_path)
    img = Image.open(image_path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        _, logits = model(tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0)
        k = min(topk, probs.numel())
        top_probs, top_idxs = torch.topk(probs, k=k)

    results = []
    for idx, prob in zip(top_idxs.tolist(), top_probs.tolist()):
        results.append({"species": str(species_classes[idx]), "confidence": float(prob)})
    return results


def predict_images(
    single_image: str | None = None,
    images_dir: str | None = None,
    model_path: str = "herbarium_dinov2_final.pth",
    topk: int = 3,
):
    if not single_image and not images_dir:
        raise ValueError("Provide at least one of single_image or images_dir.")

    image_paths = list_images(single_image, images_dir)
    if not image_paths:
        raise ValueError("No valid image files found.")

    model, transform, species_classes, device = load_model(model_path=model_path)
    outputs = {}
    with torch.no_grad():
        for path in image_paths:
            img = Image.open(path).convert("RGB")
            tensor = transform(img).unsqueeze(0).to(device)

            _, logits = model(tensor)
            probs = torch.softmax(logits, dim=1).squeeze(0)

            k = min(topk, probs.numel())
            top_probs, top_idxs = torch.topk(probs, k=k)
            outputs[str(path)] = [
                {"species": str(species_classes[idx]), "confidence": float(prob)}
                for idx, prob in zip(top_idxs.tolist(), top_probs.tolist())
            ]
    return outputs


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(
            "Usage: python predict_dino.py <image_path> [herbarium_dinov2_final.pth]\n"
            "Loads dinov2_vits14 (small) head matching train_dino_model.py."
        )
        sys.exit(1)
    img_arg = sys.argv[1]
    ckpt = sys.argv[2] if len(sys.argv) > 2 else "herbarium_dinov2_final.pth"
    result = predict_single_image(image_path=img_arg, model_path=ckpt, topk=10)
    print(json.dumps(result, indent=4))
