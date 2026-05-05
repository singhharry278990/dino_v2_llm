"""
Load efficientnetv2s_herbarium_best.pth and predict species for image paths.

Class order matches torchvision ImageFolder (sorted subfolder names under
noiseless_datasets), same as train_efficientnetv2.py.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

IMG_SIZE = 384
DEFAULT_CKPT = "efficientnetv2s_herbarium_best.pth"
DEFAULT_DATA_DIR = "./noiseless_datasets"


def imagefolder_classes(root: str) -> list[str]:
    """Same class list as torchvision.datasets.ImageFolder(root).classes."""
    root = os.path.expanduser(root)
    names = [d.name for d in os.scandir(root) if d.is_dir() and not d.name.startswith(".")]
    names.sort()
    return names


def build_model(num_classes: int) -> nn.Module:
    m = models.efficientnet_v2_s(weights=None)
    in_features = m.classifier[1].in_features
    m.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features, num_classes),
    )
    return m


def val_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(ckpt_path: str, data_dir: str, device: torch.device) -> tuple[nn.Module, list[str], transforms.Compose]:
    ckpt_path = os.path.expanduser(ckpt_path)
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    try:
        state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(ckpt_path, map_location="cpu")
    if isinstance(state, dict) and any(k.startswith("classifier.") for k in state):
        state_dict = state
    elif isinstance(state, dict) and "model_state" in state:
        state_dict = state["model_state"]
    else:
        state_dict = state

    n_cls = int(state_dict["classifier.1.weight"].shape[0])
    class_names = imagefolder_classes(data_dir)
    if len(class_names) != n_cls:
        raise ValueError(
            f"Checkpoint has {n_cls} classes but {data_dir!r} has {len(class_names)} "
            f"ImageFolder-style folders: {class_names}. Use the same dataset root as training."
        )

    model = build_model(n_cls)
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model, class_names, val_transform()


@torch.inference_mode()
def forward_probs(
    model: nn.Module, path: str, tfm: transforms.Compose, device: torch.device
) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    x = tfm(img).unsqueeze(0).to(device)
    return torch.softmax(model(x), dim=1)[0]


def ranked_from_probs(
    probs: torch.Tensor, class_names: list[str], topk: int
) -> tuple[str, float, list[tuple[str, float]]]:
    k = min(topk, len(class_names))
    top_p, top_i = probs.topk(k)
    ranked = [(class_names[i], float(p)) for p, i in zip(top_p.tolist(), top_i.tolist())]
    return ranked[0][0], ranked[0][1], ranked


def collect_image_paths(paths: list[str]) -> list[str]:
    exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
    out: list[str] = []
    for p in paths:
        p = os.path.expanduser(p)
        if os.path.isdir(p):
            for root, _, files in os.walk(p):
                for f in sorted(files):
                    if Path(f).suffix.lower() in exts:
                        out.append(os.path.join(root, f))
        elif os.path.isfile(p):
            if Path(p).suffix.lower() in exts or p.lower().endswith((".png", ".jpg", ".jpeg")):
                out.append(p)
            else:
                print(f"Skip (unsupported extension): {p}", file=sys.stderr)
        else:
            print(f"Skip (not found): {p}", file=sys.stderr)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Predict species with EfficientNetV2-S checkpoint.")
    ap.add_argument(
        "inputs",
        nargs="*",
        help="Image files and/or directories to predict (recursive for dirs).",
    )
    ap.add_argument(
        "--checkpoint",
        "-c",
        default=DEFAULT_CKPT,
        help=f"Path to .pth state dict (default: {DEFAULT_CKPT})",
    )
    ap.add_argument(
        "--data-dir",
        "-d",
        default=DEFAULT_DATA_DIR,
        help=f"Dataset root used in training, for class order (default: {DEFAULT_DATA_DIR})",
    )
    ap.add_argument("--topk", type=int, default=5, help="Show top-k probabilities.")
    ap.add_argument(
        "--csv",
        metavar="OUT.csv",
        help="If set, write path,pred,confidence,prob_class0,... columns here.",
    )
    args = ap.parse_args()

    if not args.inputs:
        ap.print_help()
        print("\nExample: python predict_efficientnetv2.py sheet.png noiseless_datasets/ellipticus/", file=sys.stderr)
        sys.exit(1)

    device = pick_device()
    model, class_names, tfm = load_model(args.checkpoint, args.data_dir, device)
    print(f"Device: {device} | Classes ({len(class_names)}): {class_names}\n")

    paths = collect_image_paths(args.inputs)
    if not paths:
        print("No images found.", file=sys.stderr)
        sys.exit(2)

    csv_lines: list[list[str]] = []
    for path in paths:
        try:
            probs = forward_probs(model, path, tfm, device)
        except Exception as e:
            print(f"{path}\tERROR\t{e}", file=sys.stderr)
            continue
        pred, conf, ranked = ranked_from_probs(probs, class_names, args.topk)
        rest = " | ".join(f"{n}:{p:.4f}" for n, p in ranked[1:])
        print(f"{path}\n  → {pred} ({conf:.2%})" + (f"\n     " + rest if rest else ""))
        if args.csv:
            pv = probs.detach().cpu().tolist()
            csv_lines.append(
                [path.replace(",", ";"), pred, f"{conf:.6f}"] + [f"{p:.6f}" for p in pv]
            )

    if args.csv and csv_lines:
        header = ["path", "pred", "confidence"] + [f"prob_{c}" for c in class_names]
        with open(args.csv, "w", encoding="utf-8") as f:
            f.write(",".join(header) + "\n")
            for row in csv_lines:
                f.write(",".join(row) + "\n")
        print(f"\nWrote {args.csv}")


if __name__ == "__main__":
    main()
