#!/usr/bin/env python3
"""
Encode all images under an ImageFolder tree with the fine-tuned DINOv2 backbone
from herbarium_dinov2_final.pth, L2-normalize vectors, and build a FAISS inner-product
index (cosine similarity).

Outputs (default --out_dir dinov2_faiss_index):
  vectors.faiss   — FAISS IndexFlatIP, float32, dim=384 for dinov2_vits14
  metadata.json   — one entry per vector row (path, species, class_idx, stem)
  manifest.json   — run config (checkpoint, data dir, counts)

Usage:
  python build_dinov2_faiss_index.py \\
    --checkpoint dinov2/herbarium_dinov2_final.pth \\
    --data_dir ./noiseless_datasets \\
    --out_dir dinov2_faiss_index
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import faiss
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from predict_dino import load_model


class ImageFolderWithPath(datasets.ImageFolder):
    def __getitem__(self, index: int):
        path, target = self.samples[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target, path


def collate_paths(batch):
    imgs = torch.stack([b[0] for b in batch])
    targets = torch.tensor([b[1] for b in batch], dtype=torch.long)
    paths = [b[2] for b in batch]
    return imgs, targets, paths


def resolve_checkpoint(path: str | None) -> Path:
    candidates = []
    if path:
        candidates.append(Path(path))
    candidates.extend(
        [
            Path("dinov2/herbarium_dinov2_final.pth"),
            Path("herbarium_dinov2_final.pth"),
        ]
    )
    for c in candidates:
        if c.is_file():
            return c.resolve()
    tried = ", ".join(str(c) for c in candidates if c)
    raise FileNotFoundError(f"No checkpoint found. Tried: {tried}")


def main():
    ap = argparse.ArgumentParser(description="Build FAISS index from DINOv2 backbone embeddings.")
    ap.add_argument(
        "--checkpoint",
        default=None,
        help="Path to herbarium_dinov2_final.pth (default: dinov2/ or repo root)",
    )
    ap.add_argument("--data_dir", default="./noiseless_datasets", help="ImageFolder root")
    ap.add_argument("--out_dir", default="dinov2_faiss_index", help="Output directory")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="DataLoader workers (default: 0 on darwin, else 2)",
    )
    args = ap.parse_args()

    ckpt_path = resolve_checkpoint(args.checkpoint)
    data_dir = Path(args.data_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not data_dir.is_dir():
        raise FileNotFoundError(f"data_dir not found: {data_dir}")

    num_workers = args.num_workers
    if num_workers is None:
        num_workers = 0 if sys.platform == "darwin" else 2

    model, transform, species_classes, device = load_model(model_path=str(ckpt_path))
    model.eval()
    dev = torch.device(device) if isinstance(device, str) else device

    img_size = 224
    # If checkpoint stores img_size, match it (train_dino_model uses 224).
    try:
        try:
            blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        except TypeError:
            blob = torch.load(ckpt_path, map_location="cpu")
        img_size = int(blob.get("img_size", img_size))
    except Exception:
        pass

    if img_size != 224:
        transform = transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    ds = ImageFolderWithPath(str(data_dir), transform=transform)
    if len(ds) == 0:
        raise RuntimeError(f"No images under {data_dir}")

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=dev.type == "cuda",
        collate_fn=collate_paths,
    )

    dim = model.backbone.embed_dim
    index = faiss.IndexFlatIP(dim)

    meta_rows: list[dict] = []
    n_seen = 0

    with torch.no_grad():
        for imgs, targets, paths in loader:
            imgs = imgs.to(dev, non_blocking=True)
            feat = model.backbone(imgs)
            feat = F.normalize(feat, dim=1)
            vec = feat.cpu().numpy().astype(np.float32)
            if not vec.flags["C_CONTIGUOUS"]:
                vec = np.ascontiguousarray(vec)
            index.add(vec)

            for j in range(vec.shape[0]):
                p = paths[j]
                t = int(targets[j].item())
                stem = Path(p).stem
                meta_rows.append(
                    {
                        "path": str(Path(p).resolve()),
                        "species": species_classes[t],
                        "class_idx": t,
                        "stem": stem,
                    }
                )

            n_seen += vec.shape[0]
            print(f"  encoded {n_seen} / {len(ds)}", flush=True)

    assert index.ntotal == len(meta_rows), "FAISS rows must match metadata"

    index_path = out_dir / "vectors.faiss"
    meta_path = out_dir / "metadata.json"
    manifest_path = out_dir / "manifest.json"

    faiss.write_index(index, str(index_path))
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta_rows, f, indent=2)

    manifest = {
        "checkpoint": str(ckpt_path),
        "data_dir": str(data_dir),
        "out_dir": str(out_dir),
        "num_vectors": int(index.ntotal),
        "embed_dim": int(dim),
        "index_type": "IndexFlatIP",
        "vector_norm": "L2",
        "species_classes": list(species_classes),
        "img_size": img_size,
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nWrote FAISS index ({index.ntotal} vectors, dim={dim}) → {index_path}")
    print(f"Metadata → {meta_path}")
    print(f"Manifest → {manifest_path}")


if __name__ == "__main__":
    main()
