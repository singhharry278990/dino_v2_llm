#!/usr/bin/env python3
"""
HTTP API for DINOv2 + FAISS similarity search (matches dinov2_faiss_search.html).

  pip install flask  # if needed

  python dinov2_faiss_server.py \\
    --checkpoint dinov2/herbarium_dinov2_final.pth \\
    --index_dir dinov2_faiss_index

Open http://127.0.0.1:8765/ or /ui for the search page (same-origin /query).
Or open dinov2/dinov2_faiss_search.html as a file (uses http://127.0.0.1:8765/query).
"""

from __future__ import annotations

import argparse
import io
import json
import os
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import faiss
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from predict_dino import load_model

try:
    from flask import Flask, jsonify, redirect, request, send_file
except ImportError as e:
    raise SystemExit("Install Flask: pip install flask") from e

_REPO_ROOT = Path(__file__).resolve().parent
_UI_HTML = _REPO_ROOT / "dinov2" / "dinov2_faiss_search.html"


def load_index_bundle(index_dir: Path):
    index_path = index_dir / "vectors.faiss"
    meta_path = index_dir / "metadata.json"
    if not index_path.is_file():
        raise FileNotFoundError(f"Missing {index_path} — run build_dinov2_faiss_index.py first")
    if not meta_path.is_file():
        raise FileNotFoundError(f"Missing {meta_path}")
    index = faiss.read_index(str(index_path))
    with open(meta_path, encoding="utf-8") as f:
        metadata = json.load(f)
    if index.ntotal != len(metadata):
        raise ValueError("FAISS ntotal does not match metadata length")
    return index, metadata


def create_app(checkpoint: Path, index_dir: Path, topk_default: int = 10):
    index, metadata = load_index_bundle(index_dir)
    model, transform, _species, device = load_model(model_path=str(checkpoint))
    model.eval()
    dim = model.backbone.embed_dim

    app = Flask(__name__)

    @app.after_request
    def cors(resp):
        resp.headers["Access-Control-Allow-Origin"] = "*"
        return resp

    @app.route("/", methods=["GET"])
    def root():
        return redirect("/ui")

    @app.route("/ui", methods=["GET"])
    def ui():
        if not _UI_HTML.is_file():
            return (
                jsonify(
                    {
                        "error": "UI file missing",
                        "expected": str(_UI_HTML),
                        "endpoints": ["/health", "POST /query"],
                    }
                ),
                404,
            )
        return send_file(_UI_HTML, mimetype="text/html; charset=utf-8")

    @app.route("/health", methods=["GET"])
    def health():
        return jsonify(
            {
                "ok": True,
                "ntotal": int(index.ntotal),
                "embed_dim": int(dim),
                "checkpoint": str(checkpoint),
            }
        )

    @app.route("/query", methods=["POST", "OPTIONS"])
    def query():
        if request.method == "OPTIONS":
            return "", 204
        if "image" not in request.files:
            return jsonify({"error": "expected multipart field 'image'"}), 400
        f = request.files["image"]
        raw = f.read()
        if not raw:
            return jsonify({"error": "empty upload"}), 400
        try:
            img = Image.open(io.BytesIO(raw)).convert("RGB")
        except Exception as err:
            return jsonify({"error": f"invalid image: {err}"}), 400

        k = request.args.get("k", type=int) or topk_default
        k = max(1, min(k, index.ntotal, 100))

        x = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            feat = model.backbone(x)
            feat = F.normalize(feat, dim=1)
        q = feat.cpu().numpy().astype(np.float32)
        scores, idxs = index.search(q, k)

        results = []
        for rank, (score, i) in enumerate(zip(scores[0].tolist(), idxs[0].tolist()), start=1):
            if i < 0:
                continue
            row = metadata[i]
            results.append(
                {
                    "rank": rank,
                    "species": row["species"],
                    "score": float(score),
                    "specimen_id": row.get("stem") or Path(row["path"]).stem,
                    "path": row["path"],
                }
            )

        return jsonify({"results": results})

    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="dinov2/herbarium_dinov2_final.pth")
    ap.add_argument("--index_dir", default="dinov2_faiss_index")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--topk", type=int, default=10)
    args = ap.parse_args()

    ckpt = Path(args.checkpoint).resolve()
    if not ckpt.is_file():
        alt = Path("dinov2/herbarium_dinov2_final.pth").resolve()
        if alt.is_file():
            ckpt = alt
        else:
            raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    index_dir = Path(args.index_dir).resolve()
    app = create_app(ckpt, index_dir, topk_default=args.topk)
    print(f"Serving FAISS query API on http://{args.host}:{args.port}")
    print(f"  Search UI:  http://{args.host}:{args.port}/ui")
    print(f"  checkpoint: {ckpt}")
    print(f"  index_dir:  {index_dir}")
    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()
