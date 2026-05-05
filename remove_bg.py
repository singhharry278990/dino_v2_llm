from PIL import Image
import torch
from torchvision import transforms
from transformers import AutoModelForImageSegmentation
import os
from pathlib import Path


device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = AutoModelForImageSegmentation.from_pretrained('briaai/RMBG-2.0', trust_remote_code=True).eval().to(device)

# Data settings
image_size = (1024, 1024)
transform_image = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def remove_background(image_path: str, output_path: str | None = None) -> Image.Image | None:
    image = Image.open(image_path)
    input_images = transform_image(image).unsqueeze(0).to(device)
    with torch.no_grad():
        preds = model(input_images)[-1].sigmoid().cpu()
    pred = preds[0].squeeze()
    pred_pil = transforms.ToPILImage()(pred)
    mask = pred_pil.resize(image.size)
    image.putalpha(mask)
    image.save(output_path)
    return image

EXTENSIONS = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp")

INPUT_DIR = "Training_Data"
OUTPUT_DIR = "noiseless_datasets"


def is_clean_output(path: str) -> bool:
    """Return True if file exists, has size > 0, and is a valid image."""
    p = Path(path)
    if not p.is_file() or p.stat().st_size == 0:
        return False
    try:
        with Image.open(path) as im:
            im.verify()
        return True
    except Exception:
        return False


if not os.path.isdir(INPUT_DIR):
    print(f"Input directory not found: {INPUT_DIR}")
    exit()

input_root = Path(INPUT_DIR)
output_root = Path(OUTPUT_DIR)

images = [p for p in input_root.rglob("*") if p.is_file() and p.suffix.lower() in EXTENSIONS]
if not images:
    print(f"No images found in {INPUT_DIR}")
    exit()

print(f"Removing background from {len(images)} image(s) in {INPUT_DIR} -> {OUTPUT_DIR}")
skipped = 0
name_counts: dict[tuple[str, str], int] = {}
for in_path_obj in sorted(images):
    rel = in_path_obj.relative_to(input_root)
    rel_parent = rel.parent
    base_name = in_path_obj.stem
    stem_key = (str(rel_parent), base_name)
    count = name_counts.get(stem_key, 0)
    out_name = f"{base_name}.png" if count == 0 else f"{base_name}__{count}.png"
    name_counts[stem_key] = count + 1
    out_path_obj = output_root / rel_parent / out_name
    out_path_obj.parent.mkdir(parents=True, exist_ok=True)
    in_path = str(in_path_obj)
    out_path = str(out_path_obj)
    if is_clean_output(out_path):
        skipped += 1
        continue
    remove_background(in_path, output_path=out_path)
    rel_out = out_path_obj.relative_to(output_root)
    print("  saved:", rel_out)
if skipped:
    print(f"  (skipped {skipped} already clean)")