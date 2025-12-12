"""
This script is used only for downloading all the datasets from kaggel (and other sites).
All data will be found in dataset forlder
"""
import shutil
from pathlib import Path
import kagglehub

DATASET_FOLDER = Path("dataset")
DATASET_FOLDER.mkdir(exist_ok=True)

DATASETS = [
    ("larserikrisholm/dinosaur-image-dataset-15-species", "dino15"),
    ("cmglonly/simple-dinosurus-dataset", "simple"),  # from: https://www.kaggle.com/datasets/cmglonly/simple-dinosurus-dataset
    ("antaresl/jurassic-park-dinosaurs-dataset", "jurassic-park"),
]

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}


def safe_move_image(src: Path, dst_dir: Path, prefix: str) -> Path:
    """Move src into dst_dir, renaming if needed to avoid overwriting."""
    dst = dst_dir / src.name
    if dst.exists():
        stem, suf = src.stem, src.suffix.lower()
        i = 1
        while True:
            candidate = dst_dir / f"{prefix}_{stem}_{i}{suf}"
            if not candidate.exists():
                dst = candidate
                break
            i += 1
    shutil.move(str(src), str(dst))
    return dst


for dataset_id, prefix in DATASETS:
    download_path = Path(kagglehub.dataset_download(dataset_id))
    print(f"Downloaded {dataset_id} to: {download_path}")

    moved = 0
    for p in download_path.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            safe_move_image(p, DATASET_FOLDER, prefix)
            moved += 1

    # Remove the original downloaded dataset directory
    shutil.rmtree(download_path, ignore_errors=True)
    print(f"Moved {moved} images from {dataset_id} and removed: {download_path}")

print(f"Done. {DATASET_FOLDER} now contains only images.")

