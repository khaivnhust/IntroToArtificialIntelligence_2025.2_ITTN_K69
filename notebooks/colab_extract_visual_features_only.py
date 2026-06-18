# %% [markdown]
# # Extract Full Visual Features Only
#
# Google Colab helper script.
#
# Use this after `notebooks/colab_create_drive_image_tar.py` has created:
#
# ```text
# raw/hm_images_remaining_except_075_095.tar
# ```
#
# and `raw/hm_images_075_095.tar` already exists. This script extracts those
# tar files to Colab local disk, then extracts ResNet-50 visual features and
# writes:
#
# ```text
# processed/visual_features_full.npz
# processed/visual_features_manifest.json
# ```
#
# It does not download Kaggle data, does not rebuild parquet files, and does
# not copy images folder-by-folder from Drive.
#
# The extraction is resumable through shard files. If Colab disconnects, rerun
# the script and it will skip completed shards.
#
# Important: `/content` is ephemeral. If Colab reconnects to a new VM, rerun
# this script; completed feature shards on Drive will still be skipped.

# %%
from __future__ import annotations

import json
import math
import shutil
import time
from pathlib import Path
from typing import Iterable


# =========================
# User configuration
# =========================

USE_GOOGLE_DRIVE = True

PROJECT_DIR = (
    Path("/content/drive/MyDrive/hm_recommender_rebuild")
    if USE_GOOGLE_DRIVE
    else Path("/content/hm_recommender_rebuild")
)
RAW_DIR = PROJECT_DIR / "raw"
PROCESSED_DIR = PROJECT_DIR / "processed"

ARTICLES_PATH = PROCESSED_DIR / "articles_cleaned.parquet"
DRIVE_IMAGE_TARS = [
    RAW_DIR / "hm_images_remaining_except_075_095.tar",
    RAW_DIR / "hm_images_075_095.tar",
]

# Optional fallback if you already created one full tar instead of partial tars.
FALLBACK_FULL_IMAGE_TAR = RAW_DIR / "hm_images.tar"
USE_FALLBACK_FULL_TAR = True

LOCAL_RAW_DIR = Path("/content/hm_recommender_rebuild/raw")
LOCAL_IMAGE_DIR = Path("/content/hm_recommender_rebuild/raw/images")
LOCAL_IMAGE_DIR_OVERRIDE: str | None = None

# If True, remove /content local images before extracting the tar.
OVERWRITE_LOCAL_IMAGES = False

OUTPUT_NPZ = PROCESSED_DIR / "visual_features_full.npz"
OUTPUT_MANIFEST = PROCESSED_DIR / "visual_features_manifest.json"
SHARD_DIR = PROCESSED_DIR / "visual_feature_shards"

# Tune if Colab runs out of VRAM.
VISUAL_BATCH_SIZE = 128

# Number of article rows per shard. Smaller shards resume more granularly.
SHARD_SIZE = 2000

# None = all articles. Use a number for a quick test.
VISUAL_LIMIT: int | None = None

# If True, recompute shard files even if they already exist.
OVERWRITE_SHARDS = False

# If True, merge all completed shards into OUTPUT_NPZ at the end.
MERGE_SHARDS_AT_END = True


# %% [markdown]
# ## Install dependencies and mount Drive

# %%
import subprocess
import sys


def run_command(command: list[str]) -> None:
    print("$", " ".join(command))
    subprocess.run(command, check=True)


run_command([sys.executable, "-m", "pip", "install", "-q", "polars", "pyarrow", "pillow", "tqdm"])

if USE_GOOGLE_DRIVE:
    from google.colab import drive  # type: ignore

    drive.mount("/content/drive")

if not ARTICLES_PATH.exists():
    raise FileNotFoundError(f"articles_cleaned.parquet not found: {ARTICLES_PATH}")


def resolve_image_tars() -> list[Path]:
    if LOCAL_IMAGE_DIR_OVERRIDE:
        return []

    missing_tars = [path for path in DRIVE_IMAGE_TARS if not path.exists()]
    if not missing_tars:
        return DRIVE_IMAGE_TARS

    if USE_FALLBACK_FULL_TAR and FALLBACK_FULL_IMAGE_TAR.exists():
        print("Some configured partial tars are missing, using fallback full tar instead:")
        for path in missing_tars:
            print("-", path)
        return [FALLBACK_FULL_IMAGE_TAR]

    raise FileNotFoundError(
        "Image tar file(s) not found: "
        + ", ".join(str(path) for path in missing_tars)
        + "\nRun notebooks/colab_create_drive_image_tar.py first, "
        "or provide FALLBACK_FULL_IMAGE_TAR."
    )


ACTIVE_IMAGE_TARS = resolve_image_tars()

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
SHARD_DIR.mkdir(parents=True, exist_ok=True)

print("ARTICLES_PATH:", ARTICLES_PATH)
print("ACTIVE_IMAGE_TARS:", [str(path) for path in ACTIVE_IMAGE_TARS])
print("LOCAL_IMAGE_DIR:", LOCAL_IMAGE_DIR)
print("OUTPUT_NPZ:", OUTPUT_NPZ)
print("SHARD_DIR:", SHARD_DIR)


# %% [markdown]
# ## Extract image tar to local disk

# %%
def normalize_top_level_image_folders() -> None:
    LOCAL_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    moved_folders = []
    for child in sorted(LOCAL_RAW_DIR.iterdir()):
        if not child.is_dir() or not child.name.isdigit() or len(child.name) != 3:
            continue

        destination = LOCAL_IMAGE_DIR / child.name
        if destination.exists():
            print("Top-level image folder already exists under images/, leaving source:", child)
            continue

        shutil.move(str(child), str(destination))
        moved_folders.append(child.name)

    if moved_folders:
        print("Moved top-level image folders into images/:", moved_folders[:5], "...", moved_folders[-5:])


def prepare_local_image_dir() -> Path:
    if LOCAL_IMAGE_DIR_OVERRIDE:
        image_dir = Path(LOCAL_IMAGE_DIR_OVERRIDE)
        if not image_dir.exists():
            raise FileNotFoundError(f"LOCAL_IMAGE_DIR_OVERRIDE does not exist: {image_dir}")
        print("Using local image override:", image_dir)
        return image_dir

    if LOCAL_IMAGE_DIR.exists() and not OVERWRITE_LOCAL_IMAGES:
        print("Local images already exist, reusing:", LOCAL_IMAGE_DIR)
        return LOCAL_IMAGE_DIR

    if LOCAL_IMAGE_DIR.exists() and OVERWRITE_LOCAL_IMAGES:
        print("Removing existing local images:", LOCAL_IMAGE_DIR)
        if not str(LOCAL_IMAGE_DIR).startswith("/content/hm_recommender_rebuild/raw/images"):
            raise RuntimeError(f"Refusing to remove unexpected local path: {LOCAL_IMAGE_DIR}")
        shutil.rmtree(LOCAL_IMAGE_DIR)

    LOCAL_RAW_DIR.mkdir(parents=True, exist_ok=True)
    for tar_path in ACTIVE_IMAGE_TARS:
        print("Extracting image tar to local disk:", tar_path)
        run_command(["tar", "-xf", str(tar_path), "-C", str(LOCAL_RAW_DIR)])

    normalize_top_level_image_folders()

    if not LOCAL_IMAGE_DIR.exists():
        raise FileNotFoundError(f"Local image extraction failed: {LOCAL_IMAGE_DIR}")

    print("Local image extraction complete:", LOCAL_IMAGE_DIR)
    return LOCAL_IMAGE_DIR


IMAGE_DIR = prepare_local_image_dir()
print("ACTIVE IMAGE_DIR:", IMAGE_DIR)


# %% [markdown]
# ## Load articles and define image dataset

# %%
import numpy as np
import polars as pl
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm.auto import tqdm


articles_df = pl.read_parquet(ARTICLES_PATH).sort("article_id")
if VISUAL_LIMIT is not None:
    articles_df = articles_df.head(VISUAL_LIMIT)

article_count = articles_df.height
print(f"Articles to process: {article_count:,}")


class ArticleImageDataset(Dataset):
    def __init__(self, article_ids: list[int], image_dir: Path) -> None:
        self.article_ids = [str(article_id) for article_id in article_ids]
        self.image_dir = image_dir
        self.transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    def __len__(self) -> int:
        return len(self.article_ids)

    def __getitem__(self, index: int):
        article_id = self.article_ids[index]
        padded = article_id.zfill(10)
        image_path = self.image_dir / padded[:3] / f"{padded}.jpg"
        if not image_path.exists():
            return article_id, torch.empty(0), False

        image = Image.open(image_path).convert("RGB")
        return article_id, self.transform(image), True


def collate_batch(batch):
    article_ids, images, exists = zip(*batch)
    valid_indices = [index for index, flag in enumerate(exists) if flag]
    if not valid_indices:
        return [], torch.empty(0)
    valid_article_ids = [article_ids[index] for index in valid_indices]
    valid_images = torch.stack([images[index] for index in valid_indices])
    return valid_article_ids, valid_images


# %% [markdown]
# ## Build ResNet-50 feature extractor

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)
if device.type != "cuda":
    print("WARNING: CUDA is not available. This will be very slow on CPU.")

weights = models.ResNet50_Weights.DEFAULT
model = models.resnet50(weights=weights)
setattr(model, "fc", torch.nn.Identity())
model.eval().to(device)


# %% [markdown]
# ## Extract feature shards

# %%
def shard_path(shard_index: int) -> Path:
    return SHARD_DIR / f"visual_features_{shard_index:05d}.npz"


def iter_shards(total_rows: int, shard_size: int) -> Iterable[tuple[int, int, int]]:
    shard_count = math.ceil(total_rows / shard_size)
    for shard_index in range(shard_count):
        start = shard_index * shard_size
        end = min(start + shard_size, total_rows)
        yield shard_index, start, end


def extract_one_shard(shard_index: int, shard_articles: pl.DataFrame) -> dict[str, int | float | str]:
    path = shard_path(shard_index)
    if path.exists() and not OVERWRITE_SHARDS:
        with np.load(path) as existing:
            count = len(existing.files)
        print(f"Shard {shard_index:05d} exists, skipping ({count:,} vectors).")
        return {
            "shard_index": shard_index,
            "status": "skipped_existing",
            "vectors": count,
            "path": str(path),
        }

    article_ids = shard_articles["article_id"].to_list()
    dataset = ArticleImageDataset(article_ids, IMAGE_DIR)
    loader = DataLoader(
        dataset,
        batch_size=VISUAL_BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        collate_fn=collate_batch,
        pin_memory=(device.type == "cuda"),
    )

    features: dict[str, np.ndarray] = {}
    started_at = time.time()
    with torch.no_grad():
        for batch_article_ids, images in tqdm(
            loader,
            total=len(loader),
            desc=f"Shard {shard_index:05d}",
        ):
            if not batch_article_ids:
                continue
            images = images.to(device, non_blocking=True)
            embeddings = model(images).detach().cpu().numpy().astype(np.float32)
            for article_id, embedding in zip(batch_article_ids, embeddings):
                features[str(article_id)] = embedding

    np.savez_compressed(path, **features)
    elapsed = time.time() - started_at
    print(f"Shard {shard_index:05d}: wrote {len(features):,} vectors to {path} in {elapsed:.1f}s")
    return {
        "shard_index": shard_index,
        "status": "written",
        "vectors": len(features),
        "seconds": elapsed,
        "path": str(path),
    }


shard_summaries = []
for shard_index, start, end in iter_shards(article_count, SHARD_SIZE):
    shard_articles = articles_df.slice(start, end - start)
    print(f"\nProcessing shard {shard_index:05d}: rows {start:,}..{end - 1:,}")
    summary = extract_one_shard(shard_index, shard_articles)
    shard_summaries.append(summary)

print("Shard extraction finished.")


# %% [markdown]
# ## Merge shards into one NPZ

# %%
def merge_shards() -> dict[str, int | str | float]:
    shard_files = sorted(SHARD_DIR.glob("visual_features_*.npz"))
    if not shard_files:
        raise FileNotFoundError(f"No shard files found in {SHARD_DIR}")

    merged: dict[str, np.ndarray] = {}
    started_at = time.time()
    for path in tqdm(shard_files, desc="Merging shards"):
        with np.load(path) as shard:
            for key in shard.files:
                merged[key] = shard[key]

    np.savez_compressed(OUTPUT_NPZ, **merged)
    elapsed = time.time() - started_at
    print(f"Merged {len(merged):,} vectors into {OUTPUT_NPZ} in {elapsed:.1f}s")
    return {
        "vectors": len(merged),
        "shards": len(shard_files),
        "seconds": elapsed,
        "output_npz": str(OUTPUT_NPZ),
    }


merge_summary = None
if MERGE_SHARDS_AT_END:
    merge_summary = merge_shards()
else:
    print("MERGE_SHARDS_AT_END=False, leaving shard files unmerged.")


# %% [markdown]
# ## Write manifest and quick checks

# %%
available_vectors = 0
for path in sorted(SHARD_DIR.glob("visual_features_*.npz")):
    with np.load(path) as shard:
        available_vectors += len(shard.files)

manifest = {
    "articles_path": str(ARTICLES_PATH),
    "image_dir": str(IMAGE_DIR),
    "image_tars": [str(path) for path in ACTIVE_IMAGE_TARS],
    "output_npz": str(OUTPUT_NPZ),
    "shard_dir": str(SHARD_DIR),
    "article_count": article_count,
    "available_vectors_in_shards": available_vectors,
    "coverage_from_shards": available_vectors / max(article_count, 1),
    "visual_batch_size": VISUAL_BATCH_SIZE,
    "shard_size": SHARD_SIZE,
    "visual_limit": VISUAL_LIMIT,
    "device": str(device),
    "shards": shard_summaries,
    "merge_summary": merge_summary,
    "created_at_unix": time.time(),
}

OUTPUT_MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
print("Manifest:", OUTPUT_MANIFEST)
print(json.dumps({k: manifest[k] for k in [
    "article_count",
    "available_vectors_in_shards",
    "coverage_from_shards",
    "output_npz",
]}, indent=2))

if OUTPUT_NPZ.exists():
    size_mb = OUTPUT_NPZ.stat().st_size / (1024 * 1024)
    print(f"Output NPZ size: {size_mb:.2f} MB")
