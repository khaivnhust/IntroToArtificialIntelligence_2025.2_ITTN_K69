# %% [markdown]
# # Resume H&M Visual Feature Extraction on Kaggle
#
# Kaggle helper script.
#
# Use this when Colab was interrupted while creating ResNet-50 visual feature
# shards. To resume exactly, add these Kaggle inputs to the notebook:
#
# 1. Official H&M competition data:
#
# ```text
# /kaggle/input/h-and-m-personalized-fashion-recommendations/
# ```
#
# 2. Your processed data/shards exported from Colab/Drive as a Kaggle Dataset,
#    containing at least:
#
# ```text
# articles_cleaned.parquet
# visual_feature_shards/visual_features_00000.npz
# visual_feature_shards/visual_features_00001.npz
# ...
# ```
#
# The script writes new/merged outputs to:
#
# ```text
# /kaggle/working/hm_visual_features/
# ```
#
# Important: for true resume, `articles_cleaned.parquet`, `SHARD_SIZE`, and
# shard filenames must match the Colab run.

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

KAGGLE_INPUT_ROOT = Path("/kaggle/input")
KAGGLE_WORKING_ROOT = Path("/kaggle/working")

DEFAULT_HM_COMPETITION_DIR = KAGGLE_INPUT_ROOT / "h-and-m-personalized-fashion-recommendations"

# Leave None for auto-discovery under /kaggle/input. Set this manually only if
# your Kaggle input dataset uses an unusual path.
IMAGE_DIR: Path | None = None

# Auto-discovered under /kaggle/input if this path is set to None. This default
# matches the Kaggle Dataset path provided for the processed articles parquet.
ARTICLES_PATH: Path | None = (
    KAGGLE_INPUT_ROOT
    / "datasets"
    / "nguyendanglong0708"
    / "articles-cleaned-parquet"
    / "articles_cleaned.parquet"
)

# Auto-discovered under /kaggle/input if this list is empty.
EXISTING_SHARD_DIRS: list[Path] = []

OUTPUT_DIR = KAGGLE_WORKING_ROOT / "hm_visual_features"
SHARD_DIR = OUTPUT_DIR / "visual_feature_shards"
OUTPUT_NPZ = OUTPUT_DIR / "visual_features_full.npz"
OUTPUT_MANIFEST = OUTPUT_DIR / "visual_features_manifest.json"

# Must match the Colab script if you want shard-level resume.
SHARD_SIZE = 2000

VISUAL_BATCH_SIZE = 128
VISUAL_LIMIT: int | None = None
LOG_EVERY_BATCHES = 10

# Kaggle notebooks can emit noisy PyTorch multiprocessing cleanup warnings with
# num_workers > 0. Keep 0 for stability; increase only if you confirm it is clean
# in your runtime.
DATALOADER_NUM_WORKERS = 0

# If True, copy uploaded read-only shard files from /kaggle/input into
# /kaggle/working before processing. This makes the Kaggle output directory
# self-contained when the notebook finishes.
COPY_INPUT_SHARDS_TO_WORKING = True

# If True, recompute shard files even if they already exist.
OVERWRITE_SHARDS = False

# If True, merge all completed shards into OUTPUT_NPZ at the end.
MERGE_SHARDS_AT_END = True

# Leave False for resume. Turning this on may break shard compatibility with
# a Colab run if your processed parquet had a filtered article set.
ALLOW_RAW_ARTICLES_CSV_FALLBACK = False


# %% [markdown]
# ## Validate inputs and discover resume files

# %%
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm.auto import tqdm


def format_seconds(seconds: float) -> str:
    seconds = int(seconds)
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes}m {seconds}s"
    if minutes:
        return f"{minutes}m {seconds}s"
    return f"{seconds}s"


def first_existing(paths: Iterable[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def discover_articles_path() -> Path:
    if ARTICLES_PATH is not None:
        if not ARTICLES_PATH.exists():
            raise FileNotFoundError(f"Configured ARTICLES_PATH does not exist: {ARTICLES_PATH}")
        return ARTICLES_PATH

    candidates = sorted(KAGGLE_INPUT_ROOT.glob("**/articles_cleaned.parquet"))
    if candidates:
        print("Discovered articles_cleaned.parquet:", candidates[0])
        return candidates[0]

    if ALLOW_RAW_ARTICLES_CSV_FALLBACK:
        candidates = [DEFAULT_HM_COMPETITION_DIR / "articles.csv"]
        candidates.extend(sorted(KAGGLE_INPUT_ROOT.glob("**/articles.csv")))
        raw_articles = first_existing(candidates)
        if raw_articles is not None:
            print(
                "WARNING: using raw articles.csv fallback. This may not match "
                "Colab shard boundaries if Colab used a filtered processed parquet."
            )
            return raw_articles

    raise FileNotFoundError(
        "articles_cleaned.parquet not found under /kaggle/input. "
        "Upload the processed parquet from Colab/Drive as a Kaggle Dataset, "
        "or set ALLOW_RAW_ARTICLES_CSV_FALLBACK=True to start from raw articles.csv."
    )


def discover_shard_dirs() -> list[Path]:
    configured = [path for path in EXISTING_SHARD_DIRS if path.exists()]
    if configured:
        return configured

    discovered = sorted(
        path
        for path in KAGGLE_INPUT_ROOT.glob("**/visual_feature_shards")
        if path.is_dir()
    )
    if discovered:
        print("Discovered existing shard dir(s):")
        for path in discovered:
            print("-", path)
    else:
        print("No existing shard dir discovered; extraction will start from missing shards.")
    return discovered


def is_hm_image_dir(path: Path) -> bool:
    if not path.is_dir():
        return False
    numeric_children = [
        child
        for child in path.iterdir()
        if child.is_dir() and child.name.isdigit() and len(child.name) == 3
    ]
    return bool(numeric_children)


def discover_image_dir() -> Path:
    if IMAGE_DIR is not None:
        if not IMAGE_DIR.exists():
            raise FileNotFoundError(f"Configured IMAGE_DIR does not exist: {IMAGE_DIR}")
        return IMAGE_DIR

    candidates = [DEFAULT_HM_COMPETITION_DIR / "images"]
    candidates.extend(sorted(KAGGLE_INPUT_ROOT.glob("**/images")))

    valid_candidates = []
    for candidate in candidates:
        if candidate.exists() and is_hm_image_dir(candidate):
            valid_candidates.append(candidate)

    if valid_candidates:
        print("Discovered H&M image directory:", valid_candidates[0])
        return valid_candidates[0]

    visible_inputs = sorted(path for path in KAGGLE_INPUT_ROOT.glob("*") if path.exists())
    raise FileNotFoundError(
        "H&M image directory not found under /kaggle/input.\n"
        "Add the official H&M competition dataset to this Kaggle notebook, or set IMAGE_DIR manually.\n"
        "Visible /kaggle/input entries:\n"
        + "\n".join(f"- {path}" for path in visible_inputs)
    )


def read_article_ids(path: Path) -> list[int]:
    if path.suffix == ".parquet":
        articles = pd.read_parquet(path, columns=["article_id"])
    else:
        articles = pd.read_csv(path, usecols=["article_id"])

    article_ids = pd.to_numeric(articles["article_id"], errors="raise").astype("int64")
    article_ids = article_ids.sort_values(kind="mergesort")
    if VISUAL_LIMIT is not None:
        article_ids = article_ids.head(VISUAL_LIMIT)
    return article_ids.astype(int).tolist()


RESOLVED_IMAGE_DIR = discover_image_dir()
RESOLVED_ARTICLES_PATH = discover_articles_path()
INPUT_SHARD_DIRS = discover_shard_dirs()

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SHARD_DIR.mkdir(parents=True, exist_ok=True)

print("IMAGE_DIR:", RESOLVED_IMAGE_DIR)
print("ARTICLES:", RESOLVED_ARTICLES_PATH)
print("INPUT_SHARD_DIRS:", [str(path) for path in INPUT_SHARD_DIRS])
print("OUTPUT_DIR:", OUTPUT_DIR)
print("SHARD_DIR:", SHARD_DIR)
print("OUTPUT_NPZ:", OUTPUT_NPZ)


# %% [markdown]
# ## Copy existing shards to working directory

# %%
def copy_existing_shards_to_working() -> int:
    if not COPY_INPUT_SHARDS_TO_WORKING:
        print("COPY_INPUT_SHARDS_TO_WORKING=False, leaving input shards read-only.")
        return 0

    copied = 0
    started_at = time.time()
    for input_dir in INPUT_SHARD_DIRS:
        shard_files = sorted(input_dir.glob("visual_features_*.npz"))
        print(f"Copying {len(shard_files):,} shard file(s) from {input_dir}")
        for index, source in enumerate(shard_files, start=1):
            destination = SHARD_DIR / source.name
            if destination.exists():
                continue
            shutil.copy2(source, destination)
            copied += 1
            if copied == 1 or copied % 10 == 0 or index == len(shard_files):
                print(
                    "Shard copy progress: "
                    f"copied={copied:,}; current={source.name}; "
                    f"elapsed={format_seconds(time.time() - started_at)}"
                )

    print(f"Copied {copied:,} existing shard file(s) into {SHARD_DIR}")
    return copied


copy_existing_shards_to_working()


# %% [markdown]
# ## Load articles and define image dataset

# %%
article_ids_all = read_article_ids(RESOLVED_ARTICLES_PATH)
article_count = len(article_ids_all)
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
    print("WARNING: CUDA is not available. Enable GPU accelerator in Kaggle.")

torch.backends.cudnn.benchmark = device.type == "cuda"

try:
    weights = models.ResNet50_Weights.DEFAULT
    model = models.resnet50(weights=weights)
except Exception as exc:
    raise RuntimeError(
        "Failed to load ResNet-50 pretrained weights. In Kaggle, enable Internet "
        "for this notebook or attach a dataset containing the cached torchvision weights."
    ) from exc

setattr(model, "fc", torch.nn.Identity())
model.eval().to(device)


# %% [markdown]
# ## Extract missing feature shards

# %%
def shard_path(shard_index: int) -> Path:
    return SHARD_DIR / f"visual_features_{shard_index:05d}.npz"


def iter_shards(total_rows: int, shard_size: int) -> Iterable[tuple[int, int, int]]:
    shard_count = math.ceil(total_rows / shard_size)
    for shard_index in range(shard_count):
        start = shard_index * shard_size
        end = min(start + shard_size, total_rows)
        yield shard_index, start, end


def extract_one_shard(shard_index: int, shard_article_ids: list[int]) -> dict[str, int | float | str]:
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

    dataset = ArticleImageDataset(shard_article_ids, RESOLVED_IMAGE_DIR)
    loader = DataLoader(
        dataset,
        batch_size=VISUAL_BATCH_SIZE,
        shuffle=False,
        num_workers=DATALOADER_NUM_WORKERS,
        collate_fn=collate_batch,
        pin_memory=(device.type == "cuda"),
    )

    features: dict[str, np.ndarray] = {}
    started_at = time.time()
    processed_batches = 0
    processed_images = 0
    with torch.no_grad():
        for batch_article_ids, images in tqdm(
            loader,
            total=len(loader),
            desc=f"Shard {shard_index:05d}",
        ):
            processed_batches += 1
            if not batch_article_ids:
                if processed_batches == 1 or processed_batches % LOG_EVERY_BATCHES == 0:
                    print(
                        f"Shard {shard_index:05d}: batch {processed_batches:,}/{len(loader):,} "
                        f"had no existing images; vectors={len(features):,}; "
                        f"elapsed={format_seconds(time.time() - started_at)}"
                    )
                continue
            images = images.to(device, non_blocking=True)
            embeddings = model(images).detach().cpu().numpy().astype(np.float32)
            for article_id, embedding in zip(batch_article_ids, embeddings):
                features[str(article_id)] = embedding
            processed_images += len(batch_article_ids)
            if (
                processed_batches == 1
                or processed_batches % LOG_EVERY_BATCHES == 0
                or processed_batches == len(loader)
            ):
                print(
                    f"Shard {shard_index:05d}: batch {processed_batches:,}/{len(loader):,}; "
                    f"valid_images={processed_images:,}; vectors={len(features):,}; "
                    f"elapsed={format_seconds(time.time() - started_at)}"
                )

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
    print(f"\nProcessing shard {shard_index:05d}: rows {start:,}..{end - 1:,}")
    summary = extract_one_shard(shard_index, article_ids_all[start:end])
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
    for index, path in enumerate(tqdm(shard_files, desc="Merging shards"), start=1):
        with np.load(path) as shard:
            for key in shard.files:
                merged[key] = shard[key]
        if index == 1 or index % 10 == 0 or index == len(shard_files):
            print(
                f"Merging shards: {index:,}/{len(shard_files):,}; "
                f"vectors={len(merged):,}; elapsed={format_seconds(time.time() - started_at)}"
            )

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
    "environment": "kaggle",
    "articles_path": str(RESOLVED_ARTICLES_PATH),
    "image_dir": str(RESOLVED_IMAGE_DIR),
    "input_shard_dirs": [str(path) for path in INPUT_SHARD_DIRS],
    "output_npz": str(OUTPUT_NPZ),
    "shard_dir": str(SHARD_DIR),
    "article_count": article_count,
    "available_vectors_in_shards": available_vectors,
    "coverage_from_shards": available_vectors / max(article_count, 1),
    "visual_batch_size": VISUAL_BATCH_SIZE,
    "dataloader_num_workers": DATALOADER_NUM_WORKERS,
    "shard_size": SHARD_SIZE,
    "visual_limit": VISUAL_LIMIT,
    "device": str(device),
    "shards": shard_summaries,
    "merge_summary": merge_summary,
    "created_at_unix": time.time(),
}

OUTPUT_MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
print("Manifest:", OUTPUT_MANIFEST)
print(
    json.dumps(
        {
            key: manifest[key]
            for key in [
                "article_count",
                "available_vectors_in_shards",
                "coverage_from_shards",
                "output_npz",
            ]
        },
        indent=2,
    )
)

if OUTPUT_NPZ.exists():
    size_mb = OUTPUT_NPZ.stat().st_size / (1024 * 1024)
    print(f"Output NPZ size: {size_mb:.2f} MB")
