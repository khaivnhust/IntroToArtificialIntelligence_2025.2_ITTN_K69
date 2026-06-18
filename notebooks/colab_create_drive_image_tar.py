# %% [markdown]
# # Create Remaining H&M Image Tar on Google Drive
#
# Google Colab helper script.
#
# Run this after the full `raw/images/` folder is already complete on Google
# Drive and `raw/hm_images_075_095.tar` already exists. It packages the image
# folders not covered by that existing tar into several smaller tar chunks:
#
# ```text
# /content/drive/MyDrive/hm_recommender_rebuild/raw/hm_images_remaining_010_029.tar
# /content/drive/MyDrive/hm_recommender_rebuild/raw/hm_images_remaining_030_049.tar
# ...
# ```
#
# The GPU visual extraction notebook will extract both tar files into the same
# local `/content` image directory.
#
# This script does not download Kaggle data, does not rebuild parquet files,
# and does not run any model inference.

# %%
from __future__ import annotations

import shlex
import shutil
import subprocess
import time
from pathlib import Path


# =========================
# User configuration
# =========================

USE_GOOGLE_DRIVE = True

PROJECT_DIR = Path("/content/drive/MyDrive/hm_recommender_rebuild")
RAW_DIR = PROJECT_DIR / "raw"
IMAGE_DIR = RAW_DIR / "images"
EXISTING_PARTIAL_TARS = [RAW_DIR / "hm_images_075_095.tar"]
LOCAL_TMP_DIR = Path("/content/hm_tar_tmp")

OUTPUT_TAR_PREFIX = "hm_images_remaining"
FOLDERS_PER_TAR = 20

# Folders already covered by EXISTING_PARTIAL_TARS.
EXCLUDE_IMAGE_FOLDERS = {f"{folder_id:03d}" for folder_id in range(75, 96)}

# If False and a chunk tar already exists, the script leaves that chunk unchanged.
OVERWRITE_TARS = False

# Counting/listing tar contents can take time, but is useful for validation.
VERIFY_TARS = True

# Remove /content temporary tar files after each chunk is copied to Drive.
DELETE_LOCAL_TMP_AFTER_COPY = True


# %% [markdown]
# ## Mount Drive and validate input

# %%
def run_command(command: list[str]) -> None:
    print("$", " ".join(command))
    result = subprocess.run(
        command,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if result.stdout:
        print(result.stdout)
    result.check_returncode()


def chunked(values: list[str], size: int) -> list[list[str]]:
    return [values[index : index + size] for index in range(0, len(values), size)]


def output_tar_path(folders: list[str]) -> Path:
    return RAW_DIR / f"{OUTPUT_TAR_PREFIX}_{folders[0]}_{folders[-1]}.tar"


if USE_GOOGLE_DRIVE:
    from google.colab import drive  # type: ignore

    drive.mount("/content/drive")

if not IMAGE_DIR.exists():
    raise FileNotFoundError(f"Image directory not found: {IMAGE_DIR}")
missing_partial_tars = [path for path in EXISTING_PARTIAL_TARS if not path.exists()]
if missing_partial_tars:
    raise FileNotFoundError(
        "Expected existing partial tar file(s) not found: "
        + ", ".join(str(path) for path in missing_partial_tars)
    )

RAW_DIR.mkdir(parents=True, exist_ok=True)

print("IMAGE_DIR:", IMAGE_DIR)
print("EXISTING_PARTIAL_TARS:", [str(path) for path in EXISTING_PARTIAL_TARS])
print("EXCLUDE_IMAGE_FOLDERS:", sorted(EXCLUDE_IMAGE_FOLDERS))
print("LOCAL_TMP_DIR:", LOCAL_TMP_DIR)
print("FOLDERS_PER_TAR:", FOLDERS_PER_TAR)
print("OVERWRITE_TARS:", OVERWRITE_TARS)

available_folders = sorted(path.name for path in IMAGE_DIR.iterdir() if path.is_dir())
included_folders = [folder for folder in available_folders if folder not in EXCLUDE_IMAGE_FOLDERS]
excluded_existing_folders = [folder for folder in available_folders if folder in EXCLUDE_IMAGE_FOLDERS]

if not included_folders:
    raise RuntimeError("No image folders left to package after exclusions.")

print("Available image folders:", available_folders[:5], "...", available_folders[-5:])
print("Included folder count:", len(included_folders))
print("Excluded existing folder count:", len(excluded_existing_folders))
print("Included tail:", included_folders[-10:])


# %% [markdown]
# ## Create chunked tars on Drive

# %%
def create_tar_chunk(folders: list[str]) -> Path:
    output_tar = output_tar_path(folders)
    drive_tmp_tar = RAW_DIR / f"{output_tar.name}.tmp"
    local_tmp_tar = LOCAL_TMP_DIR / f"{output_tar.name}.tmp"

    if output_tar.exists() and not OVERWRITE_TARS:
        print("Chunk tar already exists, leaving it unchanged:", output_tar)
        return output_tar

    if output_tar.exists():
        print("Removing existing chunk tar:", output_tar)
        output_tar.unlink()
    if drive_tmp_tar.exists():
        print("Removing stale Drive temp tar:", drive_tmp_tar)
        drive_tmp_tar.unlink()
    if local_tmp_tar.exists():
        print("Removing stale local temp tar:", local_tmp_tar)
        local_tmp_tar.unlink()

    LOCAL_TMP_DIR.mkdir(parents=True, exist_ok=True)
    started_at = time.time()
    print(f"Creating chunk tar for folders {folders[0]}..{folders[-1]}")
    tar_members = [f"images/{folder}" for folder in folders]
    run_command(["tar", "-cf", str(local_tmp_tar), "-C", str(RAW_DIR), *tar_members])

    local_size_gb = local_tmp_tar.stat().st_size / (1024**3)
    print(f"Local temp tar size: {local_size_gb:.2f} GB")
    print("Copying completed local tar to Drive:", drive_tmp_tar)
    shutil.copy2(local_tmp_tar, drive_tmp_tar)
    drive_tmp_tar.replace(output_tar)

    elapsed = time.time() - started_at
    print(f"Created chunk tar in {elapsed:.1f}s: {output_tar}")

    if DELETE_LOCAL_TMP_AFTER_COPY and local_tmp_tar.exists():
        local_tmp_tar.unlink()

    return output_tar


tar_chunks = chunked(included_folders, FOLDERS_PER_TAR)
output_tars = [create_tar_chunk(folders) for folders in tar_chunks]

print("Created/available remaining tar chunks:")
for path in output_tars:
    size_gb = path.stat().st_size / (1024**3)
    print(f"- {path.name}: {size_gb:.2f} GB")


# %% [markdown]
# ## Quick validation

# %%
if VERIFY_TARS:
    for output_tar in output_tars:
        tar_q = shlex.quote(str(output_tar))
        print("Tar tail:", output_tar.name)
        run_command(["bash", "-lc", f"tar -tf {tar_q} | tail"])
        print("Tar entry count:", output_tar.name)
        run_command(["bash", "-lc", f"tar -tf {tar_q} | wc -l"])

print("Done. Next run notebooks/colab_extract_visual_features_only.py on a GPU runtime.")
