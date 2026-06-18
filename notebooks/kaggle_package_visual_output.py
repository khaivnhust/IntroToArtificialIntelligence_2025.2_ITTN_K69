# %% [markdown]
# # Package Kaggle Visual Feature Output for Download
#
# Run this in the Kaggle notebook after visual feature extraction finishes.
# It creates one downloadable zip file containing the final NPZ and manifest:
#
# ```text
# /kaggle/working/hm_visual_features.zip
# ```

# %%
from __future__ import annotations

import shutil
from pathlib import Path

from IPython.display import FileLink, display


OUTPUT_DIR = Path("/kaggle/working/hm_visual_features")
ZIP_BASE = Path("/kaggle/working/hm_visual_features")
ZIP_PATH = ZIP_BASE.with_suffix(".zip")
FILES_TO_PACKAGE = [
    OUTPUT_DIR / "visual_features_full.npz",
    OUTPUT_DIR / "visual_features_manifest.json",
]
PACKAGE_DIR = Path("/kaggle/working/hm_visual_features_download")


missing = [path for path in FILES_TO_PACKAGE if not path.exists()]
if missing:
    raise FileNotFoundError(
        "Expected final output file(s) not found:\n"
        + "\n".join(f"- {path}" for path in missing)
    )

print("Files to package:")
for path in FILES_TO_PACKAGE:
    size_mb = path.stat().st_size / (1024 * 1024)
    print(f"- {path.name} ({size_mb:.2f} MB)")

if ZIP_PATH.exists():
    print("Removing existing zip:", ZIP_PATH)
    ZIP_PATH.unlink()
if PACKAGE_DIR.exists():
    print("Removing existing package dir:", PACKAGE_DIR)
    shutil.rmtree(PACKAGE_DIR)

PACKAGE_DIR.mkdir(parents=True, exist_ok=True)
for path in FILES_TO_PACKAGE:
    shutil.copy2(path, PACKAGE_DIR / path.name)

print("Creating zip with final visual feature output only.")
created_zip = shutil.make_archive(str(ZIP_BASE), "zip", root_dir=PACKAGE_DIR)
created_zip_path = Path(created_zip)
size_gb = created_zip_path.stat().st_size / (1024**3)

print(f"Created: {created_zip_path}")
print(f"Zip size: {size_gb:.2f} GB")
display(FileLink(str(created_zip_path)))
