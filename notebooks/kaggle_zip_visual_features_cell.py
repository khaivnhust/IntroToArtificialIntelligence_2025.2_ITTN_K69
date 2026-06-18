from pathlib import Path
import shutil

OUTPUT_DIR = Path("/kaggle/working/hm_visual_features")
PACKAGE_DIR = Path("/kaggle/working/hm_visual_features_download")
ZIP_PATH = Path("/kaggle/working/hm_visual_features.zip")

if PACKAGE_DIR.exists():
    shutil.rmtree(PACKAGE_DIR)
PACKAGE_DIR.mkdir(parents=True, exist_ok=True)

for name in ["visual_features_full.npz", "visual_features_manifest.json"]:
    src = OUTPUT_DIR / name
    if not src.exists():
        raise FileNotFoundError(f"Missing file: {src}")
    size_mb = src.stat().st_size / (1024 * 1024)
    print(f"{src} | {size_mb:.2f} MB")
    shutil.copy2(src, PACKAGE_DIR / name)

if ZIP_PATH.exists():
    ZIP_PATH.unlink()

created_zip = shutil.make_archive(
    "/kaggle/working/hm_visual_features",
    "zip",
    root_dir=PACKAGE_DIR,
)

print("Created:", created_zip)
