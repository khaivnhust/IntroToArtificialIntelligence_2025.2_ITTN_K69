"""
Download Kaggle notebook output files to the local project.

Example:
    python scripts/download_kaggle_notebook_output.py ^
        --kernel nguyendanglong0708/your-notebook-slug

Notes:
    - Kaggle only exposes output from a saved/committed notebook version.
    - This script requires the Kaggle CLI and working Kaggle credentials.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "downloads" / "kaggle_notebook_output"
DEFAULT_VISUAL_DESTINATION = PROJECT_ROOT / "data" / "processed" / "visual_features_full.npz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download output from a Kaggle notebook.")
    parser.add_argument(
        "--kernel",
        required=True,
        help="Kaggle notebook ref in owner/slug form, for example nguyendanglong0708/my-notebook.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Local directory where all Kaggle output files will be downloaded.",
    )
    parser.add_argument(
        "--visual-destination",
        type=Path,
        default=DEFAULT_VISUAL_DESTINATION,
        help="Where to copy visual_features_full.npz if found. Use 'none' to skip copying.",
    )
    parser.add_argument(
        "--visual-filename",
        default="visual_features_full.npz",
        help="Output NPZ filename to search for after download.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting existing downloaded files and visual destination.",
    )
    return parser.parse_args()


def run_command(command: list[str]) -> None:
    print("$", " ".join(command))
    process = subprocess.Popen(
        command,
        cwd=PROJECT_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="")
    return_code = process.wait()
    if return_code != 0:
        raise SystemExit(return_code)


def require_kaggle_cli() -> None:
    if shutil.which("kaggle") is None:
        raise SystemExit(
            "Kaggle CLI not found. Install it first:\n"
            "  python -m pip install kaggle\n\n"
            "Then configure credentials with kaggle.json or Kaggle API token."
        )


def find_visual_npz(output_dir: Path, filename: str) -> Path | None:
    matches = sorted(output_dir.rglob(filename), key=lambda path: path.stat().st_size, reverse=True)
    if not matches:
        return None
    return matches[0]


def copy_visual_npz(source: Path, destination: Path, overwrite: bool) -> None:
    if destination.exists() and not overwrite:
        raise SystemExit(
            f"Destination already exists: {destination}\n"
            "Pass --overwrite if you want to replace it."
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    size_mb = destination.stat().st_size / (1024 * 1024)
    print(f"Copied {source} -> {destination} ({size_mb:.2f} MB)")


def main() -> None:
    args = parse_args()
    require_kaggle_cli()

    output_dir = args.output_dir.resolve()
    if output_dir.exists() and args.overwrite:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_command(["kaggle", "kernels", "output", args.kernel, "-p", str(output_dir)])

    print(f"\nDownloaded Kaggle output to: {output_dir}")
    downloaded_files = [path for path in output_dir.rglob("*") if path.is_file()]
    print(f"Downloaded file count: {len(downloaded_files):,}")
    for path in sorted(downloaded_files)[:30]:
        size_mb = path.stat().st_size / (1024 * 1024)
        print(f"- {path.relative_to(output_dir)} ({size_mb:.2f} MB)")
    if len(downloaded_files) > 30:
        print(f"... {len(downloaded_files) - 30:,} more file(s)")

    visual_npz = find_visual_npz(output_dir, args.visual_filename)
    if visual_npz is None:
        print(f"\nDid not find {args.visual_filename} under {output_dir}.")
        print("If the Kaggle notebook only wrote a folder, commit/save a notebook version first.")
        return

    print(f"\nFound visual feature file: {visual_npz}")
    visual_destination_raw = str(args.visual_destination).strip().lower()
    if visual_destination_raw in {"none", "skip", ""}:
        print("Skipping copy to project data directory.")
        return

    copy_visual_npz(visual_npz, args.visual_destination.resolve(), args.overwrite)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        raise SystemExit(130)
