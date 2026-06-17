"""
Run the report training/evaluation pipeline end to end.

The runner is intentionally a thin orchestration layer around the existing
project scripts. It keeps one command for report runs while preserving the
separate training and testing outputs needed for documentation.

Examples
--------
    python scripts/run_report_pipeline.py --profile smoke
    python scripts/run_report_pipeline.py --profile report
    python scripts/run_report_pipeline.py --profile full

To resume only testing from existing checkpoints:
    python scripts/run_report_pipeline.py --profile report --stages test
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable, Sequence


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "processed"
DEFAULT_VISUAL_FEATURES = DEFAULT_DATA_DIR / "visual_features_sample.npz"
DEFAULT_CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "reports" / "report_pipeline"

ALL_STAGES = ("check-data", "train-hybrid", "train-compare", "test")


@dataclass(frozen=True)
class RunProfile:
    max_train_rows: int | None
    hybrid_epochs: int
    compare_epochs: int
    batch_size: int
    num_negatives: int
    max_eval_users: int | None
    negative_candidates: int
    heartbeat_seconds: int
    log_every_batches: int


PROFILES: dict[str, RunProfile] = {
    "smoke": RunProfile(
        max_train_rows=2_000,
        hybrid_epochs=1,
        compare_epochs=1,
        batch_size=256,
        num_negatives=1,
        max_eval_users=5,
        negative_candidates=100,
        heartbeat_seconds=5,
        log_every_batches=1,
    ),
    "report": RunProfile(
        max_train_rows=200_000,
        hybrid_epochs=3,
        compare_epochs=3,
        batch_size=2_048,
        num_negatives=2,
        max_eval_users=1_000,
        negative_candidates=1_000,
        heartbeat_seconds=30,
        log_every_batches=0,
    ),
    "full": RunProfile(
        max_train_rows=None,
        hybrid_epochs=5,
        compare_epochs=5,
        batch_size=4_096,
        num_negatives=4,
        max_eval_users=None,
        negative_candidates=1_000,
        heartbeat_seconds=30,
        log_every_batches=0,
    ),
}


def parse_optional_positive_int(value: str | None) -> int | None:
    if value is None:
        return None
    normalized = value.strip().lower()
    if normalized in {"none", "all", "0"}:
        return None
    parsed = int(normalized)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def parse_stages(value: str) -> list[str]:
    if value.strip().lower() == "all":
        return list(ALL_STAGES)

    stages = [stage.strip().lower() for stage in value.split(",") if stage.strip()]
    unknown = sorted(set(stages) - set(ALL_STAGES))
    if unknown:
        raise argparse.ArgumentTypeError(
            f"Unknown stage(s): {', '.join(unknown)}. Valid stages: {', '.join(ALL_STAGES)}"
        )
    return stages


def build_profile(args: argparse.Namespace) -> RunProfile:
    profile = PROFILES[args.profile]
    overrides = {}
    for field_name in (
        "max_train_rows",
        "hybrid_epochs",
        "compare_epochs",
        "batch_size",
        "num_negatives",
        "max_eval_users",
        "negative_candidates",
        "heartbeat_seconds",
        "log_every_batches",
    ):
        value = getattr(args, field_name)
        if value is not None:
            overrides[field_name] = value
    return replace(profile, **overrides)


def add_optional_int(command: list[str], flag: str, value: int | None) -> None:
    if value is not None:
        command.extend([flag, str(value)])


def add_eval_users(command: list[str], value: int | None) -> None:
    command.extend(["--max-eval-users", str(value) if value is not None else "all"])


def command_to_text(command: Sequence[str]) -> str:
    return " ".join(f'"{part}"' if " " in part else part for part in command)


def validate_required_files(data_dir: Path, visual_features: Path) -> None:
    required_files = [
        data_dir / "articles_cleaned.parquet",
        data_dir / "customers_cleaned.parquet",
        data_dir / "hm_train.parquet",
        data_dir / "hm_test.parquet",
        visual_features,
    ]
    missing = [path for path in required_files if not path.exists()]
    if missing:
        print("Missing required input files:")
        for path in missing:
            print(f"  - {path}")
        raise SystemExit(1)

    print("Input files are present:")
    for path in required_files:
        print(f"  - {path}")


def run_command(stage: str, command: Sequence[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started_at = time.time()
    print(f"\n=== {stage} ===")
    print(command_to_text(command))
    print(f"Log: {log_path}")

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("MPLBACKEND", "Agg")

    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write(f"$ {command_to_text(command)}\n\n")
        process = subprocess.Popen(
            list(command),
            cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )

        assert process.stdout is not None
        for line in process.stdout:
            print(f"[{stage}] {line}", end="")
            log_file.write(line)

        return_code = process.wait()

    elapsed = time.time() - started_at
    if return_code != 0:
        raise SystemExit(
            f"Stage {stage!r} failed with exit code {return_code}. See log: {log_path}"
        )
    print(f"Stage {stage} finished in {elapsed:.1f}s")


def train_hybrid_command(args: argparse.Namespace, profile: RunProfile, output_dir: Path) -> list[str]:
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "train_hybrid.py"),
        "--data-dir",
        str(args.data_dir),
        "--npz-path",
        str(args.visual_features),
        "--checkpoint-dir",
        str(args.checkpoint_dir),
        "--num-epochs",
        str(profile.hybrid_epochs),
        "--batch-size",
        str(profile.batch_size),
        "--num-negatives",
        str(profile.num_negatives),
        "--seed",
        str(args.seed),
        "--heartbeat-seconds",
        str(profile.heartbeat_seconds),
        "--log-every-batches",
        str(profile.log_every_batches),
        "--eval-batch-size",
        str(profile.batch_size),
    ]
    add_optional_int(command, "--max-train-rows", profile.max_train_rows)
    if profile.max_eval_users is not None:
        command.extend(["--max-eval-users", str(profile.max_eval_users)])
    if args.no_amp:
        command.append("--no-amp")
    return command


def train_compare_command(args: argparse.Namespace, profile: RunProfile, output_dir: Path) -> list[str]:
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "train_compare_recommenders.py"),
        "--data-dir",
        str(args.data_dir),
        "--output-dir",
        str(output_dir / "train_compare"),
        "--models",
        args.models,
        "--checkpoint-dir",
        str(args.checkpoint_dir),
        "--hybrid-checkpoint",
        str(args.checkpoint_dir / "hybrid_best.pt"),
        "--visual-features",
        str(args.visual_features),
        "--epochs",
        str(profile.compare_epochs),
        "--batch-size",
        str(profile.batch_size),
        "--num-negatives",
        str(profile.num_negatives),
        "--negative-candidates",
        str(profile.negative_candidates),
        "--seed",
        str(args.seed),
    ]
    add_optional_int(command, "--max-train-rows", profile.max_train_rows)
    add_eval_users(command, profile.max_eval_users)
    if args.no_amp:
        command.append("--no-amp")
    return command


def test_command(args: argparse.Namespace, profile: RunProfile, output_dir: Path) -> list[str]:
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "test_recommenders.py"),
        "--data-dir",
        str(args.data_dir),
        "--output-dir",
        str(output_dir / "test_results"),
        "--models",
        args.models,
        "--checkpoint-dir",
        str(args.checkpoint_dir),
        "--mf-checkpoint",
        str(args.checkpoint_dir / "mf_best.pt"),
        "--ncf-checkpoint",
        str(args.checkpoint_dir / "ncf_best.pt"),
        "--hybrid-checkpoint",
        str(args.checkpoint_dir / "hybrid_best.pt"),
        "--visual-features",
        str(args.visual_features),
        "--batch-size",
        str(profile.batch_size),
        "--negative-candidates",
        str(profile.negative_candidates),
        "--seed",
        str(args.seed),
    ]
    add_optional_int(command, "--max-train-rows", profile.max_train_rows)
    add_eval_users(command, profile.max_eval_users)
    return command


def write_run_summary(args: argparse.Namespace, profile: RunProfile, stages: Iterable[str], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "run_summary.txt"
    lines = [
        f"profile={args.profile}",
        f"stages={','.join(stages)}",
        f"data_dir={args.data_dir}",
        f"visual_features={args.visual_features}",
        f"checkpoint_dir={args.checkpoint_dir}",
        f"models={args.models}",
        f"seed={args.seed}",
        f"max_train_rows={profile.max_train_rows}",
        f"hybrid_epochs={profile.hybrid_epochs}",
        f"compare_epochs={profile.compare_epochs}",
        f"batch_size={profile.batch_size}",
        f"num_negatives={profile.num_negatives}",
        f"max_eval_users={profile.max_eval_users}",
        f"negative_candidates={profile.negative_candidates}",
        f"heartbeat_seconds={profile.heartbeat_seconds}",
        f"log_every_batches={profile.log_every_batches}",
    ]
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Run summary: {summary_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full report pipeline.")
    parser.add_argument("--profile", choices=sorted(PROFILES), default="report")
    parser.add_argument("--stages", type=parse_stages, default=parse_stages("all"))
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--visual-features", type=Path, default=DEFAULT_VISUAL_FEATURES)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--models", default="popularity,mf,ncf,hybrid")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-amp", action="store_true", help="Disable AMP for model training.")

    parser.add_argument("--max-train-rows", type=parse_optional_positive_int, default=None)
    parser.add_argument("--hybrid-epochs", type=int, default=None)
    parser.add_argument("--compare-epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-negatives", type=int, default=None)
    parser.add_argument("--max-eval-users", type=parse_optional_positive_int, default=None)
    parser.add_argument("--negative-candidates", type=int, default=None)
    parser.add_argument("--heartbeat-seconds", type=int, default=None)
    parser.add_argument("--log-every-batches", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    profile = build_profile(args)
    output_dir = args.output_root / args.profile
    log_dir = output_dir / "logs"

    write_run_summary(args, profile, args.stages, output_dir)

    if "check-data" in args.stages:
        validate_required_files(args.data_dir, args.visual_features)

    stage_commands = {
        "train-hybrid": train_hybrid_command,
        "train-compare": train_compare_command,
        "test": test_command,
    }

    for stage in args.stages:
        if stage == "check-data":
            continue
        command = stage_commands[stage](args, profile, output_dir)
        run_command(stage, command, log_dir / f"{stage}.log")

    print("\nReport pipeline finished.")
    print(f"Outputs: {output_dir}")
    print(f"Logs: {log_dir}")


if __name__ == "__main__":
    main()
