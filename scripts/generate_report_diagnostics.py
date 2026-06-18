"""
Generate report-ready diagnostics and plots for recommender experiments.

Run this after `scripts/run_report_pipeline.py` has produced checkpoints and
test metrics. The script writes figures, CSV tables, and a short Markdown
summary under the requested output directory.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import torch


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_compare_recommenders import (  # noqa: E402
    EvaluationData,
    build_evaluation_data,
    build_user_to_seen_items,
    compute_metrics,
    load_data,
    parse_model_list,
)
from src.config import (  # noqa: E402
    CHECKPOINT_DIR,
    DATA_DIR,
    MF_EMBEDDING_DIM,
    MLP_LAYER_SIZES,
    TOP_K_RECOMMENDATIONS,
    VISUAL_FEATURES_NPZ_PATH,
)
from src.features.metadata_feature_encoder import MetadataFeatureEncoder  # noqa: E402
from src.features.visual_feature_extract import VisualFeatureExtractor  # noqa: E402
from src.models.hybrid_model import HybridRecommendationModel  # noqa: E402
from src.models.matrix_factorization import MatrixFactorization  # noqa: E402
from src.models.ncf import NeuralCollaborativeFiltering  # noqa: E402


LOGGER = logging.getLogger(__name__)

MODEL_COLORS = {
    "Popularity": "#4C78A8",
    "MF": "#F58518",
    "NCF": "#54A24B",
    "Hybrid": "#B279A2",
}


def configure_logging() -> None:
    logging.basicConfig(
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
        level=logging.INFO,
    )


def configure_plots() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 130,
            "savefig.dpi": 180,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.frameon": False,
        }
    )


def optional_positive_int(value: str | None) -> int | None:
    if value is None:
        return None
    normalized = value.strip().lower()
    if normalized in {"none", "all", "0"}:
        return None
    parsed = int(normalized)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate report diagnostics and plots.")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--visual-features", type=Path, default=VISUAL_FEATURES_NPZ_PATH)
    parser.add_argument("--checkpoint-dir", type=Path, default=CHECKPOINT_DIR)
    parser.add_argument(
        "--pipeline-output-dir",
        type=Path,
        default=PROJECT_ROOT / "reports" / "report_pipeline" / "full",
        help="Directory produced by run_report_pipeline.py for the same run.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "reports" / "report_diagnostics",
    )
    parser.add_argument("--models", type=parse_model_list, default=parse_model_list("popularity,mf,ncf,hybrid"))
    parser.add_argument("--max-train-rows", type=optional_positive_int, default=None)
    parser.add_argument("--max-eval-users", type=optional_positive_int, default=None)
    parser.add_argument("--negative-candidates", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--mf-dim", type=int, default=MF_EMBEDDING_DIM)
    parser.add_argument("--mlp-layers", type=int, nargs="+", default=list(MLP_LAYER_SIZES))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-segment-eval", action="store_true")
    parser.add_argument("--overwrite-predictions", action="store_true")
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    ensure_dir(path.parent)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    LOGGER.info("Wrote CSV: %s", path)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def metric_label(metric: str) -> str:
    return {
        "map_at_12": "MAP@12",
        "hit_rate_at_12": "HitRate@12",
        "ndcg_at_12": "NDCG@12",
    }.get(metric, metric)


def find_metrics_csv(pipeline_output_dir: Path) -> Path | None:
    candidates = [
        pipeline_output_dir / "test_results" / "metrics_comparison.csv",
        pipeline_output_dir / "train_compare" / "metrics_comparison.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def find_training_history_csv(pipeline_output_dir: Path) -> Path | None:
    path = pipeline_output_dir / "train_compare" / "training_history.csv"
    return path if path.exists() else None


def plot_overall_metrics(metrics_csv: Path, output_dir: Path) -> None:
    rows = read_csv_rows(metrics_csv)
    if not rows:
        LOGGER.warning("No metrics rows found at %s", metrics_csv)
        return

    metrics = ["map_at_12", "hit_rate_at_12", "ndcg_at_12"]
    models = [row["model"] for row in rows]
    x = np.arange(len(metrics))
    width = 0.8 / max(len(models), 1)

    fig, axis = plt.subplots(figsize=(8.8, 4.8))
    for model_index, row in enumerate(rows):
        values = [float(row[metric]) for metric in metrics]
        offsets = x - 0.4 + width / 2 + model_index * width
        bars = axis.bar(
            offsets,
            values,
            width,
            label=row["model"],
            color=MODEL_COLORS.get(row["model"]),
        )
        axis.bar_label(bars, fmt="%.3f", padding=2, fontsize=8)

    axis.set_title("Overall Recommendation Metrics")
    axis.set_ylabel("Score")
    axis.set_xticks(x)
    axis.set_xticklabels([metric_label(metric) for metric in metrics])
    axis.set_ylim(0, max(float(row[metric]) for row in rows for metric in metrics) * 1.18)
    axis.legend(ncols=min(len(models), 4), loc="upper center", bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout()
    path = output_dir / "overall_metrics.png"
    fig.savefig(path)
    plt.close(fig)
    LOGGER.info("Wrote plot: %s", path)


def plot_training_history(history_csv: Path, output_dir: Path) -> None:
    rows = read_csv_rows(history_csv)
    if not rows:
        return

    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["model"]].append(row)

    fig, axis = plt.subplots(figsize=(8.8, 4.8))
    for model, model_rows in grouped.items():
        model_rows = sorted(model_rows, key=lambda row: float(row["epoch"]))
        epochs = [float(row["epoch"]) for row in model_rows]
        train_loss = [float(row["train_loss"]) for row in model_rows]
        val_loss = [float(row["val_loss"]) for row in model_rows]
        color = MODEL_COLORS.get(model)
        axis.plot(epochs, train_loss, marker="o", label=f"{model} train", color=color)
        axis.plot(epochs, val_loss, marker="s", linestyle="--", label=f"{model} val", color=color)

    axis.set_title("MF/NCF Training and Validation Loss")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("BCE loss")
    axis.legend(ncols=2, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout()
    path = output_dir / "training_history.png"
    fig.savefig(path)
    plt.close(fig)
    LOGGER.info("Wrote plot: %s", path)


def load_visual_keys(npz_path: Path) -> set[str]:
    LOGGER.info("Loading visual feature keys from %s ...", npz_path)
    with np.load(npz_path) as data:
        keys = set(data.files)
    LOGGER.info("Loaded %s visual vectors.", f"{len(keys):,}")
    return keys


def article_key_variants(article_id: int | str) -> tuple[str, str, str]:
    key = str(article_id)
    return key, key.zfill(9), key.zfill(10)


def build_article_visual_coverage(articles_df: pl.DataFrame, visual_keys: set[str]) -> dict[int, bool]:
    coverage: dict[int, bool] = {}
    for row in articles_df.select(["item_id", "article_id"]).iter_rows(named=True):
        item_id = int(row["item_id"])
        variants = article_key_variants(row["article_id"])
        coverage[item_id] = any(key in visual_keys for key in variants)
    return coverage


def item_count_map(df: pl.DataFrame) -> dict[int, int]:
    grouped = df.group_by("item_id").agg(pl.len().alias("count"))
    return {int(row["item_id"]): int(row["count"]) for row in grouped.iter_rows(named=True)}


def user_count_map(df: pl.DataFrame) -> dict[int, int]:
    grouped = df.group_by("user_id").agg(pl.len().alias("count"))
    return {int(row["user_id"]): int(row["count"]) for row in grouped.iter_rows(named=True)}


def item_rank_map(counts: dict[int, int]) -> dict[int, int]:
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return {item_id: index + 1 for index, (item_id, _) in enumerate(ranked)}


def item_popularity_segment(item_id: int, ranks: dict[int, int]) -> str:
    rank = ranks.get(int(item_id))
    if rank is None:
        return "Unseen in train"
    if rank <= 1000:
        return "Head <=1k"
    if rank <= 10000:
        return "Mid 1k-10k"
    return "Tail >10k"


def user_history_segment(count: int) -> str:
    if count <= 0:
        return "0"
    if count <= 3:
        return "1-3"
    if count <= 10:
        return "4-10"
    if count <= 30:
        return "11-30"
    return "31+"


def plot_visual_coverage(
    articles_df: pl.DataFrame,
    train_df: pl.DataFrame,
    test_df: pl.DataFrame,
    item_has_visual: dict[int, bool],
    output_dir: Path,
) -> dict[str, float]:
    article_item_ids = [int(item_id) for item_id in articles_df["item_id"].to_list()]
    covered_articles = sum(1 for item_id in article_item_ids if item_has_visual.get(item_id, False))
    total_articles = len(article_item_ids)

    train_items = [int(item_id) for item_id in train_df["item_id"].to_list()]
    test_items = [int(item_id) for item_id in test_df["item_id"].to_list()]
    covered_train = sum(1 for item_id in train_items if item_has_visual.get(item_id, False))
    covered_test = sum(1 for item_id in test_items if item_has_visual.get(item_id, False))

    summary = {
        "article_coverage": covered_articles / max(total_articles, 1),
        "train_interaction_coverage": covered_train / max(len(train_items), 1),
        "test_interaction_coverage": covered_test / max(len(test_items), 1),
    }

    labels = ["Articles", "Train interactions", "Test interactions"]
    values = [
        summary["article_coverage"],
        summary["train_interaction_coverage"],
        summary["test_interaction_coverage"],
    ]

    fig, axis = plt.subplots(figsize=(7.4, 4.4))
    bars = axis.bar(labels, values, color=["#4C78A8", "#72B7B2", "#F58518"])
    axis.bar_label(bars, labels=[f"{value:.1%}" for value in values], padding=3)
    axis.set_title("Visual Feature Coverage")
    axis.set_ylabel("Coverage")
    axis.set_ylim(0, 1.08)
    fig.tight_layout()
    path = output_dir / "visual_coverage_summary.png"
    fig.savefig(path)
    plt.close(fig)
    LOGGER.info("Wrote plot: %s", path)
    return summary


def plot_coverage_by_popularity(
    articles_df: pl.DataFrame,
    train_counts: dict[int, int],
    item_has_visual: dict[int, bool],
    output_dir: Path,
) -> list[dict[str, object]]:
    ranks = item_rank_map(train_counts)
    segment_order = ["Head <=1k", "Mid 1k-10k", "Tail >10k", "Unseen in train"]
    segment_totals = {segment: 0 for segment in segment_order}
    segment_covered = {segment: 0 for segment in segment_order}

    for item_id in articles_df["item_id"].to_list():
        item_id_int = int(item_id)
        segment = item_popularity_segment(item_id_int, ranks)
        segment_totals[segment] += 1
        if item_has_visual.get(item_id_int, False):
            segment_covered[segment] += 1

    rows = []
    for segment in segment_order:
        total = segment_totals[segment]
        covered = segment_covered[segment]
        rows.append(
            {
                "segment": segment,
                "articles": total,
                "covered_articles": covered,
                "coverage": covered / max(total, 1),
            }
        )

    fig, axis = plt.subplots(figsize=(8.2, 4.5))
    values = [float(row["coverage"]) for row in rows]
    bars = axis.bar(segment_order, values, color="#72B7B2")
    axis.bar_label(bars, labels=[f"{value:.1%}" for value in values], padding=3)
    axis.set_title("Visual Coverage by Item Popularity Segment")
    axis.set_ylabel("Coverage")
    axis.set_ylim(0, 1.08)
    fig.tight_layout()
    path = output_dir / "visual_coverage_by_item_popularity.png"
    fig.savefig(path)
    plt.close(fig)
    LOGGER.info("Wrote plot: %s", path)
    return rows


def plot_popularity_concentration(train_df: pl.DataFrame, test_df: pl.DataFrame, output_dir: Path) -> list[dict[str, object]]:
    top_ks = [10, 50, 100, 500, 1000]
    rows = []
    for split_name, df in (("Train", train_df), ("Test", test_df)):
        counts = df.group_by("item_id").agg(pl.len().alias("count")).sort("count", descending=True)
        total = len(df)
        for top_k in top_ks:
            share = float(counts.head(top_k)["count"].sum()) / max(total, 1)
            rows.append({"split": split_name, "top_k": top_k, "interaction_share": share})

    fig, axis = plt.subplots(figsize=(8.2, 4.6))
    for split_name, color in (("Train", "#4C78A8"), ("Test", "#F58518")):
        split_rows = [row for row in rows if row["split"] == split_name]
        axis.plot(
            [int(row["top_k"]) for row in split_rows],
            [float(row["interaction_share"]) for row in split_rows],
            marker="o",
            linewidth=2,
            label=split_name,
            color=color,
        )
    axis.set_title("Popularity Concentration in Train vs Test")
    axis.set_xlabel("Top-K items")
    axis.set_ylabel("Share of interactions")
    axis.set_xscale("log")
    axis.set_xticks(top_ks)
    axis.set_xticklabels([str(value) for value in top_ks])
    axis.yaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: f"{value:.0%}"))
    axis.legend()
    fig.tight_layout()
    path = output_dir / "popularity_concentration_train_test.png"
    fig.savefig(path)
    plt.close(fig)
    LOGGER.info("Wrote plot: %s", path)
    return rows


def plot_test_user_history_distribution(
    test_df: pl.DataFrame,
    train_user_counts: dict[int, int],
    output_dir: Path,
) -> list[dict[str, object]]:
    test_users = [int(user_id) for user_id in test_df["user_id"].unique().to_list()]
    segment_order = ["0", "1-3", "4-10", "11-30", "31+"]
    counts = {segment: 0 for segment in segment_order}
    for user_id in test_users:
        counts[user_history_segment(train_user_counts.get(user_id, 0))] += 1

    rows = [{"segment": segment, "users": counts[segment]} for segment in segment_order]
    fig, axis = plt.subplots(figsize=(7.6, 4.4))
    bars = axis.bar(segment_order, [counts[segment] for segment in segment_order], color="#4C78A8")
    axis.bar_label(bars, labels=[f"{counts[segment]:,}" for segment in segment_order], padding=3, fontsize=8)
    axis.set_title("Test Users by Train History Length")
    axis.set_xlabel("Train interactions per user")
    axis.set_ylabel("Users")
    fig.tight_layout()
    path = output_dir / "test_user_history_distribution.png"
    fig.savefig(path)
    plt.close(fig)
    LOGGER.info("Wrote plot: %s", path)
    return rows


def plot_test_item_popularity_distribution(
    test_df: pl.DataFrame,
    train_item_ranks: dict[int, int],
    output_dir: Path,
) -> list[dict[str, object]]:
    segment_order = ["Head <=1k", "Mid 1k-10k", "Tail >10k", "Unseen in train"]
    counts = {segment: 0 for segment in segment_order}
    for item_id in test_df["item_id"].to_list():
        counts[item_popularity_segment(int(item_id), train_item_ranks)] += 1

    rows = [{"segment": segment, "test_interactions": counts[segment]} for segment in segment_order]
    fig, axis = plt.subplots(figsize=(8.2, 4.5))
    bars = axis.bar(segment_order, [counts[segment] for segment in segment_order], color="#F58518")
    axis.bar_label(bars, labels=[f"{counts[segment]:,}" for segment in segment_order], padding=3, fontsize=8)
    axis.set_title("Test Interactions by Train Popularity Segment")
    axis.set_ylabel("Test interactions")
    fig.tight_layout()
    path = output_dir / "test_item_popularity_distribution.png"
    fig.savefig(path)
    plt.close(fig)
    LOGGER.info("Wrote plot: %s", path)
    return rows


def build_item_to_article_id_mapping(articles_df: pl.DataFrame) -> dict[int, int]:
    if "item_id" not in articles_df.columns or "article_id" not in articles_df.columns:
        return {}
    return {
        int(row["item_id"]): int(row["article_id"])
        for row in articles_df.select(["item_id", "article_id"]).iter_rows(named=True)
    }


def load_checkpoint(model: torch.nn.Module, checkpoint_path: Path, device: torch.device) -> bool:
    if not checkpoint_path.exists():
        LOGGER.warning("Checkpoint not found: %s", checkpoint_path)
        return False
    LOGGER.info("Loading checkpoint: %s", checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state_dict)
    LOGGER.info("Loaded checkpoint: %s", checkpoint_path)
    return True


def prediction_cache_path(output_dir: Path, model_name: str) -> Path:
    return output_dir / "prediction_cache" / f"{model_name.lower()}_predictions.csv"


def save_predictions(path: Path, predictions: dict[int, list[int]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["user_id", "rank", "item_id"])
        writer.writeheader()
        for user_id, items in predictions.items():
            for rank, item_id in enumerate(items, start=1):
                writer.writerow({"user_id": user_id, "rank": rank, "item_id": item_id})
    LOGGER.info("Wrote prediction cache: %s", path)


def load_predictions(path: Path) -> dict[int, list[int]]:
    predictions: dict[int, list[tuple[int, int]]] = defaultdict(list)
    with path.open("r", newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            predictions[int(row["user_id"])].append((int(row["rank"]), int(row["item_id"])))
    return {
        user_id: [item_id for _, item_id in sorted(items)]
        for user_id, items in predictions.items()
    }


def score_popularity_predictions(
    train_counts: dict[int, int],
    evaluation_data: EvaluationData,
) -> dict[int, list[int]]:
    predictions: dict[int, list[int]] = {}
    total_users = len(evaluation_data.users)
    for index, user_id in enumerate(evaluation_data.users, start=1):
        candidates = evaluation_data.candidates_by_user[user_id]
        ranked = sorted(candidates, key=lambda item_id: train_counts.get(int(item_id), 0), reverse=True)
        predictions[user_id] = [int(item_id) for item_id in ranked[:TOP_K_RECOMMENDATIONS]]
        if index == 1 or index % 1000 == 0 or index == total_users:
            LOGGER.info("Popularity predictions: %s/%s users", f"{index:,}", f"{total_users:,}")
    return predictions


def score_torch_cf_predictions(
    model_name: str,
    model: torch.nn.Module,
    evaluation_data: EvaluationData,
    device: torch.device,
    batch_size: int,
) -> dict[int, list[int]]:
    model.eval()
    predictions: dict[int, list[int]] = {}
    total_users = len(evaluation_data.users)
    with torch.no_grad():
        for index, user_id in enumerate(evaluation_data.users, start=1):
            candidates = evaluation_data.candidates_by_user[user_id]
            scores: list[float] = []
            for start in range(0, len(candidates), batch_size):
                batch_items = candidates[start : start + batch_size]
                users = torch.full((len(batch_items),), user_id, dtype=torch.long, device=device)
                items = torch.tensor(batch_items, dtype=torch.long, device=device)
                batch_scores = model(users, items).detach().cpu().numpy().tolist()
                scores.extend(float(score) for score in batch_scores)
            top_indices = np.argsort(scores)[::-1][:TOP_K_RECOMMENDATIONS]
            predictions[user_id] = [int(candidates[item_index]) for item_index in top_indices]
            if index == 1 or index % 500 == 0 or index == total_users:
                LOGGER.info("%s predictions: %s/%s users", model_name, f"{index:,}", f"{total_users:,}")
    return predictions


def score_hybrid_predictions(
    model: HybridRecommendationModel,
    evaluation_data: EvaluationData,
    feature_extractor: VisualFeatureExtractor,
    metadata_encoder: MetadataFeatureEncoder,
    device: torch.device,
    batch_size: int,
) -> dict[int, list[int]]:
    model.eval()
    predictions: dict[int, list[int]] = {}
    total_users = len(evaluation_data.users)
    with torch.no_grad():
        for index, user_id in enumerate(evaluation_data.users, start=1):
            candidates = evaluation_data.candidates_by_user[user_id]
            scores: list[float] = []
            for start in range(0, len(candidates), batch_size):
                batch_items = candidates[start : start + batch_size]
                users = torch.full((len(batch_items),), user_id, dtype=torch.long, device=device)
                items = torch.tensor(batch_items, dtype=torch.long, device=device)
                visual = feature_extractor.get_feature_vectors(batch_items).to(device)
                metadata = metadata_encoder.get_feature_vectors(batch_items).to(device)
                batch_scores = model(users, items, visual, metadata).detach().cpu().numpy().tolist()
                scores.extend(float(score) for score in batch_scores)
            top_indices = np.argsort(scores)[::-1][:TOP_K_RECOMMENDATIONS]
            predictions[user_id] = [int(candidates[item_index]) for item_index in top_indices]
            if index == 1 or index % 500 == 0 or index == total_users:
                LOGGER.info("Hybrid predictions: %s/%s users", f"{index:,}", f"{total_users:,}")
    return predictions


def get_or_create_predictions(
    model_name: str,
    output_dir: Path,
    overwrite: bool,
    create_fn,
) -> dict[int, list[int]]:
    cache_path = prediction_cache_path(output_dir, model_name)
    if cache_path.exists() and not overwrite:
        LOGGER.info("Loading cached predictions for %s from %s", model_name, cache_path)
        return load_predictions(cache_path)
    predictions = create_fn()
    save_predictions(cache_path, predictions)
    return predictions


def evaluate_models_for_segments(
    args: argparse.Namespace,
    train_df: pl.DataFrame,
    test_df: pl.DataFrame,
    articles_df: pl.DataFrame,
    train_counts: dict[int, int],
) -> tuple[EvaluationData, dict[str, dict[int, list[int]]], dict[str, float]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOGGER.info("Segment evaluation device: %s", device)

    user_to_seen_items = build_user_to_seen_items(train_df)
    all_item_id_pool = train_df["item_id"].unique().to_numpy()
    max_user_id = max(int(train_df["user_id"].max()), int(test_df["user_id"].max()))
    max_item_id = max(int(train_df["item_id"].max()), int(test_df["item_id"].max()))
    num_users = max_user_id + 1
    num_items = max_item_id + 1

    evaluation_data = build_evaluation_data(
        test_df=test_df,
        all_item_id_pool=all_item_id_pool,
        user_to_seen_items=user_to_seen_items,
        max_negative_candidates_per_user=args.negative_candidates,
        max_eval_users=args.max_eval_users,
        seed=args.seed,
    )

    predictions_by_model: dict[str, dict[int, list[int]]] = {}
    if "popularity" in args.models:
        predictions_by_model["Popularity"] = get_or_create_predictions(
            "Popularity",
            args.output_dir,
            args.overwrite_predictions,
            lambda: score_popularity_predictions(train_counts, evaluation_data),
        )

    if "mf" in args.models:
        mf_model = MatrixFactorization(num_users=num_users, num_items=num_items, embedding_dim=args.mf_dim).to(device)
        if load_checkpoint(mf_model, args.checkpoint_dir / "mf_best.pt", device):
            predictions_by_model["MF"] = get_or_create_predictions(
                "MF",
                args.output_dir,
                args.overwrite_predictions,
                lambda: score_torch_cf_predictions("MF", mf_model, evaluation_data, device, args.batch_size),
            )

    if "ncf" in args.models:
        ncf_model = NeuralCollaborativeFiltering(
            num_users=num_users,
            num_items=num_items,
            mf_embedding_dim=args.mf_dim,
            mlp_layer_sizes=args.mlp_layers,
        ).to(device)
        if load_checkpoint(ncf_model, args.checkpoint_dir / "ncf_best.pt", device):
            predictions_by_model["NCF"] = get_or_create_predictions(
                "NCF",
                args.output_dir,
                args.overwrite_predictions,
                lambda: score_torch_cf_predictions("NCF", ncf_model, evaluation_data, device, args.batch_size),
            )

    if "hybrid" in args.models:
        item_to_article_id = build_item_to_article_id_mapping(articles_df)
        feature_extractor = VisualFeatureExtractor(args.visual_features, item_id_to_article_id=item_to_article_id)
        metadata_encoder = MetadataFeatureEncoder(articles_df)
        hybrid_model = HybridRecommendationModel(
            num_users=num_users,
            num_items=num_items,
            visual_feature_dim=feature_extractor.feature_dim,
            metadata_feature_dim=metadata_encoder.feature_dim,
            mf_embedding_dim=args.mf_dim,
            mlp_layer_sizes=list(args.mlp_layers),
        ).to(device)
        if load_checkpoint(hybrid_model, args.checkpoint_dir / "hybrid_best.pt", device):
            predictions_by_model["Hybrid"] = get_or_create_predictions(
                "Hybrid",
                args.output_dir,
                args.overwrite_predictions,
                lambda: score_hybrid_predictions(
                    hybrid_model,
                    evaluation_data,
                    feature_extractor,
                    metadata_encoder,
                    device,
                    args.batch_size,
                ),
            )

    full_popularity_metrics = compute_full_catalog_popularity(
        train_counts=train_counts,
        evaluation_data=evaluation_data,
        user_to_seen_items=user_to_seen_items,
    )
    return evaluation_data, predictions_by_model, full_popularity_metrics


def compute_full_catalog_popularity(
    train_counts: dict[int, int],
    evaluation_data: EvaluationData,
    user_to_seen_items: dict[int, set[int]],
) -> dict[str, float]:
    ranked_all = [item_id for item_id, _ in sorted(train_counts.items(), key=lambda item: (-item[1], item[0]))]
    predictions: dict[int, list[int]] = {}
    for user_id in evaluation_data.actual_by_user:
        seen = user_to_seen_items.get(user_id, set())
        recommendations = []
        for item_id in ranked_all:
            if item_id in seen:
                continue
            recommendations.append(item_id)
            if len(recommendations) >= TOP_K_RECOMMENDATIONS:
                break
        predictions[user_id] = recommendations
    return compute_metrics(predictions, evaluation_data.actual_by_user)


def actuals_by_user_history_segment(
    evaluation_data: EvaluationData,
    train_user_counts: dict[int, int],
) -> list[tuple[str, dict[int, set[int]]]]:
    segment_order = ["0", "1-3", "4-10", "11-30", "31+"]
    grouped: dict[str, dict[int, set[int]]] = {segment: {} for segment in segment_order}
    for user_id, actual_items in evaluation_data.actual_by_user.items():
        segment = user_history_segment(train_user_counts.get(user_id, 0))
        grouped[segment][user_id] = actual_items
    return [(segment, grouped[segment]) for segment in segment_order]


def actuals_by_item_popularity_segment(
    evaluation_data: EvaluationData,
    train_item_ranks: dict[int, int],
) -> list[tuple[str, dict[int, set[int]]]]:
    segment_order = ["Head <=1k", "Mid 1k-10k", "Tail >10k", "Unseen in train"]
    grouped: dict[str, dict[int, set[int]]] = {segment: {} for segment in segment_order}
    for user_id, actual_items in evaluation_data.actual_by_user.items():
        for segment in segment_order:
            segment_actuals = {
                item_id
                for item_id in actual_items
                if item_popularity_segment(item_id, train_item_ranks) == segment
            }
            if segment_actuals:
                grouped[segment][user_id] = segment_actuals
    return [(segment, grouped[segment]) for segment in segment_order]


def actuals_by_visual_segment(
    evaluation_data: EvaluationData,
    item_has_visual: dict[int, bool],
) -> list[tuple[str, dict[int, set[int]]]]:
    segment_order = ["Visual covered", "Visual missing"]
    grouped: dict[str, dict[int, set[int]]] = {segment: {} for segment in segment_order}
    for user_id, actual_items in evaluation_data.actual_by_user.items():
        covered = {item_id for item_id in actual_items if item_has_visual.get(item_id, False)}
        missing = set(actual_items) - covered
        if covered:
            grouped["Visual covered"][user_id] = covered
        if missing:
            grouped["Visual missing"][user_id] = missing
    return [(segment, grouped[segment]) for segment in segment_order]


def build_segment_metric_rows(
    segment_type: str,
    segment_actuals: list[tuple[str, dict[int, set[int]]]],
    predictions_by_model: dict[str, dict[int, list[int]]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for segment, actual_by_user in segment_actuals:
        if not actual_by_user:
            continue
        actual_item_count = sum(len(items) for items in actual_by_user.values())
        for model_name, predictions in predictions_by_model.items():
            metrics = compute_metrics(predictions, actual_by_user)
            rows.append(
                {
                    "segment_type": segment_type,
                    "segment": segment,
                    "model": model_name,
                    "users": len(actual_by_user),
                    "actual_items": actual_item_count,
                    **metrics,
                }
            )
    return rows


def plot_segment_metric(
    rows: list[dict[str, object]],
    segment_type: str,
    output_path: Path,
    metric: str = "map_at_12",
) -> None:
    filtered = [row for row in rows if row["segment_type"] == segment_type]
    if not filtered:
        return
    segments = list(dict.fromkeys(str(row["segment"]) for row in filtered))
    models = list(dict.fromkeys(str(row["model"]) for row in filtered))
    values = {
        (str(row["segment"]), str(row["model"])): float(row[metric])
        for row in filtered
    }

    x = np.arange(len(segments))
    width = 0.8 / max(len(models), 1)
    fig, axis = plt.subplots(figsize=(max(8.2, len(segments) * 1.2), 4.8))
    for model_index, model_name in enumerate(models):
        offsets = x - 0.4 + width / 2 + model_index * width
        model_values = [values.get((segment, model_name), 0.0) for segment in segments]
        axis.bar(
            offsets,
            model_values,
            width,
            label=model_name,
            color=MODEL_COLORS.get(model_name),
        )

    axis.set_title(f"{metric_label(metric)} by {segment_type}")
    axis.set_ylabel(metric_label(metric))
    axis.set_xticks(x)
    axis.set_xticklabels(segments)
    axis.legend(ncols=min(len(models), 4), loc="upper center", bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    LOGGER.info("Wrote plot: %s", output_path)


def plot_popularity_sampled_vs_full(
    sampled_metrics: dict[str, float] | None,
    full_metrics: dict[str, float],
    output_dir: Path,
) -> None:
    if not sampled_metrics:
        return
    metrics = ["map_at_12", "hit_rate_at_12", "ndcg_at_12"]
    labels = ["Sampled candidates", "Full catalog filtered"]
    values = [
        [sampled_metrics[metric] for metric in metrics],
        [full_metrics[metric] for metric in metrics],
    ]
    x = np.arange(len(metrics))
    width = 0.35
    fig, axis = plt.subplots(figsize=(8.2, 4.6))
    for index, label in enumerate(labels):
        offsets = x + (index - 0.5) * width
        axis.bar(offsets, values[index], width, label=label, color=["#4C78A8", "#E45756"][index])
    axis.set_title("Popularity: Sampled vs Full-Catalog Evaluation")
    axis.set_ylabel("Score")
    axis.set_xticks(x)
    axis.set_xticklabels([metric_label(metric) for metric in metrics])
    axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncols=2)
    fig.tight_layout()
    path = output_dir / "popularity_sampled_vs_full_catalog.png"
    fig.savefig(path)
    plt.close(fig)
    LOGGER.info("Wrote plot: %s", path)


def find_model_metrics(rows: list[dict[str, str]], model_name: str) -> dict[str, float] | None:
    for row in rows:
        if row.get("model") == model_name:
            return {
                "map_at_12": float(row["map_at_12"]),
                "hit_rate_at_12": float(row["hit_rate_at_12"]),
                "ndcg_at_12": float(row["ndcg_at_12"]),
            }
    return None


def write_summary(
    output_dir: Path,
    args: argparse.Namespace,
    visual_summary: dict[str, float],
    full_popularity_metrics: dict[str, float] | None,
    generated_files: list[Path],
) -> None:
    lines = [
        "# Report Diagnostics Summary",
        "",
        f"- data_dir: `{args.data_dir}`",
        f"- visual_features: `{args.visual_features}`",
        f"- checkpoint_dir: `{args.checkpoint_dir}`",
        f"- max_eval_users: `{args.max_eval_users}`",
        f"- negative_candidates: `{args.negative_candidates}`",
        "",
        "## Visual Coverage",
        "",
        f"- Article coverage: {visual_summary['article_coverage']:.2%}",
        f"- Train interaction coverage: {visual_summary['train_interaction_coverage']:.2%}",
        f"- Test interaction coverage: {visual_summary['test_interaction_coverage']:.2%}",
    ]
    if full_popularity_metrics:
        lines.extend(
            [
                "",
                "## Popularity Full-Catalog Baseline",
                "",
                f"- MAP@12: {full_popularity_metrics['map_at_12']:.6f}",
                f"- HitRate@12: {full_popularity_metrics['hit_rate_at_12']:.6f}",
                f"- NDCG@12: {full_popularity_metrics['ndcg_at_12']:.6f}",
            ]
        )

    lines.extend(["", "## Generated Files", ""])
    for path in generated_files:
        lines.append(f"- `{path.relative_to(output_dir)}`")

    summary_path = output_dir / "diagnostics_summary.md"
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    LOGGER.info("Wrote summary: %s", summary_path)


def main() -> None:
    configure_logging()
    configure_plots()
    args = parse_args()
    args.output_dir = args.output_dir.resolve()
    ensure_dir(args.output_dir)
    os.environ.setdefault("MPLBACKEND", "Agg")

    LOGGER.info("Loading data from %s", args.data_dir)
    train_df, test_df, articles_df = load_data(args.data_dir, args.max_train_rows)
    train_counts = item_count_map(train_df)
    test_item_ranks = item_rank_map(train_counts)
    train_user_counts = user_count_map(train_df)

    generated_files_before = set(path for path in args.output_dir.rglob("*") if path.is_file())

    metrics_csv = find_metrics_csv(args.pipeline_output_dir)
    metrics_rows: list[dict[str, str]] = []
    if metrics_csv:
        LOGGER.info("Using metrics CSV: %s", metrics_csv)
        metrics_rows = read_csv_rows(metrics_csv)
        plot_overall_metrics(metrics_csv, args.output_dir)
    else:
        LOGGER.warning("No pipeline metrics CSV found under %s", args.pipeline_output_dir)

    history_csv = find_training_history_csv(args.pipeline_output_dir)
    if history_csv:
        plot_training_history(history_csv, args.output_dir)

    visual_keys = load_visual_keys(args.visual_features)
    item_has_visual = build_article_visual_coverage(articles_df, visual_keys)
    visual_summary = plot_visual_coverage(articles_df, train_df, test_df, item_has_visual, args.output_dir)
    coverage_rows = plot_coverage_by_popularity(articles_df, train_counts, item_has_visual, args.output_dir)
    concentration_rows = plot_popularity_concentration(train_df, test_df, args.output_dir)
    user_history_rows = plot_test_user_history_distribution(test_df, train_user_counts, args.output_dir)
    test_popularity_rows = plot_test_item_popularity_distribution(test_df, test_item_ranks, args.output_dir)

    write_csv(
        args.output_dir / "visual_coverage_by_item_popularity.csv",
        coverage_rows,
        ["segment", "articles", "covered_articles", "coverage"],
    )
    write_csv(
        args.output_dir / "popularity_concentration_train_test.csv",
        concentration_rows,
        ["split", "top_k", "interaction_share"],
    )
    write_csv(
        args.output_dir / "test_user_history_distribution.csv",
        user_history_rows,
        ["segment", "users"],
    )
    write_csv(
        args.output_dir / "test_item_popularity_distribution.csv",
        test_popularity_rows,
        ["segment", "test_interactions"],
    )

    full_popularity_metrics = None
    if not args.skip_segment_eval:
        evaluation_data, predictions_by_model, full_popularity_metrics = evaluate_models_for_segments(
            args=args,
            train_df=train_df,
            test_df=test_df,
            articles_df=articles_df,
            train_counts=train_counts,
        )

        segment_rows: list[dict[str, object]] = []
        segment_rows.extend(
            build_segment_metric_rows(
                "User history",
                actuals_by_user_history_segment(evaluation_data, train_user_counts),
                predictions_by_model,
            )
        )
        segment_rows.extend(
            build_segment_metric_rows(
                "Item popularity",
                actuals_by_item_popularity_segment(evaluation_data, test_item_ranks),
                predictions_by_model,
            )
        )
        segment_rows.extend(
            build_segment_metric_rows(
                "Visual availability",
                actuals_by_visual_segment(evaluation_data, item_has_visual),
                predictions_by_model,
            )
        )

        write_csv(
            args.output_dir / "segment_metrics.csv",
            segment_rows,
            [
                "segment_type",
                "segment",
                "model",
                "users",
                "actual_items",
                "map_at_12",
                "hit_rate_at_12",
                "ndcg_at_12",
            ],
        )
        plot_segment_metric(segment_rows, "User history", args.output_dir / "segment_map_user_history.png")
        plot_segment_metric(segment_rows, "Item popularity", args.output_dir / "segment_map_item_popularity.png")
        plot_segment_metric(segment_rows, "Visual availability", args.output_dir / "segment_map_visual_availability.png")

        sampled_popularity = find_model_metrics(metrics_rows, "Popularity")
        plot_popularity_sampled_vs_full(sampled_popularity, full_popularity_metrics, args.output_dir)

    generated_files_after = set(path for path in args.output_dir.rglob("*") if path.is_file())
    generated_files = sorted(generated_files_after - generated_files_before)
    write_summary(args.output_dir, args, visual_summary, full_popularity_metrics, generated_files)
    LOGGER.info("Diagnostics finished. Output: %s", args.output_dir)


if __name__ == "__main__":
    main()
