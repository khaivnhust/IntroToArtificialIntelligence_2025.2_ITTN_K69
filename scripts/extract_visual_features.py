"""
extract_visual_features.py - Extract ResNet-50 image embeddings for H&M articles.

The output NPZ uses raw ``article_id`` strings as keys. Training and inference
map encoded ``item_id`` values back to these raw article IDs before lookup.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polars as pl
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


class ArticleImageDataset(Dataset):
    def __init__(self, articles_df: pl.DataFrame, image_dir: Path) -> None:
        self.article_ids = articles_df["article_id"].to_list()
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
        article_id = str(self.article_ids[index])
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
        return [], torch.empty(0), []
    valid_article_ids = [article_ids[index] for index in valid_indices]
    valid_images = torch.stack([images[index] for index in valid_indices])
    return valid_article_ids, valid_images, valid_indices


def build_resnet50(device: torch.device) -> torch.nn.Module:
    weights = models.ResNet50_Weights.DEFAULT
    model = models.resnet50(weights=weights)
    model.fc = torch.nn.Identity()
    model.eval()
    return model.to(device)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract ResNet-50 visual features.")
    parser.add_argument("--articles", type=Path, default=Path("data/processed/articles_cleaned.parquet"))
    parser.add_argument("--image-dir", type=Path, default=Path("data/images"))
    parser.add_argument("--out", type=Path, default=Path("data/processed/visual_features_sample.npz"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    articles_df = pl.read_parquet(args.articles)
    if args.limit:
        articles_df = articles_df.head(args.limit)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_resnet50(device)
    dataset = ArticleImageDataset(articles_df, args.image_dir)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch)

    features: dict[str, np.ndarray] = {}
    with torch.no_grad():
        for article_ids, images, _ in loader:
            if not article_ids:
                continue
            embeddings = model(images.to(device)).cpu().numpy().astype(np.float32)
            for article_id, embedding in zip(article_ids, embeddings):
                features[str(article_id)] = embedding

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **features)
    print(f"Wrote {len(features)} visual feature vectors to {args.out}")


if __name__ == "__main__":
    main()
