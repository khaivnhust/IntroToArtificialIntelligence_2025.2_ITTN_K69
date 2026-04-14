# Hybrid Fashion Recommender System

Hybrid recommendation system for H&M fashion products, combining
Neural Collaborative Filtering (NCF) with visual features extracted from
product images via a pretrained ResNet-50.

## Project Overview

The system recommends personalised Top-12 products for each user by fusing:

- **Collaborative Filtering** — Matrix Factorization and NCF learn user/item
  latent representations from purchase history.
- **Visual Features** — 2048-dim image embeddings from ResNet-50 capture
  visual similarity between products.
- **Hybrid Fusion** — NCF latent vectors and visual embeddings are
  concatenated and passed through dense layers for the final prediction.

### Problems addressed

- Difficult product discovery in a large catalogue
- Cold start for new users / items
- Underutilised visual information in traditional CF systems

## Tech Stack

| Category           | Tools                                  |
|--------------------|----------------------------------------|
| Language           | Python                                 |
| Data Processing    | Polars, Pandas, NumPy                  |
| Machine Learning   | Scikit-learn                           |
| Deep Learning      | PyTorch                                |
| Computer Vision    | ResNet-50 (pretrained, torchvision)    |
| Demo               | Streamlit                              |
| Storage            | Parquet, NPZ                           |

## Project Structure

```
fashion-recommender/
├── data/
│   ├── README.md                  # How to download the H&M dataset
│   └── processed/                 # Cleaned & encoded Parquet + NPZ files
│
├── src/
│   ├── __init__.py
│   ├── config.py                  # Centralised constants & hyper-parameters
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   └── data_loader.py        # Polars-based Parquet data loading
│   ├── features/
│   │   ├── __init__.py
│   │   └── visual_feature_extractor.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── popularity_baseline.py
│   │   ├── matrix_factorization.py
│   │   ├── ncf.py                 # Neural Collaborative Filtering
│   │   ├── hybrid_model.py        # NCF + visual fusion model
│   │   └── inference_pipeline.py  # High-level inference API
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── metrics.py             # MAP@12, Hit Rate, NDCG
│   └── utils/
│       ├── __init__.py
│       └── early_stopping.py
│
├── scripts/
│   └── train_hybrid.py            # Training pipeline (CLI)
│
├── app/
│   └── app.py                     # Streamlit demo
│
├── checkpoints/                   # Training checkpoints (per-epoch)
├── models/                        # Final saved model weights
├── docs/                          # Reports, proposal, slides
│
├── requirements.txt
└── README.md
```

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare the data

See [`data/README.md`](data/README.md) for instructions on downloading the
H&M dataset from Kaggle and running the preprocessing pipeline.

### 3. Train the model

```bash
python scripts/train_hybrid.py
```

Common options:

```bash
python scripts/train_hybrid.py --batch-size 2048 --num-epochs 50 --lr 5e-4
python scripts/train_hybrid.py --no-amp   # disable mixed precision
```

### 4. Run the demo

```bash
streamlit run app/app.py
```

Select a customer ID, choose a recommendation method (Hybrid or Popularity
Baseline), and view the personalised Top-12 products.

## Core Features

- **Data Preprocessing** — clean, encode user/item IDs, filter by time window
- **Popularity Baseline** — global best-sellers as a simple reference
- **Matrix Factorization** — latent-factor MF with bias terms
- **Neural Collaborative Filtering** — GMF + MLP dual-path architecture
- **Visual Feature Extraction** — ResNet-50 embeddings from product images
- **Hybrid Recommendation** — CF + visual features fused via dense layers
- **Evaluation** — MAP@12, Hit Rate@K, NDCG@K
- **Streamlit Demo** — interactive Top-12 display with product metadata

## Evaluation Metric

The primary metric is **MAP@12** (Mean Average Precision at 12), consistent
with the
[Kaggle H&M competition](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations).

## Team

| Name                | Student ID  |
|---------------------|-------------|
| Nguyen The Khai     | 202400050   |
| Pham Gia Linh       | 202416262   |
| Nguyen Dang Long    | 202400057   |

**Programme:** CTTN Computer Science — K69
**Course:** Introduction to Artificial Intelligence
