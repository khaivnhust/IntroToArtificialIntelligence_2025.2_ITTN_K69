#  Hybrid Fashion Recommender System– Hệ thống gợi ý thời trang đa phương thức


## Project Overview
Hệ thống gợi ý sản phẩm thời trang dựa trên dữ liệu hành vi người dùng và nội dung sản phẩm (metadata + hình ảnh).

Giải quyết các vấn đề:

Khó tìm sản phẩm phù hợp trong kho lớn

Cold start (user/item mới)

Thiếu khai thác thông tin hình ảnh trong recommendation

Hệ thống sử dụng Hybrid Recommendation kết hợp Collaborative Filtering và Multimodal Learning.

## Tech Stack
Language: Python

Data Processing: Polars / Pandas / NumPy

Machine Learning: Scikit-learn

Deep Learning: PyTorch / TensorFlow

Computer Vision: ResNet-50 (pretrained)

Deployment / Demo: Streamlit

Storage: Parquet
## Project Structure
```
fashion-recommender/
├── data/                          # Raw & processed datasets
│   ├── raw/                       # Original H&M dataset
│   ├── processed/                 # Cleaned & encoded data (Parquet)
│   └── images/                    # Product images
│
├── src/
│   ├── preprocessing/             # Data cleaning & encoding
│   ├── features/                  # Feature engineering (metadata + image)
│   ├── models/
│   │   ├── baseline/              # Popularity model
│   │   ├── mf/                    # Matrix Factorization
│   │   ├── ncf/                   # Neural Collaborative Filtering
│   │   └── hybrid/                # Hybrid model (CF + content + image)
│   ├── evaluation/                # MAP@K, metrics
│   └── utils/                     # Helper functions
│
├── notebooks/                     # EDA & experiments
├── app/                           # Streamlit demo
│   └── app.py
│
├── models/                        # Saved model weights
├── docs/                          # Reports, diagrams
├── requirements.txt
└── README.md
```
## Core Features Implementation
Data Preprocessing: Làm sạch, encode user/item, lọc theo thời gian

 Baseline Model: Popularity-based recommendation

 Collaborative Filtering:

Matrix Factorization (MF)

Neural Collaborative Filtering (NCF)

 Visual Feature Extraction:

Trích xuất embedding từ ảnh bằng ResNet-50

 Hybrid Recommendation:

Kết hợp CF + metadata + image features

 Evaluation:

Metric: MAP@12

 Demo System:

Streamlit app hiển thị Top-12 sản phẩm
## Methodology

Input Data:

Transactions (user behavior)

Articles (metadata)

Images (visual features)

Models:

Baseline → MF → NCF → Hybrid

Fusion Strategy:

Concatenation / Weighted sum

MLP for final prediction

## Development Timeline

Week 1–2: EDA & preprocessing

Week 3–4: Baseline + CF (MF, NCF)

Week 5–6: Image features + Hybrid model

Week 7–8: Demo + report



