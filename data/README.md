# Data Directory

This folder contains the processed datasets used by the recommendation system.

## Structure

```
data/
└── processed/
    ├── articles_cleaned.parquet     — Product metadata (name, type, colour, …)
    ├── customers_fixed.parquet      — Customer demographics
    ├── hm_train.parquet             — Training transactions (user_id, item_id, …)
    ├── hm_test.parquet              — Test transactions
    └── visual_features_sample.npz   — Pre-computed ResNet-50 image embeddings
```

## How to obtain the data

The raw dataset comes from the **H&M Personalized Fashion Recommendations**
competition on Kaggle:

> <https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/data>

### Steps

1. **Create a Kaggle account** at <https://www.kaggle.com/> (if you don't have one).

2. **Accept the competition rules** on the competition page.

3. **Download via Kaggle CLI** (recommended):

   ```bash
   pip install kaggle
   kaggle competitions download -c h-and-m-personalized-fashion-recommendations
   ```

   Or download manually from the Data tab on the competition page.

4. **Run the preprocessing pipeline** to produce the Parquet files listed above.
   The preprocessing scripts encode `customer_id` and `article_id` into
   contiguous integer indices, filter transactions by time window, and save
   the results as Parquet for efficient I/O.

5. **Place the resulting files** in `data/processed/`.

> **Note:** The Parquet and NPZ files are too large to push to GitHub.
> Add `data/processed/*.parquet` and `data/processed/*.npz` to your `.gitignore`.
