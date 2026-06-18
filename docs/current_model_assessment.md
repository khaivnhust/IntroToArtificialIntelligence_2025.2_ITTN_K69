# Current Recommender Model Assessment

Last reviewed: 2026-06-17

This document summarizes the current state of the H&M fashion recommendation
project based on the latest artifacts under `reports/report_pipeline`, the
processed dataset, and the current training/evaluation scripts.

## 1. Executive Summary

The current full-scale evaluation result is:

```text
Popularity > MF > Hybrid > NCF
```

Final full test metrics:

| Model | MAP@12 | HitRate@12 | NDCG@12 |
|---|---:|---:|---:|
| Popularity | 0.066396 | 0.297903 | 0.104827 |
| MF | 0.060607 | 0.273208 | 0.095530 |
| Hybrid | 0.053500 | 0.262410 | 0.086772 |
| NCF | 0.041196 | 0.230972 | 0.069943 |

Main interpretation:

- Popularity is currently the strongest model on the full test setup.
- MF is the best trainable collaborative filtering model.
- Hybrid beats NCF, but does not beat MF or Popularity.
- NCF has the lowest ranking performance despite low training loss.
- The main suspected weaknesses are sparse visual feature coverage, strong
  popularity/trend structure in the test window, objective mismatch between BCE
  training and ranking metrics, and overfitting after the first epoch.

This result should be presented honestly: the system includes a Hybrid model,
but the current evidence does not show that Hybrid is the best recommender.

## 2. Problem Context

The project is a fashion recommender system using H&M transaction data. The
goal is to recommend top-12 articles for each customer, matching the common
H&M recommendation challenge setup where final quality is measured by ranking
the relevant purchased items near the top.

The recommendation task is implicit feedback:

- A purchase is treated as a positive interaction.
- Negative examples are sampled from items the user has not purchased.
- Models learn to score user-item pairs.
- During evaluation, each user's true test items are ranked against sampled
  negative candidate items.

The practical target is not only to minimize binary classification loss. The
important product outcome is ranking quality: relevant items must appear in the
top-12 recommendation list.

## 3. Data Overview

Processed files used by the pipeline:

| File | Role |
|---|---|
| `data/processed/hm_train.parquet` | Training interactions |
| `data/processed/hm_test.parquet` | Held-out test interactions |
| `data/processed/articles_cleaned.parquet` | Article metadata |
| `data/processed/customers_cleaned.parquet` | Customer metadata |
| `data/processed/visual_features_sample.npz` | Precomputed visual vectors |

### 3.1 Interaction Schema

`hm_train.parquet` and `hm_test.parquet` contain:

| Column | Type | Meaning |
|---|---|---|
| `t_dat` | datetime | Transaction date |
| `user_id` | int | Encoded customer id |
| `item_id` | int | Encoded article id |
| `price` | float | Transaction price |
| `sales_channel_id` | int | Sales channel |

Current models mainly use `user_id` and `item_id`. Hybrid additionally uses
article metadata and visual features through the article/item mapping.

### 3.2 Dataset Size

| Split / Table | Rows |
|---|---:|
| Train interactions | 19,196,530 |
| Test interactions | 266,364 |
| Articles | 80,527 |
| Customers | 1,088,773 |

### 3.3 Train/Test Time Window

| Split | Date range |
|---|---|
| Train | 2019-06-22 to 2020-09-14 |
| Test | 2020-09-15 to 2020-09-22 |

This is a temporal split. The test window is only one week after the train
window, so short-term popularity and fashion trends can be very strong.

### 3.4 User and Item Coverage

| Statistic | Train | Test |
|---|---:|---:|
| Unique users | 1,081,265 | 75,481 |
| Unique items | 79,759 | 18,684 |
| Mean interactions per user | 17.75 | 3.53 |
| Median interactions per user | 9 | 2 |
| P90 interactions per user | 43 | 7 |
| Max interactions per user | 1,270 | 104 |
| Mean interactions per item | 240.68 | 14.26 |
| Median interactions per item | 37 | 3 |
| P90 interactions per item | 655 | 33 |
| Max interactions per item | 35,669 | 970 |

Overlap:

| Statistic | Value |
|---|---:|
| Test users also seen in train | 67,973 |
| Share of test users seen in train | 90.05% |
| Test users not seen in train | 7,508 |
| Test items also seen in train | 17,916 |
| Share of test items seen in train | 95.89% |
| Test items not seen in train | 768 |

Implication:

- There is a non-trivial cold-start user segment: about 9.95% of test users are
  unseen during training.
- There is also a small item cold-start segment: about 4.11% of test items are
  unseen in train.
- Pure CF methods can struggle on these groups unless there is a fallback or
  content-aware path that is well covered.

### 3.5 Popularity Concentration

The test week is more concentrated around popular items than the full train
period:

| Top-K items | Train interaction share | Test interaction share |
|---:|---:|---:|
| Top 10 | 1.04% | 2.44% |
| Top 50 | 3.23% | 8.28% |
| Top 100 | 5.25% | 13.32% |
| Top 500 | 14.36% | 35.13% |
| Top 1000 | 21.61% | 49.70% |

This strongly explains why the Popularity baseline is competitive and currently
best. Nearly half of the test interactions are covered by the top 1000 test
items. In fashion, this is plausible because the test week is heavily driven by
seasonal trends, new campaigns, and globally popular products.

### 3.6 Visual Feature Coverage

`visual_features_sample.npz` contains only 996 precomputed visual vectors:

| Coverage target | Coverage |
|---|---:|
| Articles with visual vectors | 996 / 80,527 = 1.24% |
| Train unique item coverage | 1.25% |
| Test unique item coverage | 0.92% |
| Train interaction coverage | 5.63% |
| Test interaction coverage | 1.97% |

This is a major limitation for Hybrid. The Hybrid model architecture expects a
2048-dimensional visual feature vector, but most items receive a missing/zero
visual vector at inference. Therefore, the current Hybrid model is not really
evaluating the full benefit of visual recommendation. It is mostly evaluating
NCF plus metadata plus sparse visual signals.

## 4. Models Compared

The current pipeline compares four recommenders.

### 4.1 Popularity Baseline

The Popularity model ranks items by global purchase frequency in the training
set. It does not personalize by user.

Strengths:

- Very robust.
- No gradient training needed.
- Strong when the test period follows recent global trends.
- Handles sparse user histories better than user-specific CF methods.

Weaknesses:

- Low personalization.
- Cannot model individual taste.
- Can over-recommend globally popular items.

Current result:

- Best overall on full test.
- This indicates the test window is highly popularity/trend driven.

### 4.2 Matrix Factorization (MF)

MF learns user and item latent embeddings plus user/item biases. The score is
based on the dot product of user and item embeddings plus biases.

Current configuration:

- Embedding dimension: 16.
- Loss: `BCEWithLogitsLoss`.
- Optimizer: `AdamW`.
- Negative sampling: 4 negatives per positive in full profile.
- Batch size: 4096 in full profile.
- Epochs: 5 in full profile.
- AMP enabled on CUDA.
- Lazy negative sampling and `RandomSampler(replacement=True)` are now used in
  the comparison training path.

Current result:

- Best trainable model.
- Second overall after Popularity.

### 4.3 Neural Collaborative Filtering (NCF)

NCF uses a NeuMF-style architecture:

- GMF branch: element-wise product of user and item embeddings.
- MLP branch: concatenated user/item embeddings passed through dense layers.
- Final prediction layer outputs a logit.

Current configuration:

- MF embedding dimension: 16.
- MLP layer sizes: `[128, 64, 32, 16]`.
- Loss: `BCEWithLogitsLoss`.
- Optimizer: `AdamW`.
- Negative sampling: 4 negatives per positive in full profile.
- Batch size: 4096 in full profile.
- Epochs: 5 in full profile.
- AMP enabled on CUDA.
- Lazy negative sampling and `RandomSampler(replacement=True)` are now used.

Current result:

- Lowest ranking performance in the full test run.
- Training loss decreases, but ranking metrics do not follow.

### 4.4 Hybrid Model

Hybrid combines collaborative and content signals:

- NCF latent representation.
- 2048-dimensional visual feature vector.
- Metadata feature vector.
- Dense fusion layers: 256 -> 64 -> 1 with BatchNorm, ReLU, and Dropout.

Current configuration:

- Loss: `BCEWithLogitsLoss`.
- Optimizer: Adam-style training in `train_hybrid.py`.
- Negative sampling: 4 negatives per positive in full profile.
- Batch size: 4096 in full profile.
- Epochs: 5 in full profile.
- AMP enabled on CUDA.
- Visual features loaded from `visual_features_sample.npz`.

Current result:

- Better than NCF.
- Worse than MF and Popularity.
- Best validation loss was at epoch 1, then validation loss worsened.
- Visual coverage is too low to fairly demonstrate visual recommendation value.

## 5. Evaluation Methodology

The system uses ranking metrics at top-12.

### 5.1 Candidate Construction

For each test user:

1. True test items are used as positive candidates.
2. Up to 1000 negative candidates are sampled.
3. Previously seen training items and true test items are excluded from the
   negative pool.
4. Each model scores the candidate items.
5. The top-12 items are selected.

Full profile settings:

| Setting | Value |
|---|---:|
| `max_train_rows` | all train rows |
| `max_eval_users` | all test users |
| Evaluation users | 75,481 |
| Negative candidates per user | 1000 |
| Batch size | 4096 |
| Number of negatives during training | 4 |
| Compare epochs for MF/NCF | 5 |
| Hybrid epochs | 5 |

### 5.2 Metrics

| Metric | Meaning |
|---|---|
| MAP@12 | Measures ranking quality with stronger reward for placing relevant items near the top. Primary metric. |
| HitRate@12 | Measures whether at least one relevant item appears in top-12. |
| NDCG@12 | Measures ranking quality with logarithmic discount by position. |

Important distinction:

- `BCEWithLogitsLoss` is a binary classification loss over sampled
  positive/negative pairs.
- MAP@12/NDCG@12 are ranking metrics.
- A model can have lower BCE loss but worse top-12 ranking.

## 6. Pipeline and Artifact Status

Primary directory:

```text
reports/report_pipeline/full/
```

Important files:

| File | Meaning |
|---|---|
| `run_summary.txt` | Full profile configuration |
| `train_compare/metrics_comparison.csv` | Metrics immediately after MF/NCF train and Hybrid checkpoint evaluation |
| `test_results/metrics_comparison.csv` | Final checkpoint reload test metrics |
| `train_compare/training_history.csv` | MF/NCF train and validation losses |
| `logs/train-compare.log` | Full train-compare logs |
| `logs/test.log` | Full test logs |
| `logs/train-hybrid.log` | Existing Hybrid training log from the earlier full Hybrid run |

Note: the latest `full/run_summary.txt` shows:

```text
stages=train-compare,test
```

That means the latest full run trained MF/NCF and then tested all checkpoints.
Hybrid was not retrained in that latest run. It was loaded from:

```text
checkpoints/hybrid_best.pt
```

The existing `full/logs/train-hybrid.log` is from the prior full Hybrid
training run. It is still relevant for understanding the current Hybrid
checkpoint, but the `full` directory now contains artifacts from more than one
execution. This should be noted in any final report.

## 7. Full Run Results

### 7.1 Train-Compare Metrics

`reports/report_pipeline/full/train_compare/metrics_comparison.csv`

| Model | MAP@12 | HitRate@12 | NDCG@12 |
|---|---:|---:|---:|
| MF | 0.060607 | 0.273208 | 0.095530 |
| NCF | 0.041201 | 0.230879 | 0.069935 |
| Popularity | 0.066396 | 0.297903 | 0.104827 |
| Hybrid | 0.053500 | 0.262410 | 0.086772 |

### 7.2 Final Test Metrics

`reports/report_pipeline/full/test_results/metrics_comparison.csv`

| Model | MAP@12 | HitRate@12 | NDCG@12 |
|---|---:|---:|---:|
| Popularity | 0.066396 | 0.297903 | 0.104827 |
| MF | 0.060607 | 0.273208 | 0.095530 |
| NCF | 0.041196 | 0.230972 | 0.069943 |
| Hybrid | 0.053500 | 0.262410 | 0.086772 |

The final test result is nearly identical to train-compare metrics, confirming
that checkpoint loading is consistent.

## 8. Training Dynamics

### 8.1 MF Training History

| Epoch | Train loss | Val loss | Seconds |
|---:|---:|---:|---:|
| 1 | 0.354657 | 1.556301 | 510.18 |
| 2 | 0.287601 | 1.946291 | 505.93 |
| 3 | 0.263567 | 2.166673 | 505.38 |
| 4 | 0.243568 | 2.325684 | 505.14 |
| 5 | 0.229561 | 2.455489 | 507.99 |

Observation:

- Train loss improves every epoch.
- Validation loss worsens every epoch after epoch 1.
- Best checkpoint is effectively epoch 1.
- Later epochs are likely overfitting under the current sampling/objective.

### 8.2 NCF Training History

| Epoch | Train loss | Val loss | Seconds |
|---:|---:|---:|---:|
| 1 | 0.282345 | 1.453556 | 1354.38 |
| 2 | 0.235230 | 1.556289 | 1338.51 |
| 3 | 0.206072 | 1.591837 | 1342.04 |
| 4 | 0.190049 | 1.756209 | 1422.21 |
| 5 | 0.176864 | 1.644357 | 1457.53 |

Observation:

- Train loss improves strongly.
- Validation loss is best at epoch 1.
- Ranking result is weakest among all models.
- The architecture may be overfitting or the objective is not aligned with
  ranking.

### 8.3 Hybrid Training History

From `reports/report_pipeline/full/logs/train-hybrid.log`:

| Epoch | Train loss | Val loss | Runtime |
|---:|---:|---:|---:|
| 1 | 0.29546 | 1.29456 | 42m09s |
| 2 | 0.25054 | 1.36118 | 1h03m48s |
| 3 | 0.21874 | 1.42024 | 43m30s |
| 4 | 0.19880 | 1.50545 | 47m41s |
| 5 | 0.18280 | 1.58344 | 46m40s |

Observation:

- Best validation loss is at epoch 1.
- Validation loss worsens every later epoch.
- The checkpoint used for test is the epoch-1 best checkpoint.
- Running more epochs with the same setup is unlikely to solve the current
  ranking gap. It would probably increase overfitting unless regularization,
  data coverage, or objective changes are made.

## 9. Runtime Observations

Full `train-compare`:

- Uses CUDA with Torch `2.7.1+cu118`.
- `MF/NCF Mixed Precision: True`.
- MF/NCF use `sampler=replacement`.
- Training samples per epoch: 91,183,515.
- Batches per epoch: 22,262.
- Stage runtime reported: about 14,457.7 seconds, approximately 4 hours.

Approximate model training time:

| Model | Time per epoch | Total train time |
|---|---:|---:|
| MF | about 505-510s | about 42 minutes |
| NCF | about 1338-1458s | about 1h55m |
| Hybrid | about 42-64 minutes per epoch | about 4h plus final evaluation |

Full `test`:

- Evaluates 75,481 users.
- Loads `mf_best.pt`, `ncf_best.pt`, and `hybrid_best.pt`.
- Runtime from log timestamps is about 1h28m.
- A major runtime cost is building evaluation candidates for all users.

Potential engineering improvement:

- Cache evaluation candidates to disk for a given seed/config.
- Reuse the same candidate set between `train-compare` and `test`.
- This would avoid rebuilding candidates and make results easier to audit.

## 10. Interpretation of Current Results

### 10.1 Why Popularity Is Best

Popularity wins because the test period is strongly trend-driven:

- Test top-500 items account for 35.13% of all test interactions.
- Test top-1000 items account for 49.70% of all test interactions.
- The test period is only one week after the train period.
- Fashion demand is seasonal and campaign-driven.

Under this condition, a global popularity ranking can be very hard to beat.
This does not mean personalization is useless, but it means the current learned
models are not extracting enough personalized signal to overcome the strong
global trend baseline.

### 10.2 Why Hybrid Does Not Beat MF or Popularity

Likely reasons:

1. Visual features are too sparse.
   - Only 996 article vectors are available.
   - Only 1.97% of test interactions have visual feature coverage.
   - Most items therefore use zero/missing visual vectors.

2. Hybrid overfits quickly.
   - Best validation loss is epoch 1.
   - Later epochs reduce train loss but worsen validation loss.

3. Hybrid inherits NCF weaknesses.
   - Hybrid uses an NCF latent backbone.
   - Standalone NCF performs worst in ranking.
   - Fusion with sparse content improves over NCF but does not surpass MF.

4. Objective mismatch.
   - Training optimizes BCE over sampled negatives.
   - Evaluation optimizes top-12 ranking.
   - This mismatch can cause lower loss but weaker MAP/NDCG.

5. Popularity is very strong in this temporal split.
   - A learned model must beat a high baseline, not a weak random baseline.

### 10.3 Why NCF Has Low Ranking Despite Low Loss

NCF train loss decreases from 0.2823 to 0.1769, but MAP@12 is only 0.0412.
This means binary classification loss is not translating to ranking quality.

Possible causes:

- Overfitting to sampled negatives.
- Mismatch between train negatives and evaluation candidates.
- Model scores are not calibrated for top-12 ranking.
- MLP capacity may be too high for the available personalized signal.
- Regularization/dropout/learning rate may need tuning.

### 10.4 Why MF Is Stronger Than NCF

MF is simpler and more stable:

- It has fewer parameters.
- It uses user/item biases, which can capture popularity-like effects.
- It is less likely to overfit than NCF.
- It performs better in MAP, HitRate, and NDCG.

MF is currently the best learned model in this project.

## 11. Reliability and Caveats

### 11.1 Full Results Are More Reliable Than Report/Smoke

There are three profiles:

| Profile | Train rows | Eval users | Purpose |
|---|---:|---:|---|
| `smoke` | 2,000 | 5 | Sanity check only |
| `report` | 200,000 | 1,000 | Fast report/debug run |
| `full` | all rows | 75,481 | Main result |

Only `full` should be used for final model comparison.

The `report` profile had a different ranking:

```text
MF > Popularity > NCF > Hybrid
```

But it used only 200,000 train rows and 1,000 eval users, so it is not the
strongest evidence for final conclusions.

### 11.2 Candidate Sampling Is Not Full-Catalog Ranking

Evaluation ranks positives against sampled negatives, not all 80k articles.
This is standard for tractability but means:

- Absolute metric values depend on the negative candidate sampling strategy.
- Results should be compared only under the same seed/config.
- A production top-N system may need full-catalog or ANN-based evaluation.

### 11.3 Current Hybrid Is Not a Full Visual Recommender

Because visual feature coverage is only 1.24% of articles, the current Hybrid
result should not be interpreted as a final verdict on visual recommendation.
It is a verdict on the current implementation and current feature coverage.

### 11.4 The Full Directory Contains Mixed Execution History

The latest full run has:

```text
stages=train-compare,test
```

But `logs/train-hybrid.log` exists from a prior full Hybrid training run. This
is acceptable for analysis, but final reporting should state clearly which run
produced which artifact.

## 12. Recommendations

### 12.1 Highest Priority

1. Generate visual features for all or most articles.
   - Current 996 vectors are not enough.
   - Target coverage should be close to all 80,527 articles.
   - Re-evaluate Hybrid only after visual coverage improves.

2. Add a popularity-aware learned model.
   - MF already benefits from item bias.
   - Hybrid/NCF could be blended with popularity:

   ```text
   final_score = alpha * model_score + beta * popularity_score
   ```

   This is pragmatic because Popularity is currently the strongest signal.

3. Use early stopping more aggressively.
   - Best validation is epoch 1 for MF, NCF, and Hybrid in the full run.
   - Running 5 epochs wastes time and can overfit.

4. Cache evaluation candidates.
   - Full test candidate construction is expensive.
   - Cache by seed/config to improve runtime and reproducibility.

### 12.2 Modeling Improvements

1. Try ranking-oriented losses.
   - BPR loss.
   - Pairwise hinge loss.
   - Sampled softmax.
   - Listwise/ranking-aware objectives if feasible.

2. Tune regularization.
   - Lower learning rate.
   - Higher weight decay.
   - More dropout for NCF/Hybrid.
   - Smaller MLP layers.
   - Smaller embedding dimensions.

3. Segment evaluation.
   - Warm users vs cold users.
   - Popular items vs long-tail items.
   - Items with visual features vs items without visual features.
   - Seen test items vs unseen test items.

4. Add recency-aware popularity.
   - The test week is trend-heavy.
   - A decayed popularity baseline may beat global popularity.

5. Add two-stage recommendation.
   - Candidate generation by popularity/MF.
   - Re-ranking by Hybrid where content features exist.

### 12.3 Reporting Improvements

For a senior reviewer, final report should include:

- Full profile metrics as primary table.
- Report/smoke profile metrics only as sanity checks.
- Clear statement that Popularity is currently best.
- Clear explanation that Hybrid is limited by sparse visual coverage.
- Runtime and reproducibility notes.
- Known caveats around sampled negatives and BCE vs ranking metrics.

## 13. Suggested Final Conclusion

The current system successfully implements and evaluates four recommender
families: Popularity, MF, NCF, and Hybrid NCF + visual/metadata features. On the
full temporal test set, the strongest model is the Popularity baseline, followed
by MF, then Hybrid, then NCF. This is plausible because the H&M test week is
highly popularity-driven, with the top 1000 items covering nearly half of test
interactions.

The Hybrid model improves over NCF, but it does not yet beat MF or Popularity.
This should not be interpreted as proof that visual hybrid recommendation is
ineffective. The current visual feature file covers only 996 of 80,527 articles,
and only 1.97% of test interactions have visual feature coverage. Therefore the
current Hybrid experiment is limited by feature availability.

The most important next step is to improve visual feature coverage and then
rerun the full pipeline. In parallel, the learned models should be tuned with
stronger regularization, early stopping, and ranking-aware objectives. A
popularity-aware blend is likely to be a strong practical baseline for the next
iteration.

