# Metrics Comparison Table

## ML-20M | Leave-One-Out | Full-Catalog Ranking

| Model                        | HR@10  | NDCG@10 | MRR@10 | vs Base |
|------------------------------|--------|---------|--------|---------|
| SASRec+ (base, 3000 neg)    | 0.3090 | 0.1870  | --     | --      |
| SASRec+ ReaRec-ERL (K=2)    | 0.2803 | 0.1644  | 0.1289 | -9.3%   |
| SASRec+ ReaRec-PRL (K=2)    | 0.2499 | 0.1414  | 0.1084 | -19.1%  |

### Base SASRec+ Hyperparameters
- hidden_units: 256, num_blocks: 2, num_heads: 1
- dropout: 0.1, max_length: 200, batch_size: 128
- lr: 1e-3, num_negatives: 3000, loss: sampled softmax CE
- optimizer: Adam, early stopping patience: 10

### ReaRec Hyperparameters (on top of base)
- reason_steps: 2, reason_loss_weight: 0.1
- ERL: kl_weight: 0.05, kl_temperature: 1.0, num_neg_reason: 256
- PRL: temp_scale: 5.0, pl_weight: 1.0

### Why ReaRec underperforms on ML-20M
The paper reports gains on smaller/sparser datasets (Yelp, Amazon subsets) with
NDCG@10 baselines of 0.01-0.07 and max_len=50. ML-20M is fundamentally different:
- Dense dataset: avg 144 interactions/user (vs ~10-20 in paper's datasets)
- Long sequences: max_len=200 means the base model already has rich context
- Strong baseline: SASRec+ with 3000 neg sampled softmax is already well-optimized
- Paper's ablation shows reasoning helps short-history users but hurts long-history
  users, and ML-20M is dominated by long histories

### ReaRec Paper Results (other datasets, for reference)
| Dataset       | Model          | NDCG@10 | Recall@10 | Improvement |
|---------------|----------------|---------|-----------|-------------|
| Yelp          | SASRec Base    | 0.0347  | 0.0626    | --          |
| Yelp          | SASRec + ERL   | 0.0383  | 0.0691    | +10.4%      |
| Yelp          | SASRec + PRL   | 0.0388  | 0.0730    | +11.8%      |
| Video Games   | SASRec Base    | 0.0284  | 0.0542    | --          |
| Video Games   | SASRec + ERL   | 0.0301  | 0.0581    | +6.0%       |
| Video Games   | SASRec + PRL   | 0.0299  | 0.0572    | +5.3%       |
