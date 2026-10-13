# Difficulty-Controlled Negative Sampling for Cross-Encoder Reranking

Research code for training and evaluating cross-encoder rerankers with difficulty-aware negative sampling. The code supports score-based negative sampling, random-negative mixing, exclusion of the highest-scoring candidates, and a training schedule that gradually shifts from difficulty-aware negatives toward random negatives.

## Files


| File                              | Purpose                                                                                                                                                            |
| --------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `train.py`                        | Main cross-encoder training script. Builds difficulty-aware and random training examples, applies the decay schedule, trains the reranker, and invokes evaluation. |
| `load_traindata.py`               | Loads MS MARCO data and precomputed negative scores and implements the negative-sampling strategies.                                                               |
| `generate_bm25score.py`           | Builds a BM25 index for an`ir_datasets` collection, retrieves top-\(k\) candidates, and writes candidates with BM25 scores to a gzipped TSV file.                  |
| `eval_msmarco_rerank_ce_large.py` | Evaluates a cross-encoder on MS MARCO dev, TREC DL 2019/2020, and Quora.                                                                                           |

> **Important:** the current scripts contain empty path placeholders (`data_folder = ""`, `model_folder = ""`, `model_save_path = ""`, and evaluation paths). Configure these paths before running the code.

## Method implemented in the code

The training pipeline uses an auxiliary model's scores as a proxy for negative difficulty.

For each query, `load_traindata.py` normalizes the auxiliary scores over candidate negatives. The default configuration in `train.py` calls:

```python
get_top_k_top_n_sampling(
    train_triplets,
    skip_num=4,
    k=24,
    top_n=10,
)
```

This sampler:

1. sorts candidate negatives by score;
2. removes the 10 highest-scoring candidates;
3. samples 12 negatives proportionally to the remaining scores; and
4. samples another 12 negatives randomly from the same remaining pool.

The resulting difficulty-aware training composition therefore contains six positive instances and 24 negative instances per query, where half of the negatives are score-weighted and half are random.

For each selected negative, `train.py` also prepares a random MS MARCO negative. After shuffling the training examples, the script chooses between the difficulty-aware example and its random counterpart with a linearly decreasing probability:

```text
P(difficulty-aware at position i) ≈ 1 - i / N
```

where `N` is the total number of shuffled training examples. Early training therefore favors difficulty-aware sampling, while later training increasingly uses random negatives.

## Negative-sampling variants

`load_traindata.py` includes several sampling functions:

- `get_top_k(...)`: 50% score-proportional sampling + 50% random sampling.
- `get_top_k_without_RS(...)`: all negatives are sampled proportionally to the difficulty scores.
- `get_top_k_top_n_sampling(...)`: removes the top-scoring `top_n` candidates, then uses 50% score-proportional + 50% random sampling.
- `get_msmarco_bm25(...)`: selects high- or low-BM25-score negatives.
- `get_msmarco_bm25_randommultinomial(...)`: combines BM25-score-based multinomial sampling with random sampling.

The active sampler is selected directly in `train.py` by commenting/uncommenting the corresponding call.

## Generating BM25 candidate files

`generate_bm25score.py` can create candidate files from any compatible `ir_datasets` dataset.

Example:

```bash
python generate_bm25score.py \
  --dataset beir/quora/dev \
  --k 3000 \
  --output data/quora-bm25-top3000.tsv.gz
```

Optional debugging:

```bash
python generate_bm25score.py \
  --dataset beir/quora/dev \
  --k 3000 \
  --output data/quora-bm25-top3000.tsv.gz \
  --debug
```

The script:

- loads the corpus, queries, and qrels through `ir_datasets`;
- tokenizes with English stopword removal and stemming;
- builds a global BM25 index with `bm25s`;
- retrieves the top-\(k\) documents for every query;
- appends judged documents missing from the retrieved set with score `0`; and
- writes a gzipped TSV file.

Output format:

```text
query_id<TAB>passage_id<TAB>query<TAB>passage<TAB>bm25_score
```

Because missing judged documents are appended, a query can have more than `k` output rows.

## Training

Before training, configure the constants and paths near the top of `train.py` and `load_traindata.py`.

The current defaults are:

```python
TRAIN_BATCH_SIZE = 16
EPOCHS = 2
LEARNING_RATE = 2e-5

skip = 4
pos_counts = 6
neg_counts = 24

ALPHA = 0
model1 = "TinyBert"
model2 = "MiniLM-L12"

base_model = "Bert-base"
model_name = "bert-base-uncased"
```

Set at least:

- `data_folder` in `load_traindata.py`;
- the model/checkpoint path in `train.py` if not using `bert-base-uncased`;
- `model_save_path` in `train.py`; and
- the MS MARCO dev candidate path required by `EvalCERerank`.

Then run:

```bash
python train.py
```

The script fixes the PyTorch, NumPy, and Python random seeds to `0`, creates a `sentence_transformers.CrossEncoder` with `max_length=512`, and trains for two epochs with mixed precision enabled.

Warmup steps are computed as:

```text
number_of_training_examples / (batch_size * 10)
```

and the same value is used as the evaluation interval.

## Evaluation

`EvalCERerank` evaluates the cross-encoder on:

- **MS MARCO dev**
  - MRR@10
  - Recall@10
  - Recall@50
- **TREC Deep Learning 2019**
  - nDCG@10
- **TREC Deep Learning 2020**
  - nDCG@10
- **BEIR Quora**
  - nDCG@10 through `eval_ndcg_10_training`

The evaluator returns MS MARCO MRR, which is the score used by the `sentence-transformers` training loop for model selection.

When an evaluation output path is supplied, results are appended to:

```text
CERerankingEvaluator_<name>_results_@10.csv
```

## Evaluation data behavior

For MS MARCO dev, the evaluator expects a gzipped candidate file with at least four columns:

```text
query_id<TAB>passage_id<TAB>query<TAB>passage[<TAB>score]
```

It obtains relevance judgments from:

```python
ir_datasets.load("msmarco-passage/dev/judged")
```

For TREC DL 2019 and 2020, the script downloads the official query, qrel, and top-1000 files if they are not already present in the configured evaluation directory.
