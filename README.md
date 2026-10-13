# Difficulty-Controlled Negative Sampling for Cross-Encoder Reranking

Research code for training and evaluating cross-encoder rerankers with different negative-sampling strategies. The repository contains multiple training scripts corresponding to different experimental settings, together with utilities for loading training data, generating BM25 candidate sets, and evaluating rerankers.

## Repository structure

The main components are:


| File                              | Purpose                                                                                                             |
| --------------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| `train_*.py`                      | Training scripts for different reranker, negative-sampling, and scheduling configurations.                          |
| `load_traindata.py`               | Loads MS MARCO training data, candidate negatives, and precomputed difficulty scores.                               |
| `generate_bm25score.py`           | Retrieves BM25 candidates from an`ir_datasets` collection and writes them with BM25 scores.                         |
| `eval_msmarco_rerank_ce_large.py` | Evaluates trained cross-encoders on MS MARCO and TREC-DL, with additional evaluation hooks used in the experiments. |

Examples of training scripts included in this code snapshot are `train_decay.py` and `train_top_n_sampling.py`. Other `train_*.py` files may be used for additional experimental configurations.

> **Important:** several scripts contain empty path placeholders such as `data_folder = ""`, `model_folder = ""`, and `model_save_path = ""`. Set these paths for your local environment before running the code.

## Overview

The experiments train cross-encoder rerankers using candidate negatives drawn from a first-stage retrieval pool. An auxiliary cross-encoder can be used to score candidate negatives and provide a difficulty signal.

Different training scripts implement different experimental settings, including:

- random negative sampling;
- difficulty-aware negative sampling;
- mixtures of difficulty-aware and random negatives;
- scheduling the contribution of difficulty-aware negatives during training;
- alternative candidate-generation and model configurations.

The exact model, candidate pool, sampler, batch size, and scheduling strategy are defined in the corresponding `train_*.py` script.

## Generating BM25 candidate files

`generate_bm25score.py` generates BM25 candidates for a dataset available through `ir_datasets`.

Example:

```bash
python generate_bm25score.py \
  --dataset beir/quora/dev \
  --k 3000 \
  --output data/quora-bm25-top3000.tsv.gz
```

For debugging:

```bash
python generate_bm25score.py \
  --dataset beir/quora/dev \
  --k 3000 \
  --output data/quora-bm25-top3000.tsv.gz \
  --debug
```

The script loads the corpus, queries, and qrels from `ir_datasets`, builds a BM25 index, retrieves the top-\(k\) documents for each query, and writes the results as a gzipped TSV file.

Output format:

```text
query_id<TAB>passage_id<TAB>query<TAB>passage<TAB>bm25_score
```

Judged relevant documents that are missing from the retrieved candidate set are added with score `0`, so the number of output rows for a query can be slightly larger than `k`.

## Training

Training configurations are stored directly in the individual `train_*.py` scripts.

Before running an experiment, check the selected script and configure:

- the training data path;
- the pretrained cross-encoder checkpoint;
- the auxiliary difficulty model or score directory, when used;
- the output directory;
- batch size and other optimization settings;
- the candidate/sampling configuration required by that experiment.

For example:

```bash
python train_decay.py
```

or:

```bash
python train_top_n_sampling.py
```

The provided training scripts use `sentence-transformers` cross-encoders and save the trained model to the configured output directory. Random seeds are set in the scripts for Python, NumPy, and PyTorch.

Because this repository contains multiple experimental training scripts, there is no single global training configuration. Refer to the corresponding script for the exact model and hyperparameters used in a particular experiment.

## Evaluation

`EvalCERerank` is used during training and for reranking evaluation.

The provided evaluator reports metrics including:

- **MS MARCO dev**
  - MRR@10
  - Recall@10
  - Recall@50
- **TREC Deep Learning 2019**
  - nDCG@10
- **TREC Deep Learning 2020**
  - nDCG@10

The evaluation code also contains a project-specific hook for additional nDCG evaluation.

For MS MARCO dev, the evaluator expects a gzipped candidate file with at least:

```text
query_id<TAB>passage_id<TAB>query<TAB>passage
```

A fifth score column is also supported:

```text
query_id<TAB>passage_id<TAB>query<TAB>passage<TAB>score
```

MS MARCO relevance judgments are loaded through:

```python
ir_datasets.load("msmarco-passage/dev/judged")
```

For TREC DL 2019 and 2020, the evaluator downloads the official query, qrel, and candidate files if they are not already available in the configured evaluation directory.

When CSV output is enabled, evaluation results are written to files of the form:

```text
CERerankingEvaluator_<name>_results_@10.csv
```

## Reproducibility

The training scripts set random seeds for Python, NumPy, and PyTorch. However, exact results can still depend on the CUDA, PyTorch, and `sentence-transformers` versions and on GPU nondeterminism.

For reproducible experiments, record:

- the exact training script used;
- model checkpoints;
- candidate files;
- difficulty-score files;
- package versions;
- GPU/CUDA environment.
