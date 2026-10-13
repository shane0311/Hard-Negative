import gzip
import logging
import os
import tarfile
import random
from datetime import datetime
from collections import defaultdict

import tqdm
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import csv

from torch.utils.data import DataLoader

from sentence_transformers import InputExample, LoggingHandler, util
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CERerankingEvaluator

from eval_msmarco_rerank_ce import EvalCERerank
from load_traindata import get_collection, get_queries, get_train_triples, get_top_k, get_top_k_top_n_sampling
import numpy as np
import random

print("start")
# Load collection.tsv into a dictionary: pid -> passage

#### Just some code to print debug information to stdout
logging.basicConfig(
    format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO, handlers=[LoggingHandler()]
)

TRAIN_BATCH_SIZE = 128
EPOCHS = 2
SHUFFLE = True
EVALUATION_STEPS = 100000
WARMUP_STEPS = 0
LEARNING_RATE = 2e-5

skip = 4
pos_counts = 6
neg_counts = 24
ALPHA = 0
#model1 = "TinyBert"
model1 = "MiniLM-L12"
model2 = "MiniLM-L12"


if ALPHA > 0:
    ICD_name = f"{model1}-{model2}"
else:
    ICD_name = model1
# First, we define the transformer model we want to fine-tune
base_model = "TinyBert"
model_folder = f""
model_name = f"{model_folder}BERT-Tiny_L-2_H-128_A-2"
model_save_path = (
    f""
    + datetime.now().strftime("%Y-%m-%d_%H-%M-%S").replace(":", "-")
)
os.makedirs(model_save_path, exist_ok=True)

print("Is CUDA avaliable? ", torch.cuda.is_available())
print(torch.__version__)
print(torch.version.cuda)


torch.manual_seed(0)
random.seed(0)
np.random.seed(0)
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
g = torch.Generator()
g.manual_seed(0)

model = CrossEncoder(model_name, num_labels=1, max_length=512)

data_folder = ""
collection = get_collection()
queries = get_queries()

train_triplets = get_train_triples(model1, model2, alpha=ALPHA)
train_triplets_ICD = get_top_k(train_triplets, skip_num = skip, k = neg_counts)
#train_triplets_ICD = get_top_k_top_n_sampling(train_triplets, skip_num = skip, k = neg_counts, top_n = 10)

train_samples = []

appeared_qid = set()
for qid, pos_id, neg_id in train_triplets_ICD:
    query_text = queries[qid]
    pos_passage = collection[pos_id]
    neg_text = collection.get(neg_id, "")
    if qid not in appeared_qid:
        for i in range (pos_counts):
            train_samples.append(InputExample(texts=[query_text, pos_passage], label=1))
        appeared_qid.add(qid)

    if neg_text:
        train_samples.append(InputExample(texts=[query_text, neg_text], label=0))
    else:
        print(f"Skipped empty negative passage for ID: {neg_id}")

print(f"Loaded {len(train_samples)} training samples")

train_dataloader = DataLoader(train_samples, shuffle=SHUFFLE, batch_size=TRAIN_BATCH_SIZE)

evaluator = EvalCERerank(name="train-eval")
eval_before_train = evaluator(model=model)

WARMUP_STEPS = len(train_samples) // (TRAIN_BATCH_SIZE*10)
EVALUATION_STEPS = WARMUP_STEPS
logging.info(f"Warmup-steps: {WARMUP_STEPS}")
print("Warmup-steps: ", WARMUP_STEPS)
print("Evaluation steps: ", EVALUATION_STEPS)
print("Learning rate: ", LEARNING_RATE)
print("Batch size: ", TRAIN_BATCH_SIZE)
print("Epochs: ", EPOCHS)
print("Scores before train: ", eval_before_train)
print("Train the model.")
# Train the model
model.fit(
    train_dataloader=train_dataloader,
    evaluator=evaluator,
    epochs=EPOCHS,
    evaluation_steps=EVALUATION_STEPS,
    warmup_steps=WARMUP_STEPS,
    output_path=model_save_path,
    use_amp=True,
    show_progress_bar=True,
    optimizer_params = {"lr": LEARNING_RATE}
)

# Save latest model
model.save(model_save_path + "-latest")