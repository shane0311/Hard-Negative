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

from eval_msmarco_rerank_ce_large import EvalCERerank
from load_traindata import get_collection, get_queries, get_train_triples, get_top_k, get_msmarco_qidpidtriples, get_top_k_without_RS, get_top_k_top_n_sampling
import numpy as np
import random

print("start")

#### Just some code to print debug information to stdout
logging.basicConfig(
    format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO, handlers=[LoggingHandler()]
)

TRAIN_BATCH_SIZE = 16
EPOCHS = 2
EVALUATION_STEPS = 100000
WARMUP_STEPS = 0
LEARNING_RATE = 2e-5

skip = 4
pos_counts = 6
neg_counts = 24
ALPHA = 0
model1 = "TinyBert"
#model1 = "Bert-base"
model2 = "MiniLM-L12"


if ALPHA > 0:
    ICD_name = f"{model1}-{model2}"
else:
    ICD_name = model1
# First, we define the transformer model we want to fine-tune
base_model = "Bert-base"
model_folder = f""
model_name = f"{model_folder}bert-base-uncased"
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
msmarco = get_msmarco_qidpidtriples()

train_triplets = get_train_triples(model1, model2, alpha=ALPHA)
#train_triplets_ICD = get_top_k(train_triplets, skip_num = skip, k = neg_counts)
train_triplets_ICD = get_top_k_top_n_sampling(train_triplets, skip_num = skip, k = neg_counts, top_n = 10)
#train_triplets_ICD = get_top_k_without_RS(train_triplets, skip_num = skip, k = neg_counts)

train_samples = []

appeared_qid = set()
for qid, pos_id, neg_id in train_triplets_ICD:
    query_text = queries[qid]
    pos_passage = collection[pos_id]
    neg_text = collection.get(neg_id, "")
    if qid not in appeared_qid:
        for i in range (pos_counts):
            train_samples.append([query_text, pos_passage, pos_passage, 1])
        appeared_qid.add(qid)

    if neg_text:
        random_neg_id = random.choice(list(msmarco[qid]["neg"]))
        train_samples.append([query_text, neg_text, collection.get(random_neg_id, ""),0])
    else:
        print(f"Skipped empty negative passage for ID: {neg_id}")

# training_samples: query_i, passage_1_i_x, passage_2_i_x, label_i_x
# passage_1_i_x: The x(0 < x < 31) passage from 6 positive + 
#            12 model picked negative + 12 Random sampling negative from query_i
# passage_2_i_x: The x(0 < x < 31) passage from 6 positive + 24 Random sampling negative 
#            from query_i

# passage_1_i_x and passage_2_i_x have same label same query

random.shuffle(train_samples)

len_training_samples = len(train_samples)
final_training_samples = []
for i in range (len_training_samples):
    random_float = random.random() # a random float between 0.0 and 1.0
    if(random_float >= i/len_training_samples):
        #final_training_samples.append([query_i, passage_1_i_x, label_i_x])
        final_training_samples.append(InputExample(texts=[train_samples[i][0], train_samples[i][1]], label=train_samples[i][3]))
    else:
        #final_training_samples.append([query_i, passage_2_i_x, label_i_x])
        final_training_samples.append(InputExample(texts=[train_samples[i][0], train_samples[i][2]], label=train_samples[i][3]))

train_dataloader = DataLoader(final_training_samples, shuffle=False, batch_size=TRAIN_BATCH_SIZE)

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