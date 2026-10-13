import gzip
import random
import numpy as np
from collections import defaultdict

data_folder = ""
def get_collection():
    collection = {}
    with open(f"{data_folder}/collection.tsv", encoding='utf8') as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                pid, passage = parts[0], parts[1]
                collection[pid] = passage
    print("collection.tsv loaded with {} passages".format(len(collection)))
    return collection

# Load queries.train.tsv into a dictionary: qid -> query
def get_queries():
    queries = {}
    with open(f"{data_folder}/queries.train.tsv", encoding='utf8') as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                qid, query = parts[0], parts[1]
                queries[qid] = query
    print("queries.train.tsv loaded with {} queries".format(len(queries)))
    return queries

def get_msmarco_qidpidtriples():
    msmarco = {}
    with gzip.open(f"{data_folder}/msmarco-qidpidtriples.rnd-shuf.train.tsv.gz", mode='rt', encoding='utf8') as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            qid, pos_id, neg_id = parts[0], parts[1], parts[2]
            if qid not in msmarco:
                msmarco[qid] = {"pos": set(), "neg": set()}
            if pos_id:
                msmarco[qid]["pos"].add(pos_id)
            if neg_id:
                msmarco[qid]["neg"].add(neg_id)
    print("MSMarco_train.tsv.gz loaded with {} queries".format(len(msmarco)))
    return msmarco

def get_msmarco_qidpidtriples_retrieved(file):
    msmarco = {}
    with gzip.open(file, mode='rt', encoding='utf8') as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            qid, pos_id, neg_id = parts[0], parts[1], parts[2]
            if qid not in msmarco:
                msmarco[qid] = {"pos": set(), "neg": set()}
            if pos_id:
                msmarco[qid]["pos"].add(pos_id)
            if neg_id:
                msmarco[qid]["neg"].add(neg_id)
    print("MSMarco_train.tsv.gz loaded with {} queries".format(len(msmarco)))
    return msmarco

def get_msmarco_qidpidtriples_withscore():
    msmarco = {}
    with gzip.open(f"{data_folder}/msmarco-qidpidtriples-withdocscore.rnd-shuf.train.tsv.gz", mode='rt', encoding='utf8') as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 4:
                continue
            qid, pos_id, neg_id, score = parts[0], parts[1], parts[2], parts[3]
            if qid not in msmarco:
                msmarco[qid] = {"pos": set(), "neg": set()}
            if pos_id:
                msmarco[qid]["pos"].add((pos_id, score))
            if neg_id:
                msmarco[qid]["neg"].add((neg_id, score))
    print("MSMarco_train.tsv.gz loaded with {} queries".format(len(msmarco)))
    return msmarco

def get_train_triples(model1_name, model2_name, alpha):
    all_results = []

    for i in range(10):
        f1_dict = {}
        f2_dict = {}

        # Read model1 shard
        with gzip.open(f"{data_folder}/{model1_name}/bestmodel/{i}-msmarco-qidpidtriples.rnd-shuf.train.tsv.gz", 
                       mode='rt', encoding='utf8') as f1:
            for line in f1:
                if ("score" in line.lower() or "qid" in line.lower() or "pos_id" in line.lower() or "neg_id" in line.lower()):
                    print(line)
                    continue
                parts = line.strip().split('\t')
                qid, pos_id, neg_id, score_str = parts[0], parts[1], parts[2], parts[3]

                f1_dict[(qid, pos_id, neg_id)] = float(score_str)

        if(alpha == 0):
            combined_dict = defaultdict(lambda: defaultdict(float))
        
            # Compute raw combined scores before normalization
            for triple, score1 in f1_dict.items():
                # triple is (qid, pos_id, neg_id)
                qid, pos_id, neg_id = triple
                combined_dict[(qid, pos_id)][neg_id] = score1

            # Normalize combined scores
            for (qid, pos_id), neg_dict in combined_dict.items():
                denom = sum(neg_dict.values())
                denom = max(denom, 1e-8)  # Avoid division by zero
                for neg_id, raw_combined in neg_dict.items():
                    # final score = combined / sum_of_combined
                    final_score = raw_combined / denom
                    all_results.append((qid, pos_id, neg_id, final_score))
        else: 
            # Read model2 shard
            with gzip.open(f"{data_folder}/{model2_name}/bestmodel/{i}-msmarco-qidpidtriples.rnd-shuf.train.tsv.gz", 
                        mode='rt', encoding='utf8') as f2:
                for line in f2:
                    if ("score" in line.lower() or "qid" in line.lower() or "pos_id" in line.lower() or "neg_id" in line.lower()):
                        print(line)
                        continue
                    parts = line.strip().split('\t')
                    qid, pos_id, neg_id, score_str = parts[0], parts[1], parts[2], parts[3]
                    # if score_str can't be converted to float, print it
                    
                    f2_dict[(qid, neg_id)] = float(score_str)

            # Combine scores into (qid,pos_id)-based groups
            combined_dict = defaultdict(lambda: defaultdict(float))
            
            # Compute raw combined scores before normalization
            for triple, score1 in f1_dict.items():
                # triple is (qid, pos_id, neg_id)
                # We assume everything matches in f2_dict
                score2 = f2_dict[(triple[0], triple[2])]
                combined_score = score1 + alpha * (score1 - score2)
                if combined_score < 0:
                    combined_score = 0
                qid, pos_id, neg_id = triple
                combined_dict[(qid, pos_id)][neg_id] = combined_score

            # Normalize combined scores
            for (qid, pos_id), neg_dict in combined_dict.items():
                denom = sum(neg_dict.values())
                denom = max(denom, 1e-8)  # Avoid division by zero
                for neg_id, raw_combined in neg_dict.items():
                    # final score = combined / sum_of_combined
                    final_score = raw_combined / denom
                    all_results.append((qid, pos_id, neg_id, final_score))

    return all_results

def get_train_triples_msmarcoquora(model1_name, model2_name, alpha, m_or_q):
    if m_or_q == "msmarco":
        data_folder = f""
        all_results = []

        for i in range(10):
            f1_dict = {}
            f2_dict = {}

            # Read model1 shard
            with gzip.open(f"{data_folder}/{model1_name}/{i}-msmarco-qidpidtriples.rnd-shuf.train.tsv.gz", 
                        mode='rt', encoding='utf8') as f1:
                for line in f1:
                    if ("score" in line.lower() or "qid" in line.lower() or "pos_id" in line.lower() or "neg_id" in line.lower()):
                        print(line)
                        continue
                    parts = line.strip().split('\t')
                    qid, pos_id, neg_id, score_str = parts[0], parts[1], parts[2], parts[3]

                    f1_dict[(qid, pos_id, neg_id)] = float(score_str)

            if(alpha == 0):
                combined_dict = defaultdict(lambda: defaultdict(float))
            
                # Compute raw combined scores before normalization
                for triple, score1 in f1_dict.items():
                    # triple is (qid, pos_id, neg_id)
                    qid, pos_id, neg_id = triple
                    combined_dict[(qid, pos_id)][neg_id] = score1

                # Normalize combined scores
                for (qid, pos_id), neg_dict in combined_dict.items():
                    denom = sum(neg_dict.values())
                    denom = max(denom, 1e-8)  # Avoid division by zero
                    for neg_id, raw_combined in neg_dict.items():
                        # final score = combined / sum_of_combined
                        final_score = raw_combined / denom
                        all_results.append((qid, pos_id, neg_id, final_score))
            else: 
                # Read model2 shard
                with gzip.open(f"{data_folder}/{model2_name}/{i}-msmarco-qidpidtriples.rnd-shuf.train.tsv.gz", 
                            mode='rt', encoding='utf8') as f2:
                    for line in f2:
                        if ("score" in line.lower() or "qid" in line.lower() or "pos_id" in line.lower() or "neg_id" in line.lower()):
                            print(line)
                            continue
                        parts = line.strip().split('\t')
                        qid, pos_id, neg_id, score_str = parts[0], parts[1], parts[2], parts[3]
                        # if score_str can't be converted to float, print it
                        
                        f2_dict[(qid, neg_id)] = float(score_str)

                # Combine scores into (qid,pos_id)-based groups
                combined_dict = defaultdict(lambda: defaultdict(float))
                
                # Compute raw combined scores before normalization
                for triple, score1 in f1_dict.items():
                    # triple is (qid, pos_id, neg_id)
                    # We assume everything matches in f2_dict
                    score2 = f2_dict[(triple[0], triple[2])]
                    combined_score = score1 + alpha * (score1 - score2)
                    if combined_score < 0:
                        combined_score = 0
                    qid, pos_id, neg_id = triple
                    combined_dict[(qid, pos_id)][neg_id] = combined_score

                # Normalize combined scores
                for (qid, pos_id), neg_dict in combined_dict.items():
                    denom = sum(neg_dict.values())
                    denom = max(denom, 1e-8)  # Avoid division by zero
                    for neg_id, raw_combined in neg_dict.items():
                        # final score = combined / sum_of_combined
                        final_score = raw_combined / denom
                        all_results.append((qid, pos_id, neg_id, final_score))

    elif m_or_q == "quora":
        data_folder = f""
        all_results = []

        f1_dict = {}
        f2_dict = {}

        # Read model1 shard
        with gzip.open(f"{data_folder}/{model1_name}/queries.train.tsv", 
                    mode='rt', encoding='utf8') as f1:
            for line in f1:
                if ("score" in line.lower() or "qid" in line.lower() or "pos_id" in line.lower() or "neg_id" in line.lower()):
                    print(line)
                    continue
                parts = line.strip().split('\t')
                qid, pos_id, neg_id, score_str = parts[0], parts[1], parts[2], parts[3]

                f1_dict[(qid, pos_id, neg_id)] = float(score_str)

        if(alpha == 0):
            combined_dict = defaultdict(lambda: defaultdict(float))
        
            # Compute raw combined scores before normalization
            for triple, score1 in f1_dict.items():
                # triple is (qid, pos_id, neg_id)
                qid, pos_id, neg_id = triple
                combined_dict[(qid, pos_id)][neg_id] = score1

            # Normalize combined scores
            for (qid, pos_id), neg_dict in combined_dict.items():
                denom = sum(neg_dict.values())
                denom = max(denom, 1e-8)  # Avoid division by zero
                for neg_id, raw_combined in neg_dict.items():
                    # final score = combined / sum_of_combined
                    final_score = raw_combined / denom
                    all_results.append((qid, pos_id, neg_id, final_score))
        else: 
            # Read model2 shard
            with gzip.open(f"{data_folder}/{model2_name}/queries.train.tsv", 
                        mode='rt', encoding='utf8') as f2:
                for line in f2:
                    if ("score" in line.lower() or "qid" in line.lower() or "pos_id" in line.lower() or "neg_id" in line.lower()):
                        print(line)
                        continue
                    parts = line.strip().split('\t')
                    qid, pos_id, neg_id, score_str = parts[0], parts[1], parts[2], parts[3]
                    # if score_str can't be converted to float, print it
                    
                    f2_dict[(qid, neg_id)] = float(score_str)

            # Combine scores into (qid,pos_id)-based groups
            combined_dict = defaultdict(lambda: defaultdict(float))
            
            # Compute raw combined scores before normalization
            for triple, score1 in f1_dict.items():
                # triple is (qid, pos_id, neg_id)
                # We assume everything matches in f2_dict
                score2 = f2_dict[(triple[0], triple[2])]
                combined_score = score1 + alpha * (score1 - score2)
                if combined_score < 0:
                    combined_score = 0
                qid, pos_id, neg_id = triple
                combined_dict[(qid, pos_id)][neg_id] = combined_score

            # Normalize combined scores
            for (qid, pos_id), neg_dict in combined_dict.items():
                denom = sum(neg_dict.values())
                denom = max(denom, 1e-8)  # Avoid division by zero
                for neg_id, raw_combined in neg_dict.items():
                    # final score = combined / sum_of_combined
                    final_score = raw_combined / denom
                    all_results.append((qid, pos_id, neg_id, final_score))

    return all_results

def get_train_triples_retrieved(model1_name, model2_name, alpha, sample_size):
    data_path = ""
    all_results = []

    for i in range(10):
        f1_dict = {}
        f2_dict = {}

        # Read model1 shard
        with gzip.open(f"{data_path}/{model1_name}/retrieved_top{sample_size}/{i}-msmarco-qidpidtriples.rnd-shuf.train.tsv.gz", 
                       mode='rt', encoding='utf8') as f1:
            for line in f1:
                if ("score" in line.lower() or "qid" in line.lower() or "pos_id" in line.lower() or "neg_id" in line.lower()):
                    print(line)
                    continue
                parts = line.strip().split('\t')
                qid, pos_id, neg_id, score_str = parts[0], parts[1], parts[2], parts[3]

                f1_dict[(qid, pos_id, neg_id)] = float(score_str)

        if(alpha == 0):
            combined_dict = defaultdict(lambda: defaultdict(float))
        
            # Compute raw combined scores before normalization
            for triple, score1 in f1_dict.items():
                # triple is (qid, pos_id, neg_id)
                qid, pos_id, neg_id = triple
                combined_dict[(qid, pos_id)][neg_id] = score1

            # Normalize combined scores
            for (qid, pos_id), neg_dict in combined_dict.items():
                denom = sum(neg_dict.values())
                denom = max(denom, 1e-8)  # Avoid division by zero
                for neg_id, raw_combined in neg_dict.items():
                    # final score = combined / sum_of_combined
                    final_score = raw_combined / denom
                    all_results.append((qid, pos_id, neg_id, final_score))
        else: 
            # Read model2 shard
            with gzip.open(f"{data_path}/{model1_name}/retrieved_top{sample_size}/{i}-msmarco-qidpidtriples.rnd-shuf.train.tsv.gz", 
                        mode='rt', encoding='utf8') as f2:
                for line in f2:
                    if ("score" in line.lower() or "qid" in line.lower() or "pos_id" in line.lower() or "neg_id" in line.lower()):
                        print(line)
                        continue
                    parts = line.strip().split('\t')
                    qid, pos_id, neg_id, score_str = parts[0], parts[1], parts[2], parts[3]
                    # if score_str can't be converted to float, print it
                    
                    f2_dict[(qid, neg_id)] = float(score_str)

            # Combine scores into (qid,pos_id)-based groups
            combined_dict = defaultdict(lambda: defaultdict(float))
            
            # Compute raw combined scores before normalization
            for triple, score1 in f1_dict.items():
                # triple is (qid, pos_id, neg_id)
                # We assume everything matches in f2_dict
                score2 = f2_dict[(triple[0], triple[2])]
                combined_score = score1 + alpha * (score1 - score2)
                if combined_score < 0:
                    combined_score = 0
                qid, pos_id, neg_id = triple
                combined_dict[(qid, pos_id)][neg_id] = combined_score

            # Normalize combined scores
            for (qid, pos_id), neg_dict in combined_dict.items():
                denom = sum(neg_dict.values())
                denom = max(denom, 1e-8)  # Avoid division by zero
                for neg_id, raw_combined in neg_dict.items():
                    # final score = combined / sum_of_combined
                    final_score = raw_combined / denom
                    all_results.append((qid, pos_id, neg_id, final_score))

    return all_results

def get_train_triples_beir_retrieved(file_name):
    all_results = []

    f1_dict = {}
    f2_dict = {}

    # Read model1 shard
    with gzip.open(file_name, mode='rt', encoding='utf8') as f1:
        for line in f1:
            if ("score" in line.lower() or "qid" in line.lower() or "pos_id" in line.lower() or "neg_id" in line.lower()):
                print(line)
                continue
            parts = line.strip().split('\t')
            qid, pos_id, neg_id, score_str = parts[0], parts[1], parts[2], parts[3]

            f1_dict[(qid, pos_id, neg_id)] = float(score_str)

    combined_dict = defaultdict(lambda: defaultdict(float))

    # Compute raw combined scores before normalization
    for triple, score1 in f1_dict.items():
        # triple is (qid, pos_id, neg_id)
        qid, pos_id, neg_id = triple
        combined_dict[(qid, pos_id)][neg_id] = score1

    # Normalize combined scores
    for (qid, pos_id), neg_dict in combined_dict.items():
        denom = sum(neg_dict.values())
        denom = max(denom, 1e-8)  # Avoid division by zero
        for neg_id, raw_combined in neg_dict.items():
            # final score = combined / sum_of_combined
            final_score = raw_combined / denom
            all_results.append((qid, pos_id, neg_id, final_score))
    return all_results

def get_top_k(all_results, skip_num, k):
    grouped_by_qid = defaultdict(list)
    # Group the data by qid
    for (qid, pos_id, neg_id, final_score) in all_results:
        grouped_by_qid[qid].append((pos_id, neg_id, final_score))
    
    final_results = []
    
    for qid, triple_list in grouped_by_qid.items():
        if len(triple_list) < skip_num:
            continue        

        pvals = np.array([t[2] for t in triple_list], dtype=float)
        counts = np.random.multinomial(k//2, pvals)
        #counts = np.random.multinomial(k, pvals)

        # Take k random samples
        random_k_triples = random.choices(triple_list, k=k//2)
        
        # Combine into the final list
        for (count, (pos_id, neg_id, score)) in zip(counts, triple_list):
            for _ in range(count):
                final_results.append((qid, pos_id, neg_id))
        for pos_id, neg_id, score in random_k_triples:
            final_results.append((qid, pos_id, neg_id))
    
    return final_results

def get_top_k_without_RS(all_results, skip_num, k):
    grouped_by_qid = defaultdict(list)
    # Group the data by qid
    for (qid, pos_id, neg_id, final_score) in all_results:
        grouped_by_qid[qid].append((pos_id, neg_id, final_score))
    
    final_results = []
    
    for qid, triple_list in grouped_by_qid.items():
        if len(triple_list) < skip_num:
            continue        

        pvals = np.array([t[2] for t in triple_list], dtype=float)
        counts = np.random.multinomial(k, pvals)
        
        # Combine into the final list
        for (count, (pos_id, neg_id, score)) in zip(counts, triple_list):
            for _ in range(count):
                final_results.append((qid, pos_id, neg_id))
    
    return final_results

def get_top_k_top_n_sampling(all_results, skip_num, k, top_n: 10):
    grouped_by_qid = defaultdict(list)
    # Group the data by qid
    for (qid, pos_id, neg_id, final_score) in all_results:
        grouped_by_qid[qid].append((pos_id, neg_id, final_score))
    
    final_results = []
    
    for qid, triple_list in grouped_by_qid.items():
        triple_list_sorted = sorted(triple_list, key=lambda x: x[2], reverse=True) # Sort by score in descending order
        triple_list = triple_list_sorted[top_n:]
        if len(triple_list) < skip_num:
            continue        

        pvals = np.array([t[2] for t in triple_list], dtype=float)
        counts = np.random.multinomial(k//2, pvals)
        #counts = np.random.multinomial(k, pvals)

        # Take k random samples
        random_k_triples = random.choices(triple_list, k=k//2)
        
        # Combine into the final list
        for (count, (pos_id, neg_id, score)) in zip(counts, triple_list):
            for _ in range(count):
                final_results.append((qid, pos_id, neg_id))
        for pos_id, neg_id, score in random_k_triples:
            final_results.append((qid, pos_id, neg_id))
    
    return final_results

def get_msmarco_bm25(data_folder, k=24, topBM25=True, skip = 0):
    msmarco = {}
    final_results = []

    with gzip.open(f"{data_folder}/msmarco-qidpidtriples-withdocscore.rnd-shuf.train.tsv.gz", 
                   mode='rt', encoding='utf8') as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 4:
                continue

            qid, pos_id, neg_id, score = parts[0], parts[1], parts[2], parts[3]

            if qid not in msmarco:
                msmarco[qid] = {"pos": set(), "neg": set()}

            if pos_id:
                msmarco[qid]["pos"].add(pos_id)
            if neg_id:
                msmarco[qid]["neg"].add((neg_id, score))

    for qid, items in msmarco.items():
        pos_ids = list(items["pos"])
        neg_ids_scores = list(items["neg"])
        len_neg = len(neg_ids_scores)

        if not pos_ids:
            continue
        if len_neg < skip:
            continue
        elif len_neg < k and len_neg >= skip:
            neg_ids_scores = random.choices(neg_ids_scores, k=len_neg)
        else:
            neg_ids_scores.sort(key=lambda x: float(x[1]), reverse=True)
            if topBM25:
                neg_ids_scores = neg_ids_scores[:k]
            else:
                neg_ids_scores = neg_ids_scores[-k:]

        pos_id_chosen = random.choice(pos_ids)

        for (neg_id, _) in neg_ids_scores:
            final_results.append([qid, pos_id_chosen, neg_id])

    return final_results

def get_msmarco_bm25_randommultinomial(data_folder, k=24, topBM25=True, skip = 0):
    msmarco = defaultdict(lambda: {"pos": set(), "neg": []})

    path = f"{data_folder}/msmarco-qidpidtriples-withdocscore.rnd-shuf.train.tsv.gz"
    with gzip.open(path, mode="rt", encoding="utf8") as f:
        for line in f:
            qid, pos_id, neg_id, score = line.rstrip().split("\t")[:4]

            if pos_id:
                msmarco[qid]["pos"].add(pos_id)
            if neg_id:
                msmarco[qid]["neg"].append((neg_id, float(score)))

    final_results = []

    for qid, items in msmarco.items():
        pos_ids          = list(items["pos"])
        neg_ids_scores   = items["neg"]
        if not pos_ids or len(neg_ids_scores) < skip:
            continue

        # Build probability vector
        scores = np.array([s for _, s in neg_ids_scores], dtype=float)
        scores = np.maximum(scores, 1e-12)
        if scores.sum() == 0:  # If all scores are zero, avoid division by zero
            print(f'sum of scores is zero for qid {qid}')
        if not topBM25:                                 # inverse for low‑score‑best
            scores = 1.0 / np.maximum(scores, 1e-12)    # avoid divide‑by‑zero
        pvals  = scores / scores.sum()

        # Sample counts with numpy
        counts = np.random.multinomial(k//2, pvals)

        # Keep one positive pid per qid
        pos_id_chosen = random.choice(pos_ids)

        for (neg_pid, _), c in zip(neg_ids_scores, counts):
            if c:                                         
                final_results.extend([[qid, pos_id_chosen, neg_pid]] * c)  
        neg_ids_scores = random.choices(neg_ids_scores, k=k//2)
        for (neg_id, _) in neg_ids_scores:
            final_results.append([qid, pos_id_chosen, neg_id])
    return final_results
