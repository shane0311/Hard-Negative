from __future__ import annotations

from collections import defaultdict
import gzip
from sentence_transformers import CrossEncoder, util
import logging
import numpy as np
import tqdm
import pytrec_eval
import csv
import os
import ir_datasets 
from eval_ndcg import eval_ndcg_10_training

logger = logging.getLogger(__name__)

class EvalCERerank:
    """
    This class evaluates a CrossEncoder model for the task of re-ranking.
    """

    def __init__(self, at_k: int = 10, name: str = "", write_csv: bool = True, mrr_at_k: int | None = None):
        self.name = name
        if mrr_at_k is not None:
            logger.warning(f"The `mrr_at_k` parameter has been deprecated; please use `at_k={mrr_at_k}` instead.")
            self.at_k = mrr_at_k
        else:
            self.at_k = at_k

        self.csv_file = "CERerankingEvaluator" + ("_" + name if name else "") + f"_results_@{self.at_k}.csv"
        self.csv_headers = [
            "epoch",
            "steps",
            f"MRR@{self.at_k}",
            f"std_MRR@{self.at_k}",
            f"Recall@10",
            f"std_Recall@10",
            f"Recall@50",
            f"std_Recall@50",           
            f"19_NDCG@{self.at_k}",
            f"19_NDCG_1@{self.at_k}",
            f"19_NDCG_2@{self.at_k}",
            f"20_NDCG@{self.at_k}",
            f"20_NDCG_1@{self.at_k}",
            f"20_NDCG_2@{self.at_k}",
            f"Quora_NDCG@{self.at_k}"
        ]
        self.write_csv = write_csv

    def __call__(self, model, output_path: str = None, epoch: int = -1, steps: int = -1, batch_size = 256, top1000_path: str = "") -> float:
        data_path = ''

        if epoch != -1:
            if steps == -1:
                out_txt = f" after epoch {epoch}:"
            else:
                out_txt = f" in epoch {epoch} after {steps} steps:"
        else:
            out_txt = ":"

        logger.info("CERerankingEvaluator: Evaluating the model on " + self.name + " dataset" + out_txt)

        # === (A) DEV MRR@K ON MS MARCO DEV USING YOUR NEW TOP1000 FILE ===

        if not os.path.exists(top1000_path):
            raise FileNotFoundError(f"Missing file: {top1000_path}")

        # First pass: collect qids present in the reranking file
        dev_qids = set()
        with gzip.open(top1000_path, "rt", encoding="utf-8") as fIn:
            for line in fIn:
                parts = line.rstrip("\n").split("\t")
                # Expected: qid, pid, query, passage, bm25_score
                if len(parts) < 4:
                    continue
                qid = parts[0]
                dev_qids.add(qid)

        # Build relevance dict from ir_datasets qrels (instead of qrels.dev.tsv)
        dataset = ir_datasets.load("msmarco-passage/dev/judged")
        dev_rel_docs: dict[str, set[str]] = {}
        for qrel in dataset.qrels_iter():
            # qrel has fields: query_id, doc_id, relevance, iteration
            if qrel.query_id in dev_qids and int(qrel.relevance) > 0:
                dev_rel_docs.setdefault(qrel.query_id, set()).add(qrel.doc_id)

        # Second pass: gather queries, passages, and retrieved docs per qid
        dev_queries: dict[str, str] = {}
        corpus: dict[str, str] = {}
        retrieved_docs: dict[str, set[str]] = {}

        with gzip.open(top1000_path, "rt", encoding="utf-8") as fIn:
            for line in fIn:
                parts = line.rstrip("\n").split("\t")
                # Handle both 4-col and 5-col variants gracefully
                # (qid, pid, query, passage[, bm25_score])
                if len(parts) < 4:
                    continue
                qid, pid, query, passage = parts[0], parts[1], parts[2], parts[3]
                if qid not in dev_qids:
                    continue
                dev_queries[qid] = query  # safe to overwrite; same query repeats
                corpus[pid] = passage
                retrieved_docs.setdefault(qid, set()).add(pid)

        logging.info("Queries: {}".format(len(dev_queries)))

        mrr_scores = []
        recall_scores = []
        recall_scores_10 = []
        recall_at_10 = 10
        recall_at_k = 50
        for qid in tqdm.tqdm(dev_queries, total=len(dev_queries)):
            query = dev_queries[qid]
            top_pids = list(retrieved_docs.get(qid, []))
            if not top_pids:
                mrr_scores.append(0.0)
                recall_scores.append(0.0)
                recall_scores_10.append(0.0)
                continue

            cross_inp = [[query, corpus[pid]] for pid in top_pids]
            if getattr(model.config, "num_labels", 1) > 1:
                bert_score = model.predict(cross_inp, show_progress_bar=False, apply_softmax=True, batch_size=batch_size)[:, 1]
            else:
                bert_score = model.predict(cross_inp, show_progress_bar=False, batch_size=batch_size)

            argsort = np.argsort(-bert_score)

            rank_score = 0.0
            for rank, idx in enumerate(argsort[0:self.at_k]):
                pid = top_pids[idx]
                if qid in dev_rel_docs and pid in dev_rel_docs[qid]:
                    rank_score = 1.0 / (rank + 1)
                    break

            mrr_scores.append(rank_score)

            # Calculate Recall@k
            recall_score = 0.0
            if qid in dev_rel_docs and len(dev_rel_docs[qid]) > 0:
                relevant_in_topk = 0
                for idx in argsort[0:recall_at_k]:
                    pid = top_pids[idx]
                    if pid in dev_rel_docs[qid]:
                        relevant_in_topk += 1
                recall_score = relevant_in_topk / len(dev_rel_docs[qid])
            recall_scores.append(recall_score)

            recall_score_10 = 0.0
            if qid in dev_rel_docs and len(dev_rel_docs[qid]) > 0:
                relevant_in_topk = 0
                for idx in argsort[0:recall_at_10]:
                    pid = top_pids[idx]
                    if pid in dev_rel_docs[qid]:
                        relevant_in_topk += 1
                recall_score_10 = relevant_in_topk / len(dev_rel_docs[qid])
            recall_scores_10.append(recall_score_10)

            if len(mrr_scores) % 100 == 0:
                logger.info(
                    f"MRR@{self.at_k}: {100*np.mean(mrr_scores):.2f}, "
                    f"Recall@{recall_at_10}: {100*np.mean(recall_scores_10):.2f}, "
                    f"Recall@{recall_at_k}: {100*np.mean(recall_scores):.2f}"
                    )
        mean_mrr = float(np.mean(mrr_scores)) if mrr_scores else 0.0
        std_mrr = float(np.std(mrr_scores)) if mrr_scores else 0.0
        mean_recall = float(np.mean(recall_scores)) if recall_scores else 0.0
        std_recall = float(np.std(recall_scores)) if recall_scores else 0.0

        mean_recall_10 = float(np.mean(recall_scores_10)) if recall_scores_10 else 0.0
        std_recall_10 = float(np.std(recall_scores_10)) if recall_scores_10 else 0.0

        # === (B) TREC DL 2019 NDCG@10 (unchanged) ===
        queries_trec = {}
        queries_trec_filepath = os.path.join(data_path, "msmarco-test2019-queries.tsv.gz")
        if not os.path.exists(queries_trec_filepath):
            logging.info("Download " + os.path.basename(queries_trec_filepath))
            util.http_get(
                "https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco-test2019-queries.tsv.gz",
                queries_trec_filepath
            )

        with gzip.open(queries_trec_filepath, "rt", encoding="utf8") as fIn:
            for line in fIn:
                qid, query = line.strip().split("\t")
                queries_trec[qid] = query

        relevant_docs = defaultdict(lambda: defaultdict(int))
        qrels_trec_filepath = os.path.join(data_path, "2019qrels-pass.txt")
        if not os.path.exists(qrels_trec_filepath):
            logging.info("Download " + os.path.basename(qrels_trec_filepath))
            util.http_get("https://trec.nist.gov/data/deep/2019qrels-pass.txt", qrels_trec_filepath)

        with open(qrels_trec_filepath) as fIn:
            for line in fIn:
                qid, _, pid, score = line.strip().split()
                score = int(score)
                if score > 0:
                    relevant_docs[qid][pid] = score
        # Split relevant_docs to half
        relevant_docs_1 = {}
        relevant_docs_2 = {}
        all_qids_19 = [qid for qid in queries_trec if len(relevant_docs[qid]) > 0]
        mid_19 = len(all_qids_19) // 2
        relevant_qid_1 = set(all_qids_19[:mid_19])
        relevant_qid_2 = set(all_qids_19[mid_19:])

        relevant_docs_1 = {qid: relevant_docs[qid] for qid in relevant_qid_1}
        relevant_docs_2 = {qid: relevant_docs[qid] for qid in relevant_qid_2}

        relevant_qid = [qid for qid in queries_trec if len(relevant_docs[qid]) > 0]

        passage_trec_filepath = os.path.join(data_path, "msmarco-passagetest2019-top1000.tsv.gz")
        if not os.path.exists(passage_trec_filepath):
            logging.info("Download " + os.path.basename(passage_trec_filepath))
            util.http_get(
                "https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco-passagetest2019-top1000.tsv.gz",
                passage_trec_filepath,
            )

        passage_cand = {}
        with gzip.open(passage_trec_filepath, "rt", encoding="utf8") as fIn:
            for line in fIn:
                qid, pid, query, passage = line.strip().split("\t")
                passage_cand.setdefault(qid, []).append([pid, passage])

        logging.info(f"queries_trec: {len(queries_trec)}")

        run = {}
        for qid in tqdm.tqdm(relevant_qid):
            query = queries_trec[qid]
            cand = passage_cand[qid]
            pids = [c[0] for c in cand]
            corpus_sentences = [c[1] for c in cand]

            cross_inp = [[query, sent] for sent in corpus_sentences]
            if getattr(model.config, "num_labels", 1) > 1:
                cross_scores = model.predict(cross_inp, apply_softmax=True, batch_size=batch_size)[:, 1].tolist()
            else:
                cross_scores = model.predict(cross_inp, batch_size=batch_size).tolist()

            run[qid] = {pid: float(score) for pid, score in zip(pids, cross_scores)}

        evaluator = pytrec_eval.RelevanceEvaluator(relevant_docs, {"ndcg_cut.10"})
        ndcg_scores = evaluator.evaluate(run)
        mean_ndcg = float(np.mean([ele["ndcg_cut_10"] for ele in ndcg_scores.values()]))

        run1 = {}
        for qid in tqdm.tqdm(relevant_qid_1):
            query = queries_trec[qid]
            cand = passage_cand[qid]
            pids = [c[0] for c in cand]
            corpus_sentences = [c[1] for c in cand]

            cross_inp = [[query, sent] for sent in corpus_sentences]
            if getattr(model.config, "num_labels", 1) > 1:
                cross_scores = model.predict(cross_inp, apply_softmax=True, batch_size=batch_size)[:, 1].tolist()
            else:
                cross_scores = model.predict(cross_inp, batch_size=batch_size).tolist()

            run1[qid] = {pid: float(score) for pid, score in zip(pids, cross_scores)}

        evaluator_1 = pytrec_eval.RelevanceEvaluator(relevant_docs_1, {"ndcg_cut.10"})
        ndcg_scores_1 = evaluator_1.evaluate(run1)
        mean_ndcg_1 = float(np.mean([ele["ndcg_cut_10"] for ele in ndcg_scores_1.values()]))

        run2 = {}
        for qid in tqdm.tqdm(relevant_qid_2):
            query = queries_trec[qid]
            cand = passage_cand[qid]
            pids = [c[0] for c in cand]
            corpus_sentences = [c[1] for c in cand]

            cross_inp = [[query, sent] for sent in corpus_sentences]
            if getattr(model.config, "num_labels", 1) > 1:
                cross_scores = model.predict(cross_inp, apply_softmax=True, batch_size=batch_size)[:, 1].tolist()
            else:
                cross_scores = model.predict(cross_inp, batch_size=batch_size).tolist()

            run2[qid] = {pid: float(score) for pid, score in zip(pids, cross_scores)}

        evaluator_2 = pytrec_eval.RelevanceEvaluator(relevant_docs_2, {"ndcg_cut.10"})
        ndcg_scores_2 = evaluator_2.evaluate(run2)
        mean_ndcg_2 = float(np.mean([ele["ndcg_cut_10"] for ele in ndcg_scores_2.values()]))

        # === (C) TREC DL 2020 NDCG@10 (unchanged) ===
        queries_trec_20 = {}
        queries_trec_20_filepath = os.path.join(data_path, "msmarco-test2020-queries.tsv.gz")
        if not os.path.exists(queries_trec_20_filepath):
            logging.info("Download " + os.path.basename(queries_trec_20_filepath))
            util.http_get(
                "https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco-test2020-queries.tsv.gz",
                queries_trec_20_filepath
            )

        with gzip.open(queries_trec_20_filepath, "rt", encoding="utf8") as fIn:
            for line in fIn:
                qid, query = line.strip().split("\t")
                queries_trec_20[qid] = query

        relevant_docs_20 = defaultdict(lambda: defaultdict(int))
        qrels_trec_filepath_20 = os.path.join(data_path, "2020qrels-pass.txt")
        if not os.path.exists(qrels_trec_filepath_20):
            logging.info("Download " + os.path.basename(qrels_trec_filepath_20))
            util.http_get("https://trec.nist.gov/data/deep/2020qrels-pass.txt", qrels_trec_filepath_20)

        with open(qrels_trec_filepath_20) as fIn:
            for line in fIn:
                qid, _, pid, score = line.strip().split()
                score = int(score)
                if score > 0:
                    relevant_docs_20[qid][pid] = score

        relevant_docs_20_1 = {}
        relevant_docs_20_2 = {}
        all_qids_20 = [qid for qid in queries_trec_20 if len(relevant_docs_20[qid]) > 0]
        mid_20 = len(all_qids_20) // 2
        relevant_qid_20_1 = set(all_qids_20[:mid_20])
        relevant_qid_20_2 = set(all_qids_20[mid_20:])

        relevant_docs_20_1 = {qid: relevant_docs_20[qid] for qid in relevant_qid_20_1}
        relevant_docs_20_2 = {qid: relevant_docs_20[qid] for qid in relevant_qid_20_2}

        relevant_qid_20 = [qid for qid in queries_trec_20 if len(relevant_docs_20[qid]) > 0]


        passage_trec_20_filepath = os.path.join(data_path, "msmarco-passagetest2020-top1000.tsv.gz")
        if not os.path.exists(passage_trec_20_filepath):
            logging.info("Download " + os.path.basename(passage_trec_20_filepath))
            util.http_get(
                "https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco-passagetest2020-top1000.tsv.gz",
                passage_trec_20_filepath,
            )

        passage_cand_20 = {}
        with gzip.open(passage_trec_20_filepath, "rt", encoding="utf8") as fIn:
            for line in fIn:
                qid, pid, query, passage = line.strip().split("\t")
                passage_cand_20.setdefault(qid, []).append([pid, passage])

        logging.info(f"queries_trec_20: {len(queries_trec_20)}")

        run_20 = {}
        for qid in tqdm.tqdm(relevant_qid_20):
            query = queries_trec_20[qid]
            cand = passage_cand_20[qid]
            pids = [c[0] for c in cand]
            corpus_sentences = [c[1] for c in cand]

            cross_inp = [[query, sent] for sent in corpus_sentences]
            if getattr(model.config, "num_labels", 1) > 1:
                cross_scores = model.predict(cross_inp, apply_softmax=True, batch_size=batch_size)[:, 1].tolist()
            else:
                cross_scores = model.predict(cross_inp, batch_size=batch_size).tolist()

            run_20[qid] = {pid: float(score) for pid, score in zip(pids, cross_scores)}

        evaluator_20 = pytrec_eval.RelevanceEvaluator(relevant_docs_20, {"ndcg_cut.10"})
        ndcg_scores_20 = evaluator_20.evaluate(run_20)
        mean_ndcg_20 = float(np.mean([ele["ndcg_cut_10"] for ele in ndcg_scores_20.values()]))

        run_20_1 = {}
        for qid in tqdm.tqdm(relevant_qid_20_1):
            query = queries_trec_20[qid]
            cand = passage_cand_20[qid]
            pids = [c[0] for c in cand]
            corpus_sentences = [c[1] for c in cand]

            cross_inp = [[query, sent] for sent in corpus_sentences]
            if getattr(model.config, "num_labels", 1) > 1:
                cross_scores = model.predict(cross_inp, apply_softmax=True, batch_size=batch_size)[:, 1].tolist()
            else:
                cross_scores = model.predict(cross_inp, batch_size=batch_size).tolist()

            run_20_1[qid] = {pid: float(score) for pid, score in zip(pids, cross_scores)}

        evaluator_20_1 = pytrec_eval.RelevanceEvaluator(relevant_docs_20_1, {"ndcg_cut.10"})
        ndcg_scores_20_1 = evaluator_20_1.evaluate(run_20_1)
        mean_ndcg_20_1 = float(np.mean([ele["ndcg_cut_10"] for ele in ndcg_scores_20_1.values()]))

        run_20_2 = {}
        for qid in tqdm.tqdm(relevant_qid_20_2):
            query = queries_trec_20[qid]
            cand = passage_cand_20[qid]
            pids = [c[0] for c in cand]
            corpus_sentences = [c[1] for c in cand]

            cross_inp = [[query, sent] for sent in corpus_sentences]
            if getattr(model.config, "num_labels", 1) > 1:
                cross_scores = model.predict(cross_inp, apply_softmax=True, batch_size=batch_size)[:, 1].tolist()
            else:
                cross_scores = model.predict(cross_inp, batch_size=batch_size).tolist()

            run_20_2[qid] = {pid: float(score) for pid, score in zip(pids, cross_scores)}
            
        quora_ndcg = eval_ndcg_10_training(model, beir_name="beir/quora/test", run_filepath="")
        evaluator_20_2 = pytrec_eval.RelevanceEvaluator(relevant_docs_20_2, {"ndcg_cut.10"})
        ndcg_scores_20_2 = evaluator_20_2.evaluate(run_20_2)
        mean_ndcg_20_2 = float(np.mean([ele["ndcg_cut_10"] for ele in ndcg_scores_20_2.values()]))

        # CSV output
        if output_path is not None and self.write_csv:
            csv_path = os.path.join(output_path, self.csv_file)
            output_file_exists = os.path.isfile(csv_path)
            os.makedirs(output_path, exist_ok=True)
            with open(csv_path, mode="a" if output_file_exists else "w", encoding="utf-8") as f:
                writer = csv.writer(f)
                if not output_file_exists:
                    writer.writerow(self.csv_headers)
                writer.writerow([
                    epoch, steps,
                    mean_mrr, std_mrr,
                    mean_recall_10, std_recall_10,   # Recall@10 first
                    mean_recall,   std_recall,       # then Recall@50
                    mean_ndcg, mean_ndcg_1, mean_ndcg_2,
                    mean_ndcg_20, mean_ndcg_20_1, mean_ndcg_20_2,
                    quora_ndcg
                ])

        logger.info(
            f"epoch: {epoch}, steps: {steps}, "
            f"MRR@{self.at_k}: {mean_mrr*100:.4f}, std_MRR@{self.at_k}: {std_mrr*100:.4f}, "
            f"Recall@{recall_at_k}: {mean_recall*100:.4f}, std_Recall@{recall_at_k}: {std_recall*100:.4f}, "
            f"Recall@{recall_at_10}: {mean_recall_10*100:.4f}, std_Recall@{recall_at_10}: {std_recall_10*100:.4f}, "
            f"19_NDCG@{self.at_k}: {mean_ndcg*100:.4f}, 19_NDCG_1@{self.at_k}: {mean_ndcg_1*100:.4f}, "
            f"19_NDCG_2@{self.at_k}: {mean_ndcg_2*100:.4f}, 20_NDCG@{self.at_k}: {mean_ndcg_20*100:.4f}, "
            f"20_NDCG_1@{self.at_k}: {mean_ndcg_20_1*100:.4f}, 20_NDCG_2@{self.at_k}: {mean_ndcg_20_2*100:.4f}"
            f"Quora NDCG@10: {quora_ndcg}"
        )

        return mean_mrr
