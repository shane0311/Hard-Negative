from __future__ import annotations
import argparse
import gzip
from pathlib import Path
from typing import Dict, List, Tuple, Iterable, Set
from collections import defaultdict

import bm25s
import ir_datasets
import Stemmer
import numpy as np
from tqdm import tqdm


def build_tokenizer():
    """Build tokenizer exactly matching Script 1."""
    stemmer = Stemmer.Stemmer("english")
    return lambda texts: bm25s.tokenize(texts, stopwords="en", stemmer=stemmer)


def load_corpus(dataset_id: str):
    """Load corpus from ir_datasets."""
    ds = ir_datasets.load(dataset_id)
    
    doc_ids: List[str] = []
    doc_texts: List[str] = []
    
    print(f"Loading corpus: {dataset_id}")
    for d in tqdm(ds.docs_iter(), total=ds.docs_count(), desc="Loading docs"):
        doc_ids.append(str(d.doc_id))
        
        # Handle different document field structures
        if hasattr(d, 'text') and d.text:
            text = str(d.text)
        elif hasattr(d, 'body') and d.body:
            text = str(d.body)
        elif hasattr(d, 'title') and d.title:
            text = str(d.title)
        else:
            # Try to combine available fields
            parts = []
            for attr in ['title', 'body', 'text']:
                if hasattr(d, attr):
                    val = getattr(d, attr)
                    if val and str(val).strip():
                        parts.append(str(val))
            text = " ".join(parts) if parts else ""
        
        doc_texts.append(text)
    
    print(f"Loaded {len(doc_ids):,} documents")
    
    return ds, doc_ids, doc_texts


def load_qrels(ds) -> Dict[str, Set[str]]:
    """Load qrels and organize by query_id."""
    qrels_by_query: Dict[str, Set[str]] = defaultdict(set)
    
    print("Loading qrels...")
    qrel_count = 0
    for qrel in tqdm(ds.qrels_iter(), desc="Loading qrels"):
        qrels_by_query[str(qrel.query_id)].add(str(qrel.doc_id))
        qrel_count += 1
    
    print(f"Loaded {qrel_count:,} qrels for {len(qrels_by_query):,} queries")
    return dict(qrels_by_query)


def build_global_index(doc_texts: List[str], tokenizer):
    """Build global BM25 index."""
    print("Building global BM25 index...")
    retriever = bm25s.BM25()
    
    # Tokenize and index all documents
    print("Tokenizing documents...")
    tokenized_docs = tokenizer(doc_texts)
    
    print("Building index...")
    retriever.index(tokenized_docs)
    
    print("Global index built successfully")
    return retriever


def simple_top_k_retrieval(retriever_global, query_text: str, tokenizer, k: int, debug: bool = False):
    """
    Simple top-k retrieval using global BM25 index.
    Returns (indices, scores) of top-k documents.
    """
    # Tokenize query
    q_tokens = tokenizer(query_text)
    
    if debug:
        print(f"Query tokens: {q_tokens}")
    
    if not q_tokens:
        # Empty query - return first k documents with score 0
        if debug:
            print("Empty query, returning first k documents")
        return list(range(k)), [0.0] * k
    
    # Get top-k documents using BM25
    try:
        doc_idxs, doc_scores = retriever_global.retrieve(q_tokens, k=k)
        
        if len(doc_idxs) == 0 or len(doc_idxs[0]) == 0:
            if debug:
                print("No documents retrieved, returning first k")
            return list(range(k)), [0.0] * k
        
        # Get the actual retrieved indices and scores
        candidates = [int(idx) for idx in doc_idxs[0]]
        scores = [float(score) for score in doc_scores[0]]
        
        if debug:
            print(f"Retrieved {len(candidates)} candidates")
            print(f"Score range: {min(scores):.4f} to {max(scores):.4f}")
            print(f"Non-zero scores: {sum(1 for s in scores if s > 0)}")
        
        # Pad to k if we got fewer results
        while len(candidates) < k:
            candidates.append(len(candidates))  # Add next sequential doc
            scores.append(0.0)  # With score 0
        
        # Return exactly k results
        return candidates[:k], scores[:k]
        
    except Exception as e:
        if debug:
            print(f"Error in retrieval: {e}")
        return list(range(k)), [0.0] * k


def add_missing_qrels(
    candidates: List[int],
    scores: List[float],
    qrel_doc_ids: Set[str],
    doc_ids: List[str],
    doc_id_to_idx: Dict[str, int],
    debug: bool = False
):
    """
    Add missing qrels to candidates with score 0.
    Returns updated (candidates, scores).
    """
    # Get doc_ids of current candidates
    candidate_doc_ids = set(doc_ids[idx] for idx in candidates)
    
    # Find missing qrels
    missing_qrels = qrel_doc_ids - candidate_doc_ids
    
    if debug and missing_qrels:
        print(f"Adding {len(missing_qrels)} missing qrels")
    
    # Add missing qrels with score 0
    for doc_id in missing_qrels:
        if doc_id in doc_id_to_idx:
            candidates.append(doc_id_to_idx[doc_id])
            scores.append(0.0)
    
    return candidates, scores


def get_results_with_global_scores(
    qid: str,
    query_text: str,
    candidate_indices: List[int],
    candidate_scores: List[float],
    doc_ids: List[str],
    doc_texts: List[str],
    debug: bool = False
):
    """
    Use the global BM25 scores directly as final scores (no redundant per-query BM25).
    """
    if debug:
        print(f"Using global BM25 scores for {len(candidate_indices)} candidates")
        non_zero = sum(1 for s in candidate_scores if s > 0)
        print(f"Global scores: {non_zero}/{len(candidate_scores)} non-zero")
        if candidate_scores:
            print(f"Score range: {min(candidate_scores):.4f} to {max(candidate_scores):.4f}")
    
    # Yield results with global BM25 scores
    for i, corpus_idx in enumerate(candidate_indices):
        passage_text = doc_texts[corpus_idx]
        score = candidate_scores[i]
        yield qid, doc_ids[corpus_idx], query_text, passage_text, score


def _clean_text(text: str) -> str:
    """Clean text for TSV output."""
    return text.replace("\t", " ").replace("\n", " ").replace("\r", " ")


def write_output(rows, output_path: Path):
    """Write results to gzipped TSV."""
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(output_path, "wt", encoding="utf8") as fout:
        for qid, pid, query, passage, score in rows:
            fout.write(f"{qid}\t{pid}\t{_clean_text(query)}\t{_clean_text(passage)}\t{score}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Script 2: BM25 retrieval using global corpus scores directly"
    )
    parser.add_argument("--dataset", default="beir/quora/dev",
                        help="ir_datasets dataset ID")
    parser.add_argument("--k", type=int, default=3000,
                        help="Number of candidates per query")
    parser.add_argument("--output", default="",
                        help="Output .tsv.gz file")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug output for first few queries")
    args = parser.parse_args()
    
    # Load data
    tokenizer = build_tokenizer()
    ds, doc_ids, doc_texts = load_corpus(args.dataset)
    
    # Build doc_id to index mapping
    doc_id_to_idx = {doc_id: idx for idx, doc_id in enumerate(doc_ids)}
    
    # Load qrels
    qrels_by_query = load_qrels(ds)
    
    # Build global index for retrieval
    global_retriever = build_global_index(doc_texts, tokenizer)
    
    # Process all queries
    total_queries = ds.queries_count()
    k = min(args.k, len(doc_ids))
    
    print(f"Processing {total_queries} queries with k={k}")
    
    # Track statistics
    total_rows = 0
    zero_scores = 0
    missing_qrels_added = 0
    
    def process_all_queries():
        nonlocal total_rows, zero_scores, missing_qrels_added
        
        queries = list(ds.queries_iter())
        for i, q in enumerate(tqdm(queries, desc="Processing queries")):
            qid = str(q.query_id)
            query_text = q.text
            
            debug_this = args.debug and i < 3  # Debug first 3 queries
            
            if debug_this:
                print(f"\n=== DEBUG QUERY {i+1}: {qid} ===")
                print(f"Query text: '{query_text}'")
            
            # Get top-k candidates + scores using global BM25
            candidates, scores = simple_top_k_retrieval(
                global_retriever, 
                query_text, 
                tokenizer, 
                k,
                debug=debug_this
            )
            
            # Add missing qrels with score 0
            qrel_doc_ids = qrels_by_query.get(qid, set())
            initial_count = len(candidates)
            candidates, scores = add_missing_qrels(
                candidates, scores, qrel_doc_ids, doc_ids, doc_id_to_idx, debug=debug_this
            )
            missing_qrels_added += (len(candidates) - initial_count)
            
            # Use global BM25 scores directly (no redundant per-query BM25)
            query_zeros = 0
            for row in get_results_with_global_scores(
                qid, query_text, candidates, scores, doc_ids, doc_texts, debug=debug_this
            ):
                if row[4] == 0.0:
                    zero_scores += 1
                    query_zeros += 1
                total_rows += 1
                yield row
            
            if debug_this:
                print(f"Query {i+1} results: {query_zeros}/{len(candidates)} zero scores")
            
            # Progress update every 100 queries
            if (i + 1) % 100 == 0:
                pct_zero = 100 * zero_scores / total_rows if total_rows > 0 else 0
                print(f"Progress: {i+1}/{total_queries} queries, {pct_zero:.1f}% zero scores so far")
    
    # Write output
    output_path = Path(args.output)
    write_output(process_all_queries(), output_path)
    
    # Final statistics
    pct_zero = 100 * zero_scores / total_rows if total_rows > 0 else 0
    print(f"\n=== FINAL RESULTS ===")
    print(f"Output written to: {output_path}")
    print(f"Total rows: {total_rows:,}")
    print(f"Zero scores: {zero_scores:,} ({pct_zero:.1f}%)")
    print(f"Missing qrels added: {missing_qrels_added:,}")


if __name__ == "__main__":
    main()