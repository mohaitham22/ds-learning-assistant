# ============================================================
# hybrid_search.py - Level 3
# ============================================================
# This file improves retrieval by combining two search methods:
#
#   1. Dense Search (FAISS)
#      Uses embeddings (vectors) to find chunks that are
#      SEMANTICALLY similar to the question.
#      Good at: understanding meaning and context
#      Bad at:  finding exact keywords or technical terms
#
#      Example: "how to prevent model from memorizing training data"
#      → FAISS understands this means "overfitting" even without
#        the word overfitting appearing in the question
#
#   2. Sparse Search (BM25)
#      Uses keyword matching to find chunks that contain
#      the same words as the question.
#      Good at: exact technical terms, function names, acronyms
#      Bad at:  understanding meaning or synonyms
#
#      Example: "BM25 algorithm"
#      → BM25 finds chunks containing "BM25" exactly
#        FAISS might miss this if no chunk is semantically close
#
#   3. Reciprocal Rank Fusion (RRF)
#      Combines the two ranked lists into one final ranking.
#      Formula: RRF_score = 1 / (rank + 60)
#      We add up the RRF score from each method for each chunk.
#      The chunk with the highest total score wins.
#
#      Why 60? It is a constant that reduces the influence of
#      very high ranks and works well in practice.
# ============================================================

from rank_bm25 import BM25Okapi
import numpy as np


# ---- Build the BM25 index ----------------------------------

def build_bm25_index(chunks):
    """
    Build a BM25 index from the chunks.

    BM25 needs the text split into individual words (tokens).
    We lowercase everything so "Machine" and "machine" match.

    Returns the BM25 object and the tokenized corpus.
    """
    print("Building BM25 index...")

    # Tokenize each chunk: split text into a list of words
    tokenized_corpus = []
    for chunk in chunks:
        words = chunk["text"].lower().split()
        tokenized_corpus.append(words)

    # Build the BM25 index from all tokenized chunks
    bm25 = BM25Okapi(tokenized_corpus)

    print(f"BM25 index built with {len(tokenized_corpus)} documents")
    return bm25, tokenized_corpus


# ---- BM25 search -------------------------------------------

def bm25_search(question, bm25, chunks, top_k=10):
    """
    Search for the top_k most relevant chunks using BM25.

    BM25 scores are based on term frequency and document length.
    Higher score = more relevant.

    Returns a list of (chunk_index, score) tuples.
    """
    # Tokenize the question the same way we tokenized the chunks
    question_tokens = question.lower().split()

    # Get BM25 scores for all chunks
    scores = bm25.get_scores(question_tokens)

    # Get the indices of the top_k highest scores
    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        results.append({
            "chunk_index": int(idx),
            "chunk":       chunks[idx],
            "bm25_score":  float(scores[idx])
        })

    return results


# ---- FAISS search ------------------------------------------

def faiss_search(question, index, model, chunks, top_k=10):
    """
    Search for the top_k most relevant chunks using FAISS.

    We embed the question and find the nearest vectors.
    Lower distance = more relevant (opposite of BM25).

    Returns a list of (chunk_index, score) tuples.
    """
    # Embed the question
    question_vector = model.encode([question], convert_to_numpy=True)

    # Search FAISS
    distances, indices = index.search(question_vector, top_k)

    results = []
    for i in range(top_k):
        chunk_idx = int(indices[0][i])
        distance  = float(distances[0][i])

        # FAISS pads with -1 when fewer than top_k vectors exist; skip those
        # and any index that falls outside the provided chunk list.
        if chunk_idx < 0 or chunk_idx >= len(chunks):
            continue

        results.append({
            "chunk_index":   chunk_idx,
            "chunk":         chunks[chunk_idx],
            "faiss_distance": distance
        })

    return results


# ---- Reciprocal Rank Fusion --------------------------------

def reciprocal_rank_fusion(faiss_results, bm25_results, top_k=5):
    """
    Combine FAISS and BM25 results using Reciprocal Rank Fusion.

    How RRF works:
      - Each chunk gets a score from each method
      - Score = 1 / (rank + 60)    (rank starts at 1)
      - Final score = faiss_rrf_score + bm25_rrf_score
      - Sort by final score descending

    The constant 60 is standard in RRF literature.
    It prevents very high-ranked results from dominating too much.

    Example with 3 chunks:
      Chunk A: FAISS rank 1 → 1/(1+60) = 0.0164
               BM25  rank 3 → 1/(3+60) = 0.0159
               Total = 0.0323

      Chunk B: FAISS rank 2 → 1/(2+60) = 0.0161
               BM25  rank 1 → 1/(1+60) = 0.0164
               Total = 0.0325  ← wins!

      Chunk C: FAISS rank 1 → 0.0164
               BM25  rank 0 → not found = 0
               Total = 0.0164
    """
    RRF_CONSTANT = 60

    # Dictionary to accumulate RRF scores for each chunk
    # Key: chunk_index, Value: running total score
    rrf_scores = {}

    # Also store chunk data so we can return it
    chunk_data = {}

    # Process FAISS results
    for rank, result in enumerate(faiss_results, start=1):
        # Key by the chunk's stable id, not its position in a list. FAISS and
        # BM25 may pass different list positions for the same chunk (e.g. when
        # BM25 runs on a filtered subset), so positions can't be used to merge.
        chunk_key = result["chunk"].get("chunk_id", result["chunk_index"])
        rrf_score = 1.0 / (rank + RRF_CONSTANT)

        if chunk_key not in rrf_scores:
            rrf_scores[chunk_key] = 0.0
            chunk_data[chunk_key] = result["chunk"]

        rrf_scores[chunk_key] += rrf_score

    # Process BM25 results
    for rank, result in enumerate(bm25_results, start=1):
        chunk_key = result["chunk"].get("chunk_id", result["chunk_index"])
        rrf_score = 1.0 / (rank + RRF_CONSTANT)

        if chunk_key not in rrf_scores:
            rrf_scores[chunk_key] = 0.0
            chunk_data[chunk_key] = result["chunk"]

        rrf_scores[chunk_key] += rrf_score

    # Sort all chunks by their final RRF score (highest first)
    sorted_chunks = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    # Build the final results list
    final_results = []
    for chunk_idx, score in sorted_chunks[:top_k]:
        chunk = chunk_data[chunk_idx]
        final_results.append({
            "chunk_id":   chunk.get("chunk_id", chunk_idx),
            "text":       chunk["text"],
            "title":      chunk["title"],
            "heading":    chunk["heading"],
            "tags":       chunk["tags"],
            "source_url": chunk["source_url"],
            "score":      round(score, 6)   # RRF score (higher = better)
        })

    return final_results


# ---- Main hybrid search function ---------------------------

def hybrid_search(question, chunks, index, model, bm25, top_k=5,
                  faiss_chunks=None, allowed_ids=None):
    """
    Run hybrid search: BM25 + FAISS + RRF.

    This is the function called by the main pipeline.
    Returns the top_k most relevant chunks.

    Notes on the chunk lists:
      - `bm25` (and the `chunks` used to map its hits) may be a *filtered*
        subset built by the caller for Level-2 filtering.
      - FAISS always searches the global `index`, so it must be paired with the
        *full* chunk list. Pass that as `faiss_chunks`; it defaults to `chunks`
        for the unfiltered case.
      - When `allowed_ids` is given, FAISS hits outside that set are dropped so
        the Level-2 filter applies to dense results too.
    """
    # We retrieve more than top_k from each method
    # so RRF has a bigger pool to work with
    retrieval_k = top_k * 3

    # FAISS needs the chunk list aligned with the global index
    faiss_chunks = faiss_chunks if faiss_chunks is not None else chunks

    # When filtering, pull a larger FAISS pool so enough allowed chunks survive
    faiss_k = retrieval_k
    if allowed_ids is not None:
        faiss_k = min(len(faiss_chunks), max(retrieval_k, 100))

    # Run both searches
    faiss_results = faiss_search(question, index, model, faiss_chunks, top_k=faiss_k)
    bm25_results  = bm25_search(question, bm25, chunks, top_k=retrieval_k)

    # Restrict dense results to the Level-2 filtered set
    if allowed_ids is not None:
        faiss_results = [
            r for r in faiss_results
            if r["chunk"].get("chunk_id") in allowed_ids
        ][:retrieval_k]

    # Combine with RRF
    final_results = reciprocal_rank_fusion(faiss_results, bm25_results, top_k=top_k)

    return final_results


# ---- Comparison function (for report) ----------------------

def compare_dense_vs_hybrid(question, chunks, index, model, bm25, top_k=5):
    """
    Run BOTH dense-only and hybrid search on the same question.
    Print a side-by-side comparison.

    This is required by the project for the report.
    """
    print("=" * 60)
    print(f"COMPARISON: Dense-only vs Hybrid Search")
    print(f"Question: {question}")
    print("=" * 60)

    retrieval_k = top_k * 3

    # --- Dense only (FAISS only) ---
    faiss_results = faiss_search(question, index, model, chunks, top_k=retrieval_k)
    dense_results = []
    for result in faiss_results[:top_k]:
        chunk = result["chunk"]
        dense_results.append({
            "chunk_id":   chunk.get("chunk_id"),
            "text":       chunk["text"],
            "title":      chunk["title"],
            "heading":    chunk["heading"],
            "tags":       chunk["tags"],
            "source_url": chunk["source_url"],
            "score":      round(result["faiss_distance"], 4)
        })

    # --- Hybrid (BM25 + FAISS + RRF) ---
    bm25_results   = bm25_search(question, bm25, chunks, top_k=retrieval_k)
    hybrid_results = reciprocal_rank_fusion(faiss_results, bm25_results, top_k=top_k)

    # --- Print comparison ---
    print("\n--- DENSE ONLY (FAISS) ---")
    for i, r in enumerate(dense_results, 1):
        print(f"  [{i}] {r['title']} - {r['heading']}")
        print(f"       Score (distance): {r['score']}  (lower = better)")

    print("\n--- HYBRID (FAISS + BM25 + RRF) ---")
    for i, r in enumerate(hybrid_results, 1):
        print(f"  [{i}] {r['title']} - {r['heading']}")
        print(f"       RRF Score: {r['score']}  (higher = better)")

    # --- Check overlap ---
    dense_titles  = [r["heading"] for r in dense_results]
    hybrid_titles = [r["heading"] for r in hybrid_results]
    new_in_hybrid = [t for t in hybrid_titles if t not in dense_titles]

    print(f"\n--- Summary ---")
    print(f"  Chunks in dense only : {len(dense_results)}")
    print(f"  Chunks in hybrid     : {len(hybrid_results)}")
    print(f"  New results in hybrid: {len(new_in_hybrid)}")
    if new_in_hybrid:
        print(f"  New headings found   : {new_in_hybrid}")

    return dense_results, hybrid_results


# ---- Test this file directly --------------------------------

if __name__ == "__main__":
    import sys
    import json
    import faiss
    from sentence_transformers import SentenceTransformer

    # On Windows the console defaults to cp1252, which crashes when we
    # print emoji/special characters that appear in some chunks. Force
    # UTF-8 output so printing any chunk text is safe.
    sys.stdout.reconfigure(encoding="utf-8")

    CHUNKS_PATH     = "data/chunks.json"
    INDEX_PATH      = "data/faiss_index.bin"
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"

    print("Loading chunks and index...")
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    index = faiss.read_index(INDEX_PATH)
    model = SentenceTransformer(EMBEDDING_MODEL)
    bm25, _ = build_bm25_index(chunks)

    # Test questions for the report comparison
    test_questions = [
        "What is overfitting and how to prevent it?",
        "BM25 algorithm for text retrieval",
        "how does backpropagation update weights in neural networks",
    ]

    for question in test_questions:
        print("\n")
        compare_dense_vs_hybrid(question, chunks, index, model, bm25, top_k=5)
        print()