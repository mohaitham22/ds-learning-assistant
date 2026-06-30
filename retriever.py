# ============================================================
# retriever.py - Step 3 of the RAG Pipeline
# ============================================================
# This file searches the FAISS index to find the chunks
# that are most relevant to the user's question.
#
# How retrieval works:
#   1. Take the user's question
#   2. Embed it using the SAME model we used for chunks
#   3. Ask FAISS: "which stored vectors are closest to this?"
#   4. Return the top-k most similar chunks
# ============================================================

import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# Paths
CHUNKS_PATH = "data/chunks.json"
INDEX_PATH  = "data/faiss_index.bin"

# Must use the SAME model that was used in embedder.py
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# How many chunks to retrieve
TOP_K = 5


def load_chunks():
    """
    Load chunks from disk.
    """
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    return chunks


def load_index():
    """
    Load the FAISS index from disk.
    """
    index = faiss.read_index(INDEX_PATH)
    return index


def retrieve(question, chunks, index, model, top_k=TOP_K, allowed_ids=None):
    """
    Find the top_k chunks most relevant to the question.

    IMPORTANT: `chunks` must be the FULL list of chunks (same order
    used to build the FAISS index), because FAISS returns positions
    into that full list.

    To restrict results to a subset (e.g. after filtering by topic),
    pass `allowed_ids` = a set of chunk_id values to keep. Any FAISS
    hit whose chunk_id is not in that set is skipped.

    Returns a list of result dicts, each with:
    - chunk_id
    - text
    - title
    - heading
    - tags
    - source_url
    - score (lower = more similar in L2 distance)
    """
    # Step 1: Embed the question into a vector
    question_vector = model.encode([question], convert_to_numpy=True)

    # Step 2: Search FAISS for the nearest vectors.
    # When we are filtering, we search more candidates (the whole
    # index) so that enough of them survive the filter to fill top_k.
    search_k = index.ntotal if allowed_ids is not None else top_k
    search_k = min(search_k, index.ntotal)
    distances, indices = index.search(question_vector, search_k)

    # Step 3: Build the list of results
    results = []

    for chunk_index, distance in zip(indices[0], distances[0]):
        # FAISS returns -1 when it has no more neighbours to give.
        if chunk_index < 0 or chunk_index >= len(chunks):
            continue

        chunk = chunks[chunk_index]

        # Skip chunks that were filtered out by Level 2.
        if allowed_ids is not None and chunk["chunk_id"] not in allowed_ids:
            continue

        result = {
            "chunk_id":   chunk["chunk_id"],
            "text":       chunk["text"],
            "title":      chunk["title"],
            "heading":    chunk["heading"],
            "tags":       chunk["tags"],
            "source_url": chunk["source_url"],
            "score":      round(float(distance), 4)
        }

        results.append(result)

        if len(results) >= top_k:
            break

    return results


def run_retriever():
    """
    Test the retriever with a sample question.
    Prints retrieved chunks and their scores.
    """
    print("--- Step 3: Testing Retriever ---")

    # Load everything
    chunks = load_chunks()
    index  = load_index()
    model  = SentenceTransformer(EMBEDDING_MODEL)

    # Test question
    question = "What is the difference between supervised and unsupervised learning?"

    print(f"\nQuestion: {question}")
    print(f"Retrieving top {TOP_K} chunks...\n")

    results = retrieve(question, chunks, index, model)

    # Print the results
    for i, result in enumerate(results, start=1):
        print(f"--- Result {i} ---")
        print(f"  Title   : {result['title']}")
        print(f"  Heading : {result['heading']}")
        print(f"  Tags    : {result['tags']}")
        print(f"  Score   : {result['score']}  (lower = more relevant)")
        print(f"  Preview : {result['text'][:150]}...")
        print()

    return results


# Run if called directly
if __name__ == "__main__":
    run_retriever()