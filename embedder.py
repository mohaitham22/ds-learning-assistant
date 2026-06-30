# ============================================================
# embedder.py - Step 2 of the RAG Pipeline
# ============================================================
# This file takes the chunks from chunker.py and:
#   1. Converts each chunk into a vector (embedding)
#   2. Stores all vectors in a FAISS index for fast search
#
# What is an embedding?
# An embedding is a list of numbers that represents the meaning
# of a piece of text. Texts with similar meaning will have
# similar numbers. This lets us do semantic search.
#
# What is FAISS?
# FAISS is a library that stores vectors and finds the
# most similar ones very quickly.
#
# Model we use: all-MiniLM-L6-v2
# Why? It is small, fast, free, and works great for English.
# It produces 384-dimensional vectors.
# ============================================================

import json
import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# Paths
CHUNKS_PATH = "data/chunks.json"
INDEX_PATH  = "data/faiss_index.bin"

# The embedding model we chose
# all-MiniLM-L6-v2 is a great balance of speed and quality
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def load_chunks():
    """
    Load the chunks we made in chunker.py
    """
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    print(f"Loaded {len(chunks)} chunks from {CHUNKS_PATH}")
    return chunks


def embed_chunks(chunks):
    """
    Convert each chunk's text into a vector using the embedding model.

    Returns a numpy array of shape (num_chunks, 384)
    Each row is the embedding for one chunk.
    """
    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)

    # Get the text from each chunk
    texts = [chunk["text"] for chunk in chunks]

    print(f"Embedding {len(texts)} chunks... (this may take a minute)")

    # Convert all texts to vectors at once
    # show_progress_bar=True prints a nice progress bar
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        convert_to_numpy=True
    )

    print(f"Embeddings shape: {embeddings.shape}")
    # Should print something like: (674, 384)
    # 674 chunks, each with 384 numbers

    return embeddings


def build_faiss_index(embeddings):
    """
    Store all the embeddings in a FAISS index.

    We use IndexFlatL2 which finds the nearest vectors
    using L2 (Euclidean) distance. It is simple and works well.
    """
    # How many numbers per vector
    dimension = embeddings.shape[1]

    print(f"Building FAISS index with dimension={dimension}")

    # Create the index
    index = faiss.IndexFlatL2(dimension)

    # Add all embeddings to the index
    index.add(embeddings)

    print(f"FAISS index contains {index.ntotal} vectors")

    return index


def save_index(index):
    """
    Save the FAISS index to disk so we can load it later
    without re-embedding everything.
    """
    os.makedirs("ds_rag/data", exist_ok=True)
    faiss.write_index(index, INDEX_PATH)
    print(f"FAISS index saved to {INDEX_PATH}")


def run_embedder():
    """
    Main function - load chunks, embed them, save the index.
    """
    print("--- Step 2: Embedding chunks ---")

    chunks     = load_chunks()
    embeddings = embed_chunks(chunks)
    index      = build_faiss_index(embeddings)
    save_index(index)

    print()
    return embeddings


# Run if called directly
if __name__ == "__main__":
    run_embedder()