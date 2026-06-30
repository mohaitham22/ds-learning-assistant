# ============================================================
# rag_pipeline.py - Level 1 + Level 2 + Level 3
# ============================================================
# Full pipeline:
#
#   Question
#      ↓
#   [Level 2] Query Rewriting + Classification + Filter Extraction
#      ↓
#   [Level 3] Hybrid Search: BM25 + FAISS + RRF
#      ↓
#   [Level 1] Gemini Answer Generation
#
# Usage:
#   python rag_pipeline.py
#
# Set your Gemini API key first:
#   Windows  : set GEMINI_API_KEY=your_key_here
#   Mac/Linux: export GEMINI_API_KEY=your_key_here
# ============================================================

import os
import json
import faiss
from sentence_transformers import SentenceTransformer
from google import genai

# Import our pipeline modules
from chunker            import run_chunker
from embedder           import run_embedder
from generator          import generate_answer
from query_intelligence import run_query_intelligence, filter_chunks
from hybrid_search      import build_bm25_index, hybrid_search, compare_dense_vs_hybrid

# --- Settings ---
CHUNKS_PATH     = "data/chunks.json"
INDEX_PATH      = "data/faiss_index.bin"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
TOP_K           = 5


# ---- Setup -------------------------------------------------

def setup_pipeline():
    """
    Build chunks and FAISS index if needed, then load everything.
    """
    # Build chunks if missing
    if not os.path.exists(CHUNKS_PATH):
        print("Chunks not found. Running chunker...")
        run_chunker()
    else:
        print("Chunks already exist - skipping chunking.")

    # Build FAISS index if missing
    if not os.path.exists(INDEX_PATH):
        print("FAISS index not found. Running embedder...")
        run_embedder()
    else:
        print("FAISS index already exists - skipping embedding.")

    # Load chunks
    print("\nLoading chunks...")
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    # Load FAISS index
    print("Loading FAISS index...")
    index = faiss.read_index(INDEX_PATH)

    # Load embedding model
    print(f"Loading embedding model ({EMBEDDING_MODEL})...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    # Build BM25 index (fast, no disk save needed)
    bm25, _ = build_bm25_index(chunks)

    # Set up Gemini
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        print("\nERROR: GEMINI_API_KEY is not set!")
        print("  Windows  : set GEMINI_API_KEY=your_key_here")
        print("  Mac/Linux: export GEMINI_API_KEY=your_key_here")
        exit()

    client = genai.Client(api_key=api_key)

    print(f"\nReady! {len(chunks)} chunks loaded.")
    print()

    return chunks, index, model, bm25, client


# ---- Full pipeline for one question ------------------------

def ask_question(question, chunks, index, model, bm25, client):
    """
    Run the complete pipeline for one question.

    Level 2 → Level 3 → Level 1 → Answer
    """
    print("\n" + "=" * 55)
    print(f"QUESTION: {question}")
    print("=" * 55)

    # ---- Level 2: Query Intelligence ----
    analysis, filtered_chunks = run_query_intelligence(
        question, chunks, client
    )

    rewritten_query = analysis["rewritten_query"]
    query_type      = analysis["query_type"]
    filters         = analysis["filters"]

    # Stop immediately if question is not about data science
    if query_type == "out_of_scope":
        print("\nThis question is not related to data science.")
        print("Please ask about ML, Python, statistics, NLP, etc.")
        return

    # ---- Level 3: Hybrid Search ----
    print(f"\nRunning hybrid search on {len(filtered_chunks)} filtered chunks...")

    # Build a BM25 index just for the filtered chunks
    # (so the filter from Level 2 also applies to BM25)
    from hybrid_search import build_bm25_index as build_bm25
    filtered_bm25, _ = build_bm25(filtered_chunks)

    # Chunk ids allowed through the Level-2 filter (also used to restrict FAISS,
    # which searches the global index over the full chunk list).
    allowed_ids = {c.get("chunk_id") for c in filtered_chunks}

    results = hybrid_search(
        question      = rewritten_query,
        chunks        = filtered_chunks,
        index         = index,
        model         = model,
        bm25          = filtered_bm25,
        top_k         = TOP_K,
        faiss_chunks  = chunks,
        allowed_ids   = allowed_ids
    )

    # Show retrieved chunks
    print("\n--- Retrieved Chunks (Hybrid Search) ---")
    for i, result in enumerate(results, start=1):
        print(f"  [{i}] {result['title']} - {result['heading']}")
        print(f"       RRF Score: {result['score']}   Tags: {result['tags']}")
        print(f"       Preview: {result['text'][:100]}...")

    # ---- Level 1: Answer Generation ----
    print("\n--- Sending to Gemini ---")
    answer = generate_answer(rewritten_query, results, client)

    # ---- Final Output ----
    print("\n" + "=" * 55)
    print("ANSWER:")
    print("=" * 55)
    print(answer)

    print("\n--- Query Details ---")
    print(f"  Original query  : {question}")
    print(f"  Rewritten query : {rewritten_query}")
    print(f"  Query type      : {query_type}")
    print(f"  Filters         : {filters}")

    print("\n--- Sources ---")
    for i, result in enumerate(results, start=1):
        print(f"  {i}. {result['title']}")
        print(f"     {result['source_url']}")

    print()
    return answer, results, analysis


# ---- Main --------------------------------------------------

def main():
    print("=" * 55)
    print("  DS Learning Assistant - Full RAG Pipeline")
    print("  Level 1 + Level 2 + Level 3")
    print("=" * 55)
    print()

    # Setup everything
    chunks, index, model, bm25, client = setup_pipeline()

    # --- Optional: run the comparison for your report ---
    # Uncomment this block to generate the dense vs hybrid comparison
    # required by the PDF spec (Section 3.3)
    #
    # print("\n=== REPORT: Dense vs Hybrid Comparison ===")
    # comparison_questions = [
    #     "What is overfitting and how to prevent it?",
    #     "how does backpropagation update weights",
    #     "difference between supervised and unsupervised learning",
    # ]
    # for q in comparison_questions:
    #     compare_dense_vs_hybrid(q, chunks, index, model, bm25, top_k=5)
    #     print()

    # --- Test questions ---
    test_questions = [
        "whats overfitting lol",
        "difference between random forest and decision tree",
        "how do i handle missing values in pandas",
        "give me an example of k-means clustering",
        "what is the best pizza in cairo",
    ]

    for question in test_questions:
        ask_question(question, chunks, index, model, bm25, client)
        print("-" * 55)

    # --- Interactive mode ---
    print("\n=== Interactive Mode ===")
    print("Type your question and press Enter. Type 'quit' to exit.\n")

    while True:
        question = input("Your question: ").strip()

        if question.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        if not question:
            continue

        ask_question(question, chunks, index, model, bm25, client)


if __name__ == "__main__":
    main()