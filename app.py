# ============================================================
# app.py - Streamlit GUI
# ============================================================
# This is the user interface for the DS Learning Assistant.
#
# It connects to our RAG pipeline (chunker, embedder, retriever,
# query_intelligence, hybrid_search, generator) and shows:
#   - A text box to ask a question
#   - The final answer from Gemini
#   - The retrieved chunks with scores and metadata
#   - Original query, rewritten query, query type, filters
#
# Run with:
#   streamlit run app.py
#
# Note (per project rules):
# This GUI file was built with AI assistance, which is allowed
# for the Streamlit interface only. All RAG/LLM pipeline logic
# (chunker.py, embedder.py, retriever.py, generator.py,
# query_intelligence.py, hybrid_search.py) was written by hand.
# ============================================================

import os
import json
import streamlit as st
import faiss
from sentence_transformers import SentenceTransformer
from google import genai

# Import our own pipeline functions
from chunker            import run_chunker
from embedder           import run_embedder
from generator          import generate_answer
from query_intelligence import run_query_intelligence
from hybrid_search      import build_bm25_index, hybrid_search
from llm                import generate_text

# --- Settings ---
CHUNKS_PATH     = "data/chunks.json"
INDEX_PATH      = "data/faiss_index.bin"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
TOP_K           = 5


# ---- Page setup ----------------------------------------------

st.set_page_config(
    page_title="DS Learning Assistant",
    page_icon="📊",
    layout="wide"
)


# ---- Load everything once and cache it ------------------------
# @st.cache_resource means this only runs ONCE, not on every click.
# This is important because loading the model takes time.

@st.cache_resource
def load_pipeline():
    """
    Build chunks/index if needed, then load everything.
    Cached so it only runs once per app session.
    """
    # Build chunks if missing
    if not os.path.exists(CHUNKS_PATH):
        run_chunker()

    # Build FAISS index if missing
    if not os.path.exists(INDEX_PATH):
        run_embedder()

    # Load chunks
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    # Load FAISS index
    index = faiss.read_index(INDEX_PATH)

    # Load embedding model
    model = SentenceTransformer(EMBEDDING_MODEL)

    # Build BM25 index
    bm25, _ = build_bm25_index(chunks)

    return chunks, index, model, bm25


def get_gemini_client():
    """
    Set up the Gemini client using the API key.

    Looks for the key in two places so it works both locally and on
    Streamlit Community Cloud:
      1. st.secrets["GEMINI_API_KEY"]  (set in the Cloud dashboard)
      2. the GEMINI_API_KEY environment variable / local .env
    """
    api_key = ""

    # st.secrets raises if there is no secrets file at all (common
    # locally), so guard the lookup.
    try:
        api_key = st.secrets.get("GEMINI_API_KEY", "")
    except Exception:
        api_key = ""

    if not api_key:
        api_key = os.environ.get("GEMINI_API_KEY", "")

    if not api_key:
        return None

    client = genai.Client(api_key=api_key)
    return client


# ---- Agentic fallback (GUI-side orchestration) -----------------
# When the RAG pipeline (grounded in our scraped tutorials) cannot
# answer from the retrieved context, the generator returns the
# canonical phrase "I don't have enough information to answer that."
#
# Instead of dead-ending there, we run a SECOND agent: ask the LLM
# to answer from its own general knowledge, and clearly label the
# answer so the user knows it's NOT grounded in the scraped sources.

NO_INFO_MARKERS = (
    "don't have enough information",
    "do not have enough information",
)


def is_insufficient_answer(answer):
    """
    True if the grounded RAG answer signals that the retrieved
    context did not actually contain the answer.
    """
    if not answer or not answer.strip():
        return True
    text = answer.strip().lower()
    return any(marker in text for marker in NO_INFO_MARKERS)


def general_ai_answer(question, client):
    """
    Fallback agent: answer the question using the LLM's own general
    knowledge (no scraped context). Used only when the knowledge
    base could not answer.
    """
    prompt = f"""You are a helpful, knowledgeable data science tutor.

Our curated tutorial knowledge base did not contain a relevant answer to the
student's question, so answer using your own general knowledge instead.

Give a clear, accurate, beginner-friendly explanation. If the question is not
about data science, machine learning, statistics, or programming, say so briefly.

=== STUDENT QUESTION ===
{question}

=== YOUR ANSWER ===
"""
    return generate_text(client, prompt)


# ---- Main app --------------------------------------------------

def main():

    # --- Header ---
    st.title("📊 DS Learning Assistant")
    st.caption("Ask any data science question. Answers are grounded in real tutorials using RAG.")

    # --- Check for Gemini API key ---
    client = get_gemini_client()
    if client is None:
        st.error(
            "GEMINI_API_KEY is not set. Please set it before running this app:\n\n"
            "Windows: `set GEMINI_API_KEY=your_key_here`\n\n"
            "Mac/Linux: `export GEMINI_API_KEY=your_key_here`"
        )
        st.stop()

    # --- Load pipeline (cached, runs once) ---
    with st.spinner("Loading data and models... (first run may take a minute)"):
        chunks, index, model, bm25 = load_pipeline()

    st.success(f"Ready! {len(chunks)} knowledge chunks loaded.")

    st.divider()

    # --- Question input ---
    question = st.text_input(
        "Your question:",
        placeholder="e.g. What is the difference between supervised and unsupervised learning?"
    )

    ask_button = st.button("Ask", type="primary")

    # --- When the button is clicked ---
    if ask_button and question.strip():

        # ---- Level 2: Query Intelligence ----
        with st.spinner("Analyzing your question..."):
            analysis, filtered_chunks = run_query_intelligence(
                question, chunks, client
            )

        rewritten_query = analysis["rewritten_query"]
        query_type      = analysis["query_type"]
        filters         = analysis["filters"]

        # --- Show query intelligence info ---
        st.subheader("🔍 Query Analysis")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown(f"**Original query:**\n\n{question}")
            st.markdown(f"**Rewritten query:**\n\n{rewritten_query}")

        with col2:
            st.markdown(f"**Query type:** `{query_type}`")
            st.markdown(f"**Extracted filters:**")
            st.json(filters)

        # --- Handle out-of-scope questions ---
        if query_type == "out_of_scope":
            st.warning(
                "This question doesn't seem related to data science. "
                "Try asking about machine learning, statistics, Python, NLP, etc."
            )
            return

        st.divider()

        # ---- Level 3: Hybrid Search ----
        with st.spinner("Searching the knowledge base (hybrid search)..."):
            filtered_bm25, _ = build_bm25_index(filtered_chunks)

            results = hybrid_search(
                question = rewritten_query,
                chunks   = filtered_chunks,
                index    = index,
                model    = model,
                bm25     = filtered_bm25,
                top_k    = TOP_K
            )

        # ---- Level 1: Generate answer (grounded in scraped sources) ----
        with st.spinner("Generating answer with Gemini..."):
            answer = generate_answer(rewritten_query, results, client)

        # ---- Agentic fallback: if the knowledge base couldn't answer,
        #      ask the LLM from its own general knowledge instead. ----
        answered_from_kb = bool(results) and not is_insufficient_answer(answer)

        if not answered_from_kb:
            with st.spinner("Not in the knowledge base — asking the AI directly..."):
                answer = general_ai_answer(rewritten_query, client)

        # --- Show the final answer ---
        st.subheader("💡 Answer")
        if answered_from_kb:
            st.success("✅ Answered from the scraped knowledge base (RAG).")
        else:
            st.info("🤖 Not found in the scraped tutorials — answered from the AI's general knowledge.")
        st.markdown(answer)

        st.divider()

        # --- Show retrieved chunks ---
        st.subheader(f"📚 Retrieved Sources (top {TOP_K})")

        if not answered_from_kb:
            st.caption(
                "These are the closest matches from the knowledge base, but they "
                "didn't cover the question — the answer above came from the AI's "
                "general knowledge instead."
            )

        for i, result in enumerate(results, start=1):
            with st.expander(f"[{i}] {result['title']} — {result['heading']}  (score: {result['score']})"):
                st.markdown(f"**Source URL:** {result['source_url']}")
                st.markdown(f"**Tags:** {', '.join(result['tags'])}")
                st.markdown(f"**RRF Score:** {result['score']}  *(higher = more relevant)*")
                st.markdown("**Chunk text:**")
                st.text(result["text"])

    elif ask_button and not question.strip():
        st.warning("Please type a question first.")


    # --- Sidebar with info ---
    with st.sidebar:
        st.header("About")
        st.markdown(
            "This is a RAG-based learning assistant for data science topics. "
            "It uses:\n\n"
            "- **FAISS** for dense (semantic) search\n"
            "- **BM25** for sparse (keyword) search\n"
            "- **Reciprocal Rank Fusion** to combine both\n"
            "- **Gemini API** for grounded answer generation\n"
        )
        st.divider()
        st.markdown("**Example questions to try:**")
        st.markdown("- What is overfitting?")
        st.markdown("- Difference between supervised and unsupervised learning")
        st.markdown("- How do I handle missing values in pandas?")
        st.markdown("- Give me an example of k-means clustering")


if __name__ == "__main__":
    main()