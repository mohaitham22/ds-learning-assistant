# ============================================================
# query_intelligence.py - Level 2
# ============================================================
# Before we search for chunks, we first process the user's
# question through 3 steps:
#
#   1. Query Rewriting
#      Make the question cleaner and more specific so the
#      embedding model can find better results.
#      Example:
#        Original : "whats overfitting lol"
#        Rewritten: "What is overfitting in machine learning
#                    and how can it be prevented?"
#
#   2. Query Classification
#      Decide what type of question this is.
#      Types we support:
#        - "definition"    → "What is X?"
#        - "comparison"    → "What is the difference between X and Y?"
#        - "how_to"        → "How do I do X?"
#        - "example"       → "Give me an example of X"
#        - "out_of_scope"  → question not about data science
#
#   3. Filter Extraction
#      Pull out any specific topics or difficulty level
#      mentioned in the question so we can filter chunks.
#      Example:
#        Question : "explain neural networks for beginners"
#        Filters  : { "topic": "neural-networks",
#                     "difficulty": "beginner" }
#
# We do all 3 in ONE Gemini call to save time and API quota.
# We ask Gemini to return a JSON object with all 3 results.
# ============================================================

import json
from google import genai
import os
from dotenv import load_dotenv
from llm import generate_text

# Load variables from the .env file into the environment
load_dotenv()

# The Gemini client (passed in from outside)
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

# All the tags that exist in our dataset
# Gemini will pick from this list when extracting filters
KNOWN_TAGS = [
    "machine-learning", "deep-learning", "neural-networks",
    "regression", "linear-regression", "logistic-regression",
    "classification", "clustering", "k-means", "unsupervised",
    "supervised", "nlp", "natural-language-processing",
    "sentiment", "time-series", "forecasting", "arima",
    "reinforcement-learning", "q-learning",
    "python", "pandas", "numpy", "scikit-learn", "matplotlib",
    "visualization", "statistics", "probability", "math",
    "sql", "databases", "data-cleaning", "preprocessing",
    "optimization", "training", "overfitting", "regularization",
    "data-science", "ethics", "datasets", "benchmark",
    "calculus", "linear-algebra"
]

# All valid query types
QUERY_TYPES = [
    "definition",
    "comparison",
    "how_to",
    "example",
    "out_of_scope"
]


# ---- Step 1+2+3 combined: ask Gemini in one call -----------

def analyze_query(question, client):
    """
    Send the question to Gemini and get back:
      - rewritten_query  : a cleaner version of the question
      - query_type       : one of the 5 types above
      - filters          : dict with topic and difficulty

    Returns a dict like:
    {
        "original_query":  "whats overfitting lol",
        "rewritten_query": "What is overfitting in machine learning?",
        "query_type":      "definition",
        "filters": {
            "topic":      "overfitting",
            "difficulty": "any"
        }
    }
    """

    # Build the prompt - we ask Gemini to return ONLY JSON
    prompt = f"""
You are a query analyzer for a data science learning assistant.

Analyze the user's question and return a JSON object with exactly these 4 fields:

1. "rewritten_query": Rewrite the question to be clear, specific, and in proper English.
   - Fix spelling mistakes
   - Make it a complete question
   - Add context if needed (e.g. "in machine learning")
   - Keep it focused on one topic

2. "query_type": Classify the question into exactly ONE of these types:
   - "definition"   → asking what something is (What is X?)
   - "comparison"   → asking about differences (X vs Y?)
   - "how_to"       → asking how to do something
   - "example"      → asking for an example or demo
   - "out_of_scope" → not related to data science at all

3. "filters": Extract filters as a JSON object with:
   - "topic": the most relevant tag from this list, or "any" if none match:
     {KNOWN_TAGS}
   - "difficulty": one of "beginner", "intermediate", "advanced", or "any"

Return ONLY the JSON object. No explanation. No markdown. No extra text.

User question: {question}

JSON:
"""

    raw_text = generate_text(client, prompt).strip()

    # Sometimes Gemini wraps the JSON in markdown code blocks
    # Remove them if present
    if raw_text.startswith("```"):
        lines = raw_text.splitlines()
        # Remove first line (```json) and last line (```)
        lines = [l for l in lines if not l.startswith("```")]
        raw_text = "\n".join(lines)

    # Parse the JSON
    try:
        result = json.loads(raw_text)
    except json.JSONDecodeError:
        # If Gemini returned something unexpected, use safe defaults
        print(f"  [WARN] Could not parse Gemini JSON response. Using defaults.")
        print(f"  Raw response was: {raw_text[:200]}")
        result = {
            "rewritten_query": question,
            "query_type":      "definition",
            "filters": {
                "topic":      "any",
                "difficulty": "any"
            }
        }

    # Always include the original question
    result["original_query"] = question

    return result


# ---- Filter chunks based on extracted filters --------------

def filter_chunks(chunks, filters):
    """
    Filter the list of chunks based on the extracted filters.

    If topic is "any" or not found, return all chunks.
    If topic is set, return only chunks that have that tag.

    This narrows the search space before FAISS retrieval.
    """
    topic = filters.get("topic", "any")

    # If no specific topic, return everything
    if topic == "any" or not topic:
        return chunks

    # Filter to chunks that have the topic tag
    filtered = [
        chunk for chunk in chunks
        if topic in chunk.get("tags", [])
    ]

    # If the filter is too strict and returns nothing,
    # fall back to all chunks
    if len(filtered) == 0:
        print(f"  [INFO] No chunks found for topic '{topic}'. Using all chunks.")
        return chunks

    print(f"  [INFO] Filtered to {len(filtered)} chunks with tag '{topic}'")
    return filtered


# ---- Main function: run full query intelligence ------------

def run_query_intelligence(question, chunks, client):
    """
    Full Level 2 pipeline for one question.

    Returns:
      - analysis     : dict with rewritten query, type, filters
      - filtered_chunks: the narrowed list of chunks to search
    """
    print("--- Level 2: Query Intelligence ---")

    # Step 1+2+3: Analyze the query with Gemini
    print(f"  Original question : {question}")
    analysis = analyze_query(question, client)

    print(f"  Rewritten query   : {analysis['rewritten_query']}")
    print(f"  Query type        : {analysis['query_type']}")
    print(f"  Filters           : {analysis['filters']}")

    # Step 4: Filter chunks using the extracted filters
    filtered_chunks = filter_chunks(chunks, analysis["filters"])

    return analysis, filtered_chunks


# ---- Test this file directly --------------------------------

if __name__ == "__main__":

    # Set up Gemini client
    if not GEMINI_API_KEY:
        print("Please set GEMINI_API_KEY environment variable.")
        exit()

    client = genai.Client(api_key=GEMINI_API_KEY)

    # Load chunks
    import json
    with open("data/chunks.json", "r", encoding="utf-8") as f:
        chunks = json.load(f)

    # Test with different types of questions
    test_questions = [
        "whats overfitting lol",
        "difference between random forest and decision tree",
        "how do i normalize data in pandas",
        "give me an example of k-means clustering",
        "what is the best pizza in cairo",
    ]

    for question in test_questions:
        print("\n" + "=" * 55)
        analysis, filtered = run_query_intelligence(question, chunks, client)
        print(f"  Chunks before filter : {len(chunks)}")
        print(f"  Chunks after filter  : {len(filtered)}")
        print()