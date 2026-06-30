# ============================================================
# generator.py - Step 4 of the RAG Pipeline
# ============================================================
# This file takes the retrieved chunks and the user's question,
# sends them to the Gemini API, and returns a grounded answer.
#
# "Grounded" means: Gemini must answer ONLY from the context
# we provide. It should not make things up (hallucinate).
#
# We tell Gemini this clearly in the prompt.
# ============================================================

import os
from google import genai
from dotenv import load_dotenv
from llm import generate_text

# Load variables from the .env file into the environment
load_dotenv()

# We read the API key from the environment variable
# NEVER put your real API key directly in the code!
# Instead, set it before running:
#   Windows: set GEMINI_API_KEY=your_key_here
#   Mac/Linux: export GEMINI_API_KEY=your_key_here
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

def setup_gemini():
    """
    Connect to the Gemini API using our key.
    """
    if not GEMINI_API_KEY:
        print("ERROR: GEMINI_API_KEY environment variable is not set!")
        print("Please set it before running:")
        print("  Windows: set GEMINI_API_KEY=your_key_here")
        print("  Mac/Linux: export GEMINI_API_KEY=your_key_here")
        return None

    client = genai.Client(api_key=GEMINI_API_KEY)
    return client


def build_prompt(question, retrieved_chunks):
    """
    Build the prompt we will send to Gemini.

    We give Gemini:
    1. The retrieved chunks as "context"
    2. The user's question
    3. Clear instructions to only use the context

    This is what prevents hallucination.
    """
    # Build the context string from all retrieved chunks
    context_parts = []

    for i, chunk in enumerate(retrieved_chunks, start=1):
        context_part = f"""
Source {i}: {chunk['title']} - {chunk['heading']}
URL: {chunk['source_url']}
---
{chunk['text']}
"""
        context_parts.append(context_part)

    context = "\n".join(context_parts)

    # Build the full prompt
    prompt = f"""You are a helpful data science tutor.
Answer the student's question using ONLY the information in the context below.
If the answer is not in the context, say: "I don't have enough information to answer that."
Do not make up any facts that are not in the context.

=== CONTEXT ===
{context}

=== STUDENT QUESTION ===
{question}

=== YOUR ANSWER ===
"""

    return prompt


def generate_answer(question, retrieved_chunks, client):
    """
    Send the question and context to Gemini.
    Returns the answer as a string.
    """
    # Build the prompt
    prompt = build_prompt(question, retrieved_chunks)

    # Send to Gemini (with automatic retry / fallback on rate limits)
    answer = generate_text(client, prompt)

    return answer


def run_generator(question, retrieved_chunks):
    """
    Main function - set up Gemini and generate an answer.
    Prints the answer and the sources used.
    """
    print("--- Step 4: Generating Answer with Gemini ---")

    # Set up the Gemini client
    client = setup_gemini()
    if client is None:
        return None

    # Generate the answer
    print("Sending to Gemini...")
    answer = generate_answer(question, retrieved_chunks, client)

    # Print the answer
    print("\n=== ANSWER ===")
    print(answer)

    # Print the sources used
    print("\n=== SOURCES USED ===")
    for i, chunk in enumerate(retrieved_chunks, start=1):
        print(f"  {i}. {chunk['title']} - {chunk['heading']}")
        print(f"     {chunk['source_url']}")

    return answer


# Run if called directly (for testing)
if __name__ == "__main__":
    # This is just a test - in the real pipeline
    # the retrieved_chunks come from retriever.py

    fake_chunks = [
        {
            "title":      "Introduction to Machine Learning",
            "heading":    "What is Machine Learning?",
            "text":       "Machine learning is a type of AI where computers learn from data instead of being explicitly programmed. There are three main types: supervised, unsupervised, and reinforcement learning.",
            "source_url": "https://example.com",
            "score":      0.15
        }
    ]

    question = "What are the types of machine learning?"
    run_generator(question, fake_chunks)