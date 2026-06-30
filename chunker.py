# ============================================================
# chunker.py - Step 1 of the RAG Pipeline
# ============================================================
# This file reads articles.json and splits each article
# into small chunks that are easy to search and retrieve.
#
# Why do we chunk?
# Because the full article is too long to pass to an LLM.
# We want to find just the relevant paragraph, not the whole article.
#
# Our chunking strategy: one chunk = one section
# Each article is already split into sections by the scraper.
# Each section becomes one chunk. This is the natural unit
# of meaning in a structured tutorial article.
# ============================================================

import json
import os

# Where to read articles from
ARTICLES_PATH = "data/articles.json"

# Where to save the chunks
CHUNKS_PATH = "data/chunks.json"


def load_articles():
    """
    Read the articles.json file and return the list of articles.
    """
    with open(ARTICLES_PATH, "r", encoding="utf-8") as f:
        articles = json.load(f)

    print(f"Loaded {len(articles)} articles from {ARTICLES_PATH}")
    return articles


def make_chunks(articles):
    """
    Turn each article section into one chunk.

    Each chunk looks like this:
    {
        "chunk_id":    0,                      <- unique number
        "text":        "Linear regression...", <- the actual content
        "title":       "Linear Regression",   <- article title
        "heading":     "What is a slope?",    <- section heading
        "tags":        ["regression", ...],   <- topic tags
        "source_url":  "https://..."          <- where it came from
    }
    """
    all_chunks = []
    chunk_id = 0

    for article in articles:
        title      = article["title"]
        tags       = article["tags"]
        source_url = article["url"]
        sections   = article["sections"]

        for section in sections:
            heading = section["heading"]
            text    = section["text"]

            # Skip sections that are too short to be useful
            if len(text.split()) < 20:
                continue

            # Build the chunk text
            # We include the heading inside the text so the
            # embedding model understands what the section is about
            chunk_text = f"{title} - {heading}\n\n{text}"

            chunk = {
                "chunk_id":   chunk_id,
                "text":       chunk_text,
                "title":      title,
                "heading":    heading,
                "tags":       tags,
                "source_url": source_url
            }

            all_chunks.append(chunk)
            chunk_id += 1

    return all_chunks


def save_chunks(chunks):
    """
    Save all chunks to chunks.json
    """
    os.makedirs("ds_rag/data", exist_ok=True)

    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(chunks)} chunks to {CHUNKS_PATH}")


def run_chunker():
    """
    Main function - load articles, chunk them, save.
    """
    print("--- Step 1: Chunking articles ---")

    articles = load_articles()
    chunks   = make_chunks(articles)
    save_chunks(chunks)

    # Show some stats
    total_words = sum(len(c["text"].split()) for c in chunks)
    avg_words   = total_words // len(chunks) if chunks else 0

    print(f"Total chunks   : {len(chunks)}")
    print(f"Average length : {avg_words} words per chunk")
    print(f"Shortest chunk : {min(len(c['text'].split()) for c in chunks)} words")
    print(f"Longest chunk  : {max(len(c['text'].split()) for c in chunks)} words")
    print()

    return chunks


# Run if called directly
if __name__ == "__main__":
    run_chunker()