# ============================================================
# Part 0 - Data Collection
# DS Learning Assistant - RAG Graduation Project
# ============================================================
# This script collects data science articles from GitHub.
# It saves everything to: ds_rag/data/articles.json
# ============================================================

import json
import time
import os
import requests

# ---- Settings -----------------------------------------------

SAVE_PATH = "ds_rag/data/articles.json"
WAIT_TIME = 0.5   # seconds to wait between requests (be polite)

# We tell the website who we are
HEADERS = {
    "User-Agent": "DS-Learning-RAG-Project/1.0 (student project)"
}


# ---- List of articles to collect ----------------------------
# Each entry is: (url, title, tags)
# Tags help us filter later (Level 2 - Query Intelligence)

ARTICLES_TO_SCRAPE = [

    # --- Microsoft ML for Beginners ---
    ("https://raw.githubusercontent.com/microsoft/ML-For-Beginners/main/1-Introduction/1-intro-to-ML/README.md",
     "Introduction to Machine Learning",
     ["machine-learning", "introduction"]),

    ("https://raw.githubusercontent.com/microsoft/ML-For-Beginners/main/1-Introduction/2-history-of-ML/README.md",
     "History of Machine Learning",
     ["machine-learning", "history"]),

    ("https://raw.githubusercontent.com/microsoft/ML-For-Beginners/main/1-Introduction/3-fairness/README.md",
     "Fairness in Machine Learning",
     ["machine-learning", "ethics"]),

    ("https://raw.githubusercontent.com/microsoft/ML-For-Beginners/main/1-Introduction/4-techniques-of-ML/README.md",
     "Techniques of Machine Learning",
     ["machine-learning", "algorithms"]),

    ("https://raw.githubusercontent.com/microsoft/ML-For-Beginners/main/2-Regression/1-Tools/README.md",
     "Tools for Regression",
     ["regression", "python", "scikit-learn"]),

    ("https://raw.githubusercontent.com/microsoft/ML-For-Beginners/main/2-Regression/2-Data/README.md",
     "Data for Regression",
     ["regression", "data", "pandas"]),

    ("https://raw.githubusercontent.com/microsoft/ML-For-Beginners/main/2-Regression/3-Linear/README.md",
     "Linear Regression",
     ["regression", "linear-regression", "supervised"]),

    ("https://raw.githubusercontent.com/microsoft/ML-For-Beginners/main/2-Regression/4-Logistic/README.md",
     "Logistic Regression",
     ["regression", "logistic-regression", "classification"]),

    ("https://raw.githubusercontent.com/microsoft/ML-For-Beginners/main/4-Classification/1-Introduction/README.md",
     "Introduction to Classification",
     ["classification", "supervised"]),

    ("https://raw.githubusercontent.com/microsoft/ML-For-Beginners/main/4-Classification/2-Classifiers-1/README.md",
     "Classifiers Part 1",
     ["classification", "algorithms"]),

    ("https://raw.githubusercontent.com/microsoft/ML-For-Beginners/main/4-Classification/3-Classifiers-2/README.md",
     "Classifiers Part 2",
     ["classification", "algorithms"]),

    ("https://raw.githubusercontent.com/microsoft/ML-For-Beginners/main/4-Classification/4-Applied/README.md",
     "Applied Classification",
     ["classification", "applied"]),

    ("https://raw.githubusercontent.com/microsoft/ML-For-Beginners/main/5-Clustering/1-Visualize/README.md",
     "Visualization for Clustering",
     ["clustering", "unsupervised", "visualization"]),

    ("https://raw.githubusercontent.com/microsoft/ML-For-Beginners/main/5-Clustering/2-K-Means/README.md",
     "K-Means Clustering",
     ["clustering", "k-means", "unsupervised"]),

    ("https://raw.githubusercontent.com/microsoft/ML-For-Beginners/main/6-NLP/1-Introduction-to-NLP/README.md",
     "Introduction to NLP",
     ["nlp", "natural-language-processing"]),

    ("https://raw.githubusercontent.com/microsoft/ML-For-Beginners/main/6-NLP/3-Translation-Sentiment/README.md",
     "NLP Translation and Sentiment Analysis",
     ["nlp", "sentiment", "translation"]),

    ("https://raw.githubusercontent.com/microsoft/ML-For-Beginners/main/6-NLP/4-Hotel-Reviews-1/README.md",
     "NLP Hotel Reviews Analysis Part 1",
     ["nlp", "text-analysis", "pandas"]),

    ("https://raw.githubusercontent.com/microsoft/ML-For-Beginners/main/6-NLP/5-Hotel-Reviews-2/README.md",
     "NLP Hotel Reviews Analysis Part 2",
     ["nlp", "text-analysis", "scikit-learn"]),

    ("https://raw.githubusercontent.com/microsoft/ML-For-Beginners/main/7-TimeSeries/1-Introduction/README.md",
     "Introduction to Time Series",
     ["time-series", "forecasting"]),

    ("https://raw.githubusercontent.com/microsoft/ML-For-Beginners/main/7-TimeSeries/2-ARIMA/README.md",
     "Time Series with ARIMA",
     ["time-series", "arima", "forecasting"]),

    ("https://raw.githubusercontent.com/microsoft/ML-For-Beginners/main/8-Reinforcement/1-QLearning/README.md",
     "Q-Learning and Reinforcement Learning",
     ["reinforcement-learning", "q-learning"]),

    ("https://raw.githubusercontent.com/microsoft/ML-For-Beginners/main/8-Reinforcement/2-Gym/README.md",
     "Reinforcement Learning with OpenAI Gym",
     ["reinforcement-learning", "openai-gym"]),

    # --- Microsoft Data Science for Beginners ---
    ("https://raw.githubusercontent.com/microsoft/Data-Science-For-Beginners/main/1-Introduction/01-defining-data-science/README.md",
     "Defining Data Science",
     ["data-science", "introduction"]),

    ("https://raw.githubusercontent.com/microsoft/Data-Science-For-Beginners/main/1-Introduction/02-ethics/README.md",
     "Data Science Ethics",
     ["data-science", "ethics"]),

    ("https://raw.githubusercontent.com/microsoft/Data-Science-For-Beginners/main/1-Introduction/03-defining-data/README.md",
     "Defining Data",
     ["data-science", "data-types"]),

    ("https://raw.githubusercontent.com/microsoft/Data-Science-For-Beginners/main/1-Introduction/04-stats-and-probability/README.md",
     "Statistics and Probability",
     ["statistics", "probability", "math"]),

    ("https://raw.githubusercontent.com/microsoft/Data-Science-For-Beginners/main/2-Working-With-Data/05-relational-databases/README.md",
     "Relational Databases and SQL",
     ["sql", "databases", "relational"]),

    ("https://raw.githubusercontent.com/microsoft/Data-Science-For-Beginners/main/2-Working-With-Data/06-non-relational/README.md",
     "Non-Relational Data",
     ["nosql", "databases"]),

    ("https://raw.githubusercontent.com/microsoft/Data-Science-For-Beginners/main/2-Working-With-Data/07-python/README.md",
     "Working with Data in Python",
     ["python", "pandas", "numpy"]),

    ("https://raw.githubusercontent.com/microsoft/Data-Science-For-Beginners/main/2-Working-With-Data/08-data-preparation/README.md",
     "Data Preparation and Cleaning",
     ["data-cleaning", "preprocessing", "pandas"]),

    ("https://raw.githubusercontent.com/microsoft/Data-Science-For-Beginners/main/3-Data-Visualization/09-visualization-quantities/README.md",
     "Visualizing Quantities",
     ["visualization", "matplotlib"]),

    ("https://raw.githubusercontent.com/microsoft/Data-Science-For-Beginners/main/3-Data-Visualization/10-visualization-distributions/README.md",
     "Visualizing Distributions",
     ["visualization", "statistics"]),

    ("https://raw.githubusercontent.com/microsoft/Data-Science-For-Beginners/main/3-Data-Visualization/11-visualization-proportions/README.md",
     "Visualizing Proportions",
     ["visualization", "matplotlib"]),

    ("https://raw.githubusercontent.com/microsoft/Data-Science-For-Beginners/main/3-Data-Visualization/12-visualization-relationships/README.md",
     "Visualizing Relationships",
     ["visualization", "correlation"]),

    ("https://raw.githubusercontent.com/microsoft/Data-Science-For-Beginners/main/3-Data-Visualization/13-meaningful-visualizations/README.md",
     "Creating Meaningful Visualizations",
     ["visualization", "data-storytelling"]),

    # --- ML Glossary (deep reference content) ---
    ("https://raw.githubusercontent.com/bfortuner/ml-glossary/master/docs/activation_functions.rst",
     "Activation Functions",
     ["deep-learning", "neural-networks"]),

    ("https://raw.githubusercontent.com/bfortuner/ml-glossary/master/docs/backpropagation.rst",
     "Backpropagation",
     ["deep-learning", "neural-networks", "training"]),

    ("https://raw.githubusercontent.com/bfortuner/ml-glossary/master/docs/calculus.rst",
     "Calculus for Machine Learning",
     ["math", "calculus"]),

    ("https://raw.githubusercontent.com/bfortuner/ml-glossary/master/docs/datasets.rst",
     "Common Machine Learning Datasets",
     ["datasets", "benchmark"]),

    ("https://raw.githubusercontent.com/bfortuner/ml-glossary/master/docs/gradient_descent.rst",
     "Gradient Descent",
     ["optimization", "training"]),

    ("https://raw.githubusercontent.com/bfortuner/ml-glossary/master/docs/layers.rst",
     "Neural Network Layers",
     ["deep-learning", "neural-networks"]),

    ("https://raw.githubusercontent.com/bfortuner/ml-glossary/master/docs/linear_algebra.rst",
     "Linear Algebra for ML",
     ["math", "linear-algebra"]),

    ("https://raw.githubusercontent.com/bfortuner/ml-glossary/master/docs/linear_regression.rst",
     "Linear Regression Deep Dive",
     ["regression", "supervised"]),

    ("https://raw.githubusercontent.com/bfortuner/ml-glossary/master/docs/logistic_regression.rst",
     "Logistic Regression Deep Dive",
     ["classification", "supervised"]),

    ("https://raw.githubusercontent.com/bfortuner/ml-glossary/master/docs/loss_functions.rst",
     "Loss Functions",
     ["training", "optimization"]),

    ("https://raw.githubusercontent.com/bfortuner/ml-glossary/master/docs/nn_concepts.rst",
     "Neural Network Concepts",
     ["deep-learning", "neural-networks"]),

    ("https://raw.githubusercontent.com/bfortuner/ml-glossary/master/docs/optimizers.rst",
     "Optimizers",
     ["optimization", "training", "deep-learning"]),

    ("https://raw.githubusercontent.com/bfortuner/ml-glossary/master/docs/regularization.rst",
     "Regularization Techniques",
     ["regularization", "overfitting"]),

    ("https://raw.githubusercontent.com/bfortuner/ml-glossary/master/docs/reinforcement_learning.rst",
     "Reinforcement Learning",
     ["reinforcement-learning"]),

    ("https://raw.githubusercontent.com/bfortuner/ml-glossary/master/docs/math_notation.rst",
     "Math Notation for ML",
     ["math", "notation"]),

    ("https://raw.githubusercontent.com/bfortuner/ml-glossary/master/docs/linear_algebra.rst",
     "Linear Algebra Reference",
     ["math", "linear-algebra"]),
]


# ---- Step 1: Download one file ------------------------------

def download_file(url):
    """
    Download a file from the internet.
    Returns the text content, or None if it failed.
    """
    try:
        response = requests.get(url, headers=HEADERS, timeout=15)

        # Check if the download was successful
        if response.status_code == 200:
            return response.text
        else:
            print(f"  ERROR: Got status {response.status_code} for {url}")
            return None

    except Exception as error:
        print(f"  ERROR: Could not download {url}")
        print(f"  Reason: {error}")
        return None


# ---- Step 2: Split text into sections -----------------------

def split_into_sections(text):
    """
    Split the article text into sections based on headings.
    A heading line starts with # (markdown) or has === under it (RST).

    Returns a list like:
    [
        {"heading": "Introduction", "text": "This is the intro..."},
        {"heading": "What is ML?", "text": "Machine learning is..."},
    ]
    """
    sections = []
    lines = text.splitlines()

    current_heading = "Introduction"
    current_text_lines = []

    i = 0
    while i < len(lines):
        line = lines[i]

        # Check if this is a Markdown heading (starts with #)
        if line.startswith("#"):
            # Save whatever we collected so far
            if current_text_lines:
                joined_text = " ".join(current_text_lines).strip()
                if len(joined_text) > 50:   # skip very short sections
                    sections.append({
                        "heading": current_heading,
                        "text": joined_text
                    })

            # Start a new section
            current_heading = line.lstrip("#").strip()
            current_text_lines = []

        # Check if this is an RST heading (next line is all = or - or ~)
        elif i + 1 < len(lines) and lines[i + 1] and all(c in "=-~^" for c in lines[i + 1]):
            # Save whatever we collected so far
            if current_text_lines:
                joined_text = " ".join(current_text_lines).strip()
                if len(joined_text) > 50:
                    sections.append({
                        "heading": current_heading,
                        "text": joined_text
                    })

            # Start a new section with this line as the heading
            current_heading = line.strip()
            current_text_lines = []
            i += 1   # skip the underline on the next line

        else:
            # Regular content line - clean it up and add it
            clean_line = line.strip()

            # Skip empty lines and image tags
            if not clean_line:
                i += 1
                continue
            if clean_line.startswith("!["):
                i += 1
                continue
            if clean_line.startswith(".. "):
                i += 1
                continue

            current_text_lines.append(clean_line)

        i += 1

    # Don't forget the last section
    if current_text_lines:
        joined_text = " ".join(current_text_lines).strip()
        if len(joined_text) > 50:
            sections.append({
                "heading": current_heading,
                "text": joined_text
            })

    return sections


# ---- Step 3: Build one article dict -------------------------

def build_article(url, title, tags, raw_text):
    """
    Takes the raw downloaded text and turns it into
    a clean structured dictionary we can save to JSON.
    """
    sections = split_into_sections(raw_text)

    # Join all sections into one big string (useful for word count)
    full_text = ""
    for section in sections:
        full_text += section["heading"] + "\n"
        full_text += section["text"] + "\n\n"

    word_count = len(full_text.split())

    article = {
        "title":      title,
        "url":        url,
        "tags":       tags,
        "sections":   sections,
        "full_text":  full_text.strip(),
        "word_count": word_count
    }

    return article


# ---- Main: run everything -----------------------------------

def main():
    # Create the output folder if it does not exist yet
    os.makedirs("ds_rag/data", exist_ok=True)

    print("=" * 55)
    print("  DS Learning Assistant - Data Scraper")
    print("=" * 55)
    print(f"  Will collect {len(ARTICLES_TO_SCRAPE)} articles")
    print(f"  Saving to: {SAVE_PATH}")
    print("=" * 55)
    print()

    all_articles = []
    success_count = 0
    fail_count = 0

    for i, (url, title, tags) in enumerate(ARTICLES_TO_SCRAPE, start=1):

        print(f"[{i:02d}/{len(ARTICLES_TO_SCRAPE)}] {title}")

        # Download the file
        raw_text = download_file(url)

        if raw_text is None:
            print("        FAILED - skipping")
            fail_count += 1
        else:
            # Turn it into a structured article
            article = build_article(url, title, tags, raw_text)

            if len(article["sections"]) == 0:
                print("        SKIPPED - no sections found")
                fail_count += 1
            else:
                all_articles.append(article)
                print(f"        OK - {article['word_count']} words, {len(article['sections'])} sections")
                success_count += 1

        # Wait a bit before the next request
        time.sleep(WAIT_TIME)

    # Save everything to JSON
    print()
    print("Saving to JSON file...")
    with open(SAVE_PATH, "w", encoding="utf-8") as f:
        json.dump(all_articles, f, ensure_ascii=False, indent=2)

    # Final report
    total_words = sum(a["word_count"] for a in all_articles)
    total_sections = sum(len(a["sections"]) for a in all_articles)

    print()
    print("=" * 55)
    print("  DONE!")
    print("=" * 55)
    print(f"  Collected : {success_count} articles")
    print(f"  Failed    : {fail_count} articles")
    print(f"  Sections  : {total_sections}")
    print(f"  Words     : {total_words:,}")
    print(f"  Saved to  : {SAVE_PATH}")
    print("=" * 55)


# Run the script
main()