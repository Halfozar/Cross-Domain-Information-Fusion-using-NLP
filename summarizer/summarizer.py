# summarizer/summarizer.py

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from summarizer.preprocessor import (
    clean_text,
    sentence_tokenize,
    preprocess_for_tfidf,
)


def score_sentences_by_prompt(
    sentences: list,
    prompt: str,
    top_n: int = 10
) -> list:
    """
    Score each sentence by TF-IDF cosine similarity with the prompt.
    Returns top_n most relevant sentences in original document order.
    """
    if not sentences:
        return []

    if not prompt.strip():
        return []

    # Preprocess prompt
    processed_prompt = preprocess_for_tfidf(prompt)

    # Preprocess all sentences
    processed_sentences = [preprocess_for_tfidf(s) for s in sentences]

    # Safety check — if preprocessing emptied everything, fallback to raw
    if not processed_prompt.strip():
        processed_prompt = prompt.lower().strip()

    processed_sentences = [
        ps if ps.strip() else sentences[i].lower()
        for i, ps in enumerate(processed_sentences)
    ]

    # Build corpus: prompt first, then all sentences
    corpus = [processed_prompt] + processed_sentences

    # TF-IDF Vectorization
    try:
        vectorizer = TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=1,           # Accept terms appearing even once
            max_df=1.0          # No upper frequency cutoff
        )
        tfidf_matrix = vectorizer.fit_transform(corpus)
    except ValueError:
        # If vectorizer fails (empty vocabulary), return first top_n sentences
        return [
            (i, sentences[i], 0.0)
            for i in range(min(top_n, len(sentences)))
        ]

    # Prompt vector = row 0; sentence vectors = rows 1 onwards
    prompt_vector    = tfidf_matrix[0]
    sentence_vectors = tfidf_matrix[1:]

    # Compute cosine similarities
    scores = cosine_similarity(prompt_vector, sentence_vectors).flatten()

    # --- Fallback: if ALL scores are zero, use sentence length ranking ---
    if scores.max() == 0:
        # Rank by sentence length (longer = more content-rich)
        scores = np.array([len(s.split()) for s in sentences], dtype=float)
        scores = scores / scores.max()  # Normalize to 0-1

    # Pick top_n indices
    top_n = min(top_n, len(sentences))
    top_indices = np.argsort(scores)[::-1][:top_n]

    # Re-sort to preserve original document order
    top_indices_ordered = sorted(top_indices)

    return [
        (int(idx), sentences[idx], round(float(scores[idx]), 4))
        for idx in top_indices_ordered
    ]


def generate_summary(
    combined_text: str,
    prompt: str,
    top_n: int = 10,
) -> dict:
    """
    Full pipeline:
    1. Clean text
    2. Tokenize into sentences
    3. Score by prompt relevance
    4. Return structured result
    """

    # Step 1: Clean
    cleaned = clean_text(combined_text)

    # Step 2: Tokenize
    sentences = sentence_tokenize(cleaned)

    # --- Debug info stored in result ---
    debug_info = {
        "total_chars"    : len(cleaned),
        "total_sentences": len(sentences),
    }

    if len(sentences) == 0:
        return {
            "summary"        : "⚠️ No meaningful sentences found in documents.",
            "sentence_count" : 0,
            "top_sentences"  : [],
            "debug"          : debug_info,
        }

    # Cap top_n
    top_n = min(top_n, len(sentences))

    # Step 3: Score
    scored = score_sentences_by_prompt(sentences, prompt, top_n=top_n)

    if not scored:
        # Last resort fallback — just return first top_n sentences
        scored = [
            (i, sentences[i], 0.0)
            for i in range(min(top_n, len(sentences)))
        ]

    # Step 4: Build summary
    summary_text = " ".join([item[1] for item in scored])

    return {
        "summary"        : summary_text,
        "sentence_count" : len(scored),
        "top_sentences"  : scored,
        "debug"          : debug_info,
    }