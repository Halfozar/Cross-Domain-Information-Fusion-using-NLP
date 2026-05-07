# summarizer/summarizer.py

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from summarizer.preprocessor import (
    clean_text,
    sentence_tokenize,
    preprocess_for_tfidf,
)

# ─── Default Thresholds ──────────────────────────────────────────────────────

# Minimum cosine score a sentence must have to be included
SENTENCE_RELEVANCE_THRESHOLD = 0.08

# If the AVERAGE of top-5 scores is below this,
# the prompt is considered unrelated to the document
DOCUMENT_RELEVANCE_THRESHOLD = 0.06

# Minimum number of sentences that must pass threshold
# before we consider the summary valid
MIN_VALID_SENTENCES = 2


def check_prompt_document_relevance(
    prompt: str,
    sentences: list,
    threshold: float = DOCUMENT_RELEVANCE_THRESHOLD
) -> tuple:
    """
    Before summarizing, check whether the prompt is
    meaningfully related to the document content at all.

    Returns:
        (is_relevant: bool, top_avg_score: float, reason: str)
    """
    if not sentences or not prompt.strip():
        return False, 0.0, "Empty document or prompt."

    processed_prompt    = preprocess_for_tfidf(prompt)
    processed_sentences = [preprocess_for_tfidf(s) for s in sentences]

    # Safety: if preprocessing kills the prompt, use raw
    if not processed_prompt.strip():
        processed_prompt = prompt.lower().strip()

    # Filter out empty processed sentences
    valid_pairs = [
        (orig, proc)
        for orig, proc in zip(sentences, processed_sentences)
        if proc.strip()
    ]

    if not valid_pairs:
        return False, 0.0, "All sentences became empty after preprocessing."

    orig_sentences, proc_sentences = zip(*valid_pairs)

    corpus = [processed_prompt] + list(proc_sentences)

    try:
        vectorizer = TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=1,
            max_df=1.0
        )
        tfidf_matrix = vectorizer.fit_transform(corpus)
    except ValueError:
        return False, 0.0, "TF-IDF vectorization failed — vocabulary may be empty."

    prompt_vector    = tfidf_matrix[0]
    sentence_vectors = tfidf_matrix[1:]

    scores = cosine_similarity(prompt_vector, sentence_vectors).flatten()

    # Take average of top-5 scores as document relevance indicator
    top_k       = min(5, len(scores))
    top_scores  = np.sort(scores)[::-1][:top_k]
    avg_top     = float(np.mean(top_scores))
    max_score   = float(np.max(scores))

    if avg_top < threshold:
        reason = (
            f"Low relevance detected. "
            f"Best match score: `{max_score:.4f}` | "
            f"Top-{top_k} average: `{avg_top:.4f}` | "
            f"Threshold: `{threshold}`"
        )
        return False, avg_top, reason

    return True, avg_top, f"Relevant. Top-{top_k} avg score: `{avg_top:.4f}`"


def score_sentences_by_prompt(
    sentences: list,
    prompt: str,
    top_n: int = 10,
    threshold: float = SENTENCE_RELEVANCE_THRESHOLD
) -> tuple:
    """
    Score each sentence by TF-IDF cosine similarity with prompt.
    Only keeps sentences that EXCEED the relevance threshold.

    Returns:
        (scored_sentences, all_scores, vectorizer_vocab_size)
        scored_sentences: list of (original_index, sentence, score)
    """
    if not sentences or not prompt.strip():
        return [], [], 0

    processed_prompt    = preprocess_for_tfidf(prompt)
    processed_sentences = [preprocess_for_tfidf(s) for s in sentences]

    # Fallback: if prompt preprocessing empties it
    if not processed_prompt.strip():
        processed_prompt = prompt.lower().strip()

    # Filter empty processed sentences — keep originals paired
    valid_pairs = []
    for i, (orig, proc) in enumerate(zip(sentences, processed_sentences)):
        if proc.strip():
            valid_pairs.append((i, orig, proc))

    if not valid_pairs:
        return [], [], 0

    indices, orig_sents, proc_sents = zip(*valid_pairs)

    corpus = [processed_prompt] + list(proc_sents)

    try:
        vectorizer = TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=1,
            max_df=1.0
        )
        tfidf_matrix  = vectorizer.fit_transform(corpus)
        vocab_size    = len(vectorizer.vocabulary_)
    except ValueError:
        return [], [], 0

    prompt_vector    = tfidf_matrix[0]
    sentence_vectors = tfidf_matrix[1:]

    scores = cosine_similarity(prompt_vector, sentence_vectors).flatten()

    # ── Apply Threshold Filter ────────────────────────────────────────────────
    # Only keep sentences whose score is above threshold
    threshold_filtered = [
        (int(indices[i]), orig_sents[i], round(float(scores[i]), 4))
        for i in range(len(scores))
        if scores[i] >= threshold
    ]

    if not threshold_filtered:
        return [], list(scores), vocab_size

    # Sort by score descending, pick top_n
    threshold_filtered.sort(key=lambda x: x[2], reverse=True)
    top_results = threshold_filtered[:top_n]

    # Re-sort by original document order
    top_results.sort(key=lambda x: x[0])

    return top_results, list(scores), vocab_size


def generate_summary(
    combined_text: str,
    prompt: str,
    top_n: int = 10,
    doc_threshold: float = DOCUMENT_RELEVANCE_THRESHOLD,
    sent_threshold: float = SENTENCE_RELEVANCE_THRESHOLD,
) -> dict:
    """
    Full NLP summarization pipeline with relevance gating.

    Args:
        combined_text: Raw text from all uploaded documents
        prompt: User's summarization prompt/query
        top_n: Maximum number of sentences to return
        doc_threshold: Minimum relevance score for prompt to match document
        sent_threshold: Minimum relevance score for individual sentences

    Returns a structured dict with:
    - summary text (or None if not relevant)
    - sentence count
    - top sentences with scores
    - relevance info
    - debug stats
    - status: "success" | "low_relevance" | "no_sentences" | "error"
    """

    # Step 1: Clean
    cleaned   = clean_text(combined_text)

    # Step 2: Tokenize
    sentences = sentence_tokenize(cleaned)

    debug_info = {
        "total_chars"    : len(cleaned),
        "total_sentences": len(sentences),
        "prompt_length"  : len(prompt),
        "doc_threshold"  : doc_threshold,
        "sent_threshold" : sent_threshold,
    }

    # ── Guard: No sentences ───────────────────────────────────────────────────
    if len(sentences) == 0:
        return {
            "status"         : "no_sentences",
            "summary"        : None,
            "sentence_count" : 0,
            "top_sentences"  : [],
            "relevance_score": 0.0,
            "relevance_msg"  : "No sentences could be extracted from documents.",
            "debug"          : debug_info,
        }

    # Step 3: Relevance Gate
    # Check if prompt is even related to the document BEFORE scoring
    is_relevant, rel_score, rel_msg = check_prompt_document_relevance(
        prompt, sentences, threshold=doc_threshold
    )

    debug_info["relevance_score"] = rel_score
    debug_info["relevance_msg"]   = rel_msg

    # ── Guard: Prompt not related to document ─────────────────────────────────
    if not is_relevant:
        return {
            "status"         : "low_relevance",
            "summary"        : None,
            "sentence_count" : 0,
            "top_sentences"  : [],
            "relevance_score": rel_score,
            "relevance_msg"  : rel_msg,
            "debug"          : debug_info,
        }

    # Step 4: Score sentences with threshold
    top_n_adjusted = min(top_n, len(sentences))
    scored, all_scores, vocab_size = score_sentences_by_prompt(
        sentences, prompt, top_n=top_n_adjusted,
        threshold=sent_threshold
    )

    debug_info["vocab_size"]  = vocab_size
    debug_info["all_scores_max"]  = round(float(max(all_scores)), 4) if all_scores else 0
    debug_info["all_scores_mean"] = round(float(np.mean(all_scores)), 4) if all_scores else 0
    debug_info["sentences_above_threshold"] = len(scored)

    # ── Guard: Not enough sentences passed threshold ──────────────────────────
    if len(scored) < MIN_VALID_SENTENCES:
        return {
            "status"         : "low_relevance",
            "summary"        : None,
            "sentence_count" : len(scored),
            "top_sentences"  : scored,
            "relevance_score": rel_score,
            "relevance_msg"  : (
                f"Only `{len(scored)}` sentence(s) matched the prompt "
                f"above the threshold of `{sent_threshold}`. "
                f"Try lowering the threshold or using a more relevant prompt."
            ),
            "debug"          : debug_info,
        }

    # Step 5: Build summary
    summary_text = " ".join([item[1] for item in scored])

    return {
        "status"         : "success",
        "summary"        : summary_text,
        "sentence_count" : len(scored),
        "top_sentences"  : scored,
        "relevance_score": rel_score,
        "relevance_msg"  : rel_msg,
        "debug"          : debug_info,
    }
