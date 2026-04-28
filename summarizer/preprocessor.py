# summarizer/preprocessor.py

import re
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords

nltk.download("punkt",     quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("punkt_tab", quiet=True)

STOP_WORDS = set(stopwords.words("english"))

# Noise patterns — ONLY block obvious junk lines
_NOISE_PATTERNS = re.compile(
    r"""
    ^={3,}$         |   # =========
    ^-{3,}$         |   # ---------
    ^\*{3,}$        |   # *********
    ^<<<.*>>>.*$        # <<<DOC_BREAK>>>
    """,
    re.VERBOSE | re.IGNORECASE
)


def clean_text(text: str) -> str:
    """
    Light cleaning — preserves as much real text as possible.
    Removes only internal doc tags and normalizes whitespace.
    """
    # Remove internal doc-break tags
    text = re.sub(r"<<<[^>]+>>>:[^\n]*\n?", "", text)
    text = re.sub(r"$$Document:[^$$]+\]",   "", text)
    text = re.sub(r"$$Column:[^$$]+\]",     "", text)
    text = re.sub(r"={3,}",                 "", text)
    # Normalize whitespace
    text = re.sub(r"\n+",  " ", text)
    text = re.sub(r"\s+",  " ", text)
    return text.strip()


def sentence_tokenize(text: str) -> list:
    """
    Split text into sentences using NLTK.
    Uses a RELAXED filter — only removes obvious junk.
    """
    raw_sentences = sent_tokenize(text)
    clean_sentences = []

    for s in raw_sentences:
        s = s.strip()

        # Skip very short strings only
        if len(s) < 15:
            continue

        # Skip separator-only lines
        if _NOISE_PATTERNS.match(s):
            continue

        # Skip lines with less than 30% alphabetic characters
        alpha_ratio = sum(c.isalpha() for c in s) / max(len(s), 1)
        if alpha_ratio < 0.30:   # Relaxed from 0.50 → 0.30
            continue

        clean_sentences.append(s)

    return clean_sentences


def simple_tokenize_and_clean(text: str) -> str:
    """
    Lightweight preprocessing for TF-IDF:
    - Lowercase
    - Remove punctuation
    - Remove stopwords
    - NO spaCy lemmatization (avoids over-filtering)
    
    This is intentionally simple to preserve vocabulary
    for better TF-IDF matching.
    """
    # Lowercase
    text = text.lower()
    # Remove punctuation but keep spaces
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    # Remove stopwords
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOP_WORDS and len(t) > 2]
    return " ".join(tokens)


def preprocess_for_tfidf(text: str) -> str:
    """
    TF-IDF preprocessing pipeline.
    Uses simple tokenization — avoids spaCy over-filtering.
    """
    text = clean_text(text)
    text = simple_tokenize_and_clean(text)
    return text