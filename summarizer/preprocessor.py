# summarizer/preprocessor.py

import re
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords

nltk.download("punkt",     quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("punkt_tab", quiet=True)

STOP_WORDS = set(stopwords.words("english"))

# ─── Noise Patterns — separator lines only ────────────────────────────────────
_LINE_NOISE = re.compile(
    r"^(={3,}|-{3,}|\*{3,}|<<<.*>>>)$",
    re.IGNORECASE
)

# ─── Academic Paper Junk Sentence Patterns ────────────────────────────────────
_ACADEMIC_JUNK = re.compile(
    r"""
    # Figure / Table captions
    (figure|fig\.|table|algorithm|listing|equation)\s*\d+  |

    # Citation-only sentences  e.g. "(Zhang, 2019)"
    ^\s*\([\w\s,\.]+\d{4}\)\s*\.?\s*$                     |

    # "Published as / Accepted at" lines
    (published\s+as|accepted\s+at|presented\s+at|
     in\s+proceedings|workshop\s+on|preprint)              |

    # ArXiv / DOI / URL references
    (arxiv|doi:|https?://)                                 |

    # Page number artifacts  e.g. "33" or "Page 12"
    ^(page\s*)?\d{1,3}\.?\s*$                              |

    # "et al." only fragments
    ^\s*et\s+al\.?\s*$                                     |

    # Image dimension references  e.g. "256x256 images"
    \d+\s*[x×]\s*\d+\s*(image|pixel|resolution)           |

    # Lines starting with number + "Published"
    ^\d+\s+published                                       |

    # Conference name lines
    (ICLR|NeurIPS|ICML|CVPR|ACL|EMNLP|NAACL|ECCV|AAAI)
    \s*\d{4}
    """,
    re.VERBOSE | re.IGNORECASE
)

# ─── Minimum meaningful sentence ratio ────────────────────────────────────────
_MIN_ALPHA_RATIO = 0.45
_MIN_SENTENCE_LEN = 30
_MAX_SENTENCE_LEN = 1200   # Skip runaway merged lines


def clean_text(text: str) -> str:
    """
    Light cleaning — preserves real content.
    Removes internal doc tags and normalizes whitespace.
    """
    # Remove internal doc-break tags
    text = re.sub(r"<<<[^>]+>>>:[^\n]*\n?", "", text)
    text = re.sub(r"\[Document:[^\]]+\]",   "", text)
    text = re.sub(r"\[Column:[^\]]+\]",     "", text)
    text = re.sub(r"={3,}",                 "", text)
    # Normalize whitespace
    text = re.sub(r"\n+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def is_junk_sentence(sentence: str) -> bool:
    """
    Returns True if a sentence should be excluded from summarization.
    Catches figure captions, page numbers, citations, and conference refs.
    """
    s = sentence.strip()

    # Too short or too long
    if len(s) < _MIN_SENTENCE_LEN:
        return True
    if len(s) > _MAX_SENTENCE_LEN:
        return True

    # Separator line
    if _LINE_NOISE.match(s):
        return True

    # Low alphabetic ratio (mostly numbers/symbols)
    alpha_ratio = sum(c.isalpha() for c in s) / max(len(s), 1)
    if alpha_ratio < _MIN_ALPHA_RATIO:
        return True

    # Academic junk patterns
    if _ACADEMIC_JUNK.search(s):
        return True

    return False


def sentence_tokenize(text: str) -> list:
    """
    Tokenize text into clean sentences.
    Filters junk using is_junk_sentence().
    """
    raw_sentences = sent_tokenize(text)
    return [s.strip() for s in raw_sentences if not is_junk_sentence(s.strip())]


def simple_tokenize_and_clean(text: str) -> str:
    """
    Lightweight preprocessing for TF-IDF:
    - Lowercase
    - Remove punctuation
    - Remove stopwords
    No spaCy — avoids over-filtering vocabulary.
    """
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = [
        t for t in text.split()
        if t not in STOP_WORDS and len(t) > 2
    ]
    return " ".join(tokens)


def preprocess_for_tfidf(text: str) -> str:
    """Full TF-IDF preprocessing pipeline."""
    text = clean_text(text)
    text = simple_tokenize_and_clean(text)
    return text
