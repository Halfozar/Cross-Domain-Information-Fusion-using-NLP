# summarizer/extractor.py

import re
import PyPDF2
import docx
import pandas as pd


# ─── Unicode Fix Map ──────────────────────────────────────────────────────────
# Fixes common PDF extraction corruptions
UNICODE_FIX_MAP = {
    "\ufb01": "fi",
    "\ufb02": "fl",
    "\ufb00": "ff",
    "\ufb03": "ffi",
    "\ufb04": "ffl",
    "\u2019": "'",
    "\u2018": "'",
    "\u201c": '"',
    "\u201d": '"',
    "\u2013": "-",
    "\u2014": "-",
    "\u2022": " ",
    "\u00d7": "x",       # × → x  (fixes 256×256 → 256x256)
    "\u00b1": "+/-",
    "\u2248": "approx",
    "\u2264": "<=",
    "\u2265": ">=",
    "\u03b1": "alpha",
    "\u03b2": "beta",
    "\u03bb": "lambda",
    "\u03bc": "mu",
    "\u03c3": "sigma",
    "\u2212": "-",        # minus sign
    "\u00b2": "2",        # superscript 2
    "\u00b3": "3",        # superscript 3
    "\xa0":   " ",        # non-breaking space
    "\t":     " ",        # tab to space
}


def fix_unicode(text: str) -> str:
    """
    Replace known problematic Unicode characters from PDF extraction
    with clean ASCII equivalents.
    """
    for char, replacement in UNICODE_FIX_MAP.items():
        text = text.replace(char, replacement)

    # Remove remaining non-ASCII characters that are not useful
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    return text


def clean_pdf_text(text: str) -> str:
    """
    Deep clean of raw PDF extracted text:
    - Fix unicode corruption
    - Remove figure/table captions
    - Remove page numbers
    - Remove citation-only lines
    - Remove headers/footers
    - Fix broken hyphenated words
    """
    # Step 1: Fix unicode
    text = fix_unicode(text)

    # Step 2: Fix hyphenated line breaks (common in PDFs)
    # e.g. "genera-\ntion" → "generation"
    text = re.sub(r"-\s*\n\s*", "", text)

    # Step 3: Remove figure/table captions
    text = re.sub(
        r"(Figure|Fig\.|Table|Equation|Algorithm|Appendix)\s*\d+[:\.\,].*?(?=\n|$)",
        "",
        text,
        flags=re.IGNORECASE
    )

    # Step 4: Remove lines that are ONLY a number (page numbers)
    text = re.sub(r"^\s*\d+\s*$", "", text, flags=re.MULTILINE)

    # Step 5: Remove "Published as a conference paper at..." lines
    text = re.sub(
        r"(Published as|Accepted at|Presented at|Proceedings of|"
        r"Workshop on|Conference on|In Proceedings).*?(?=\n|$)",
        "",
        text,
        flags=re.IGNORECASE
    )

    # Step 6: Remove arXiv / DOI / URL lines
    text = re.sub(r"arXiv:\S+", "", text)
    text = re.sub(r"doi:\S+",   "", text, flags=re.IGNORECASE)
    text = re.sub(r"https?://\S+", "", text)

    # Step 7: Remove lines that look like headers/footers
    # (short lines, all caps, or only symbols)
    lines = text.split("\n")
    cleaned_lines = []
    for line in lines:
        stripped = line.strip()

        # Skip blank lines
        if not stripped:
            continue

        # Skip very short lines (likely headers/page artifacts)
        if len(stripped) < 20 and not stripped.endswith("."):
            continue

        # Skip ALL-CAPS short lines (section headers in papers)
        if stripped.isupper() and len(stripped) < 60:
            continue

        # Skip lines that are mostly digits/symbols
        alpha_ratio = sum(c.isalpha() for c in stripped) / max(len(stripped), 1)
        if alpha_ratio < 0.40:
            continue

        cleaned_lines.append(line)

    return "\n".join(cleaned_lines).strip()


# ─── Extractors ───────────────────────────────────────────────────────────────

def extract_text_from_pdf(file) -> str:
    """
    Extract and deep-clean text from a PDF file.
    Handles unicode corruption and academic paper noise.
    """
    reader = PyPDF2.PdfReader(file)
    raw_pages = []

    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            raw_pages.append(extracted)

    raw_text = "\n".join(raw_pages)

    # Apply deep cleaning
    cleaned = clean_pdf_text(raw_text)
    return cleaned


def extract_text_from_docx(file) -> str:
    """Extract and clean text from a DOCX file."""
    doc = docx.Document(file)
    paragraphs = []

    for para in doc.paragraphs:
        text = para.text.strip()
        if text and len(text) > 10:
            paragraphs.append(text)

    raw_text = "\n".join(paragraphs)
    return fix_unicode(raw_text).strip()


def extract_text_from_txt(file) -> str:
    """Extract text from a plain TXT file."""
    raw = file.read().decode("utf-8", errors="ignore")
    return fix_unicode(raw).strip()


def extract_text_from_csv(file) -> str:
    """
    Extract text from CSV — combines all string columns.
    Each column is labeled for context.
    """
    df = pd.read_csv(file)
    text_parts = []

    for col in df.select_dtypes(include=["object"]).columns:
        values = df[col].dropna().astype(str).tolist()
        combined = " ".join(values)
        if combined.strip():
            text_parts.append(f"Column {col}: {combined}")

    return fix_unicode("\n".join(text_parts)).strip()


def extract_text(file, filename: str) -> str:
    """
    Route file to correct extractor based on extension.
    Returns cleaned extracted text as string.
    """
    ext = filename.lower().split(".")[-1]

    extractors = {
        "pdf":  extract_text_from_pdf,
        "docx": extract_text_from_docx,
        "txt":  extract_text_from_txt,
        "csv":  extract_text_from_csv,
    }

    if ext in extractors:
        return extractors[ext](file)
    else:
        return f"Unsupported file format: {filename}"
