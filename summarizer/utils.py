# summarizer/utils.py

import os
import re

DOC_SEPARATOR = "<<<DOC_BREAK>>>"


def merge_texts(extracted_texts: dict) -> str:
    """
    Merge texts with hidden internal separator tags.
    Safe for summarization pipeline.
    """
    merged_parts = []
    for filename, text in extracted_texts.items():
        if text.strip():
            merged_parts.append(
                f"{DOC_SEPARATOR}:{filename}\n{text.strip()}"
            )
    return "\n\n".join(merged_parts)


def get_clean_text_for_summarization(merged_text: str) -> str:
    """
    Strip ALL internal tags, separators, and metadata.
    Returns pure paragraph text for NLP.
    """
    lines = merged_text.split("\n")
    clean_lines = []
    for line in lines:
        stripped = line.strip()
        # Skip blank lines
        if not stripped:
            continue
        # Skip doc-break tags
        if stripped.startswith(DOC_SEPARATOR):
            continue
        # Skip === separators
        if re.match(r"^={3,}$", stripped):
            continue
        # Skip [Document: ...] headers
        if re.match(r"^\[Document:.*\]$", stripped, re.IGNORECASE):
            continue
        # Skip [Column: ...] headers
        if re.match(r"^\[Column:.*\]$", stripped, re.IGNORECASE):
            continue
        clean_lines.append(line)

    return "\n".join(clean_lines).strip()


def get_readable_merged_text(extracted_texts: dict) -> str:
    """
    Generate human-readable merged .txt for download only.
    NOT used for summarization.
    """
    merged_parts = []
    for filename, text in extracted_texts.items():
        if text.strip():
            header = (
                f"\n{'=' * 60}\n"
                f"[Document: {filename}]\n"
                f"{'=' * 60}\n"
            )
            merged_parts.append(header + text.strip())
    return "\n\n".join(merged_parts)


def save_merged_text(
    merged_text: str,
    output_path: str = "outputs/merged.txt"
) -> str:
    """Save merged text to file. Returns path."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(merged_text)
    return output_path


def save_summary(
    summary_text: str,
    output_path: str = "outputs/summary.txt"
) -> str:
    """Save summary to file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(summary_text)
    return output_path


def format_score_bar(score: float) -> str:
    """Visual bar for relevance score display."""
    filled = int(score * 20)
    return "█" * filled + "░" * (20 - filled) + f"  {score:.2%}"