# summarizer/extractor.py

import PyPDF2
import docx
import pandas as pd
import io


def extract_text_from_pdf(file) -> str:
    """Extract text from a PDF file object."""
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted + "\n"
    return text.strip()


def extract_text_from_docx(file) -> str:
    """Extract text from a DOCX file object."""
    doc = docx.Document(file)
    text = "\n".join([para.text for para in doc.paragraphs if para.text.strip()])
    return text.strip()


def extract_text_from_txt(file) -> str:
    """Extract text from a plain TXT file object."""
    return file.read().decode("utf-8", errors="ignore").strip()


def extract_text_from_csv(file) -> str:
    """Extract text from a CSV file — combines all string columns."""
    df = pd.read_csv(file)
    text_parts = []
    for col in df.select_dtypes(include=["object"]).columns:
        text_parts.append(f"[Column: {col}]\n" + " ".join(df[col].dropna().astype(str)))
    return "\n".join(text_parts).strip()


def extract_text(file, filename: str) -> str:
    """
    Route file to correct extractor based on extension.
    Returns extracted text as string.
    """
    ext = filename.lower().split(".")[-1]

    if ext == "pdf":
        return extract_text_from_pdf(file)
    elif ext == "docx":
        return extract_text_from_docx(file)
    elif ext == "txt":
        return extract_text_from_txt(file)
    elif ext == "csv":
        return extract_text_from_csv(file)
    else:
        return f"[Unsupported file format: {filename}]"
