# Cross-Domain-Information-Fusion-using-NLP

App Link
https://cross-domain-information-fusion-using-nlp-memdyehsvko3vuqrfbze.streamlit.app/
[cite: 3]

---

## 🧠 Technical Workflow

1.  **Extraction**: The system routes uploaded files to specific parsers (`PyPDF2` for PDFs, `python-docx` for Word, etc.) via `extractor.py`[cite: 2, 3].
2.  **Preprocessing**: In `preprocessor.py`, text is normalized and **lemmatized** using `spaCy` to improve vectorization accuracy[cite: 4].
3.  **Ranking**: 
    *   In `summarizer.py`, both the user's prompt and document sentences are converted into a **TF-IDF matrix**[cite: 6].
    *   **Cosine Similarity** is calculated to measure the mathematical distance between the prompt vector and every sentence vector in the documents[cite: 3, 6].
4.  **Selection**: The highest-scoring sentences are selected and re-ordered based on their original placement in the documents to maintain readability[cite: 3, 6].

---

## 📦 Core Dependencies
*   **Streamlit**: For the web-based user interface[cite: 5].
*   **NLTK & spaCy**: For sentence tokenization and linguistic cleaning[cite: 4, 5].
*   **scikit-learn**: For TF-IDF vectorization and similarity logic[cite: 5, 6].
*   **PyPDF2 & python-docx**: For document parsing[cite: 2, 5].
*   **Pandas**: For CSV text extraction and processing# Cross-Domain-Information-Fusion-using-NLP

**App Link:** [https://cross-domain-information-fusion-using-nlp-memdyehsvko3vuqrfbze.streamlit.app/](https://cross-domain-information-fusion-using-nlp-memdyehsvko3vuqrfbze.streamlit.app/)

This project is a self-contained **NLP-based multi-document summarization system**[cite: 3]. It allows users to upload multiple documents and generate a focused summary based on a natural language prompt—**without the need for external AI APIs**[cite: 1, 3].

---

## 🚀 Features

*   **Multi-Format Support**: Extract text from `pipeline.pdf`, `.docx`, `.txt`, and `.csv` files[cite: 1, 2, 3].
*   **Prompt-Guided Summarization**: Uses **TF-IDF** and **Cosine Similarity** to rank sentences based on relevance to your specific query[cite: 3, 6].
*   **No API Keys Required**: Runs 100% locally using pure Python NLP libraries like `NLTK` and `spaCy`[cite: 3, 5].
*   **Interactive UI**: Custom-styled Streamlit interface with adjustable summary length and sentence relevance scoring[cite: 1].
*   **Traceability**: Merges documents with clear headers and allows downloading of both the merged text and the final summary[cite: 1, 7].

---

## 🛠️ Project Structure
```text
Cross-Domain-Information-Fusion-using-NLP/
├── app.py                # Streamlit frontend and UI logic
├── requirements.txt      # Python dependencies
├── summarizer/           # Core NLP package
│   ├── extractor.py      # Document text extraction (PDF, Word, CSV, TXT)
│   ├── preprocessor.py   # NLP text cleaning and lemmatization
│   ├── summarizer.py     # TF-IDF & Cosine Similarity ranking logic
│   └── utils.py          # File merging and helper functions
└── outputs/              # Directory for auto-generated text exports

