# app.py

import streamlit as st
import os
import re

from summarizer.extractor import extract_text
from summarizer.utils import (
    merge_texts,
    get_clean_text_for_summarization,
    get_readable_merged_text,
    save_merged_text,
    save_summary,
    format_score_bar
)
from summarizer.summarizer import generate_summary

# ─── Page Configuration ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="📄 Multi-Doc NLP Summarizer",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
        padding: 0.5em 1.5em;
        font-size: 16px;
        border: none;
        transition: background-color 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .summary-box {
        background-color: #e8f5e9;
        border-left: 5px solid #4CAF50;
        padding: 20px;
        border-radius: 8px;
        font-size: 15px;
        line-height: 1.9;
        color: #1b1b1b;
    }
    .score-item {
        background-color: #fff3e0;
        border-left: 4px solid #FF9800;
        padding: 12px 15px;
        margin: 6px 0;
        border-radius: 6px;
        font-size: 13px;
        line-height: 1.7;
    }
    .debug-box {
        background-color: #e3f2fd;
        border-left: 4px solid #2196F3;
        padding: 12px 15px;
        border-radius: 6px;
        font-size: 13px;
    }
    .warning-box {
        background-color: #fff8e1;
        border-left: 5px solid #FFC107;
        padding: 18px 20px;
        border-radius: 8px;
        font-size: 14px;
        line-height: 1.8;
    }
    .relevance-bar {
        background-color: #f1f8e9;
        border-radius: 6px;
        padding: 10px 15px;
        font-size: 13px;
        margin-bottom: 12px;
    }
    .stat-badge {
        display: inline-block;
        background-color: #4CAF50;
        color: white;
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 13px;
        margin: 2px;
    }
    .empty-state {
        text-align: center;
        padding: 40px 20px;
        color: #888;
    }
</style>
""", unsafe_allow_html=True)

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/clouds/100/document.png", width=80)
    st.title("⚙️ Settings")

    st.markdown("### 📊 Summary Options")
    top_n = st.slider(
        "Number of Summary Sentences",
        min_value=3,
        max_value=30,
        value=10,
        help="How many relevant sentences to include in the final summary"
    )

    st.markdown("---")

    st.markdown("### 🖥️ Display Options")
    show_scores = st.checkbox(
        "🔍 Show Sentence Relevance Scores",
        value=False,
        help="Display TF-IDF cosine similarity scores for each selected sentence"
    )
    show_merged = st.checkbox(
        "📋 Show Merged Document Text",
        value=False,
        help="Preview the combined text from all uploaded documents"
    )
    show_debug = st.checkbox(
        "🛠️ Show Debug Info",
        value=False,
        help="Show internal processing stats — useful for troubleshooting"
    )

    st.markdown("---")

    st.markdown("### 🎚️ Relevance Thresholds")
    st.caption(
        "These control how strictly the prompt must "
        "match the document. Lower = more lenient."
    )
    custom_doc_threshold = st.slider(
        "Document Relevance Threshold",
        min_value=0.01,
        max_value=0.20,
        value=0.06,
        step=0.01,
        help=(
            "Minimum average top-5 score required to consider "
            "the prompt relevant to the document"
        )
    )
    custom_sent_threshold = st.slider(
        "Sentence Relevance Threshold",
        min_value=0.01,
        max_value=0.20,
        value=0.08,
        step=0.01,
        help=(
            "Minimum score a sentence must have "
            "to be included in the summary"
        )
    )

    st.markdown("---")

    st.markdown("### 📁 Supported File Formats")
    st.markdown("""
    - 📕 **PDF** — `.pdf`
    - 📘 **Word Document** — `.docx`
    - 📄 **Plain Text** — `.txt`
    - 📊 **Spreadsheet** — `.csv`
    """)

    st.markdown("---")

    st.markdown("### ℹ️ How It Works")
    st.markdown("""
    1. Upload one or more documents
    2. Enter a focused prompt
    3. App checks if prompt relates
       to the document content
    4. NLP scores every sentence
       by relevance to your prompt
    5. Only top scoring sentences
       above threshold are returned
    """)

    st.markdown("---")
    st.caption("🐍 Built with Python · NLP · Streamlit")
    st.caption("🔬 Engine: TF-IDF + Cosine Similarity")

# ─── Main Header ──────────────────────────────────────────────────────────────
st.title("📄 Multi-Document NLP Text Summarizer")
st.markdown(
    "Upload **multiple documents**, enter a **focused prompt**, and get a "
    "prompt-guided NLP summary — **no AI APIs, no internet required!**"
)
st.divider()

# ─── Step 1: File Upload ──────────────────────────────────────────────────────
st.subheader("📤 Step 1: Upload Your Documents")
st.markdown(
    "You can upload **multiple files** at once. "
    "All text will be merged before summarization."
)

uploaded_files = st.file_uploader(
    label="Upload documents (PDF, DOCX, TXT, CSV)",
    type=["pdf", "docx", "txt", "csv"],
    accept_multiple_files=True,
    help="Hold Ctrl / Cmd to select multiple files at once"
)

# Show uploaded file summary
if uploaded_files:
    st.markdown(f"**📂 {len(uploaded_files)} file(s) selected:**")
    for uf in uploaded_files:
        ext     = uf.name.split(".")[-1].upper()
        size_kb = round(uf.size / 1024, 1)
        st.markdown(f"- `{uf.name}` — **{ext}** · {size_kb} KB")

st.divider()

# ─── Step 2: Prompt Input ─────────────────────────────────────────────────────
st.subheader("💬 Step 2: Enter Your Summary Prompt")
st.markdown(
    "Be **specific** — the more focused your prompt, "
    "the more relevant your summary will be. "
    "Use words and topics that actually appear in your documents."
)

prompt = st.text_area(
    label="What should the summary focus on?",
    placeholder=(
        "e.g., 'Summarize the key findings related to artificial intelligence "
        "and its impact on healthcare and drug discovery'"
    ),
    height=110,
    help="Your prompt guides which sentences are selected from the documents"
)

# Prompt character counter + quality hint
if prompt:
    char_count = len(prompt.strip())
    word_count = len(prompt.strip().split())
    col_pc1, col_pc2 = st.columns(2)
    with col_pc1:
        st.caption(f"📝 Characters: `{char_count}` | Words: `{word_count}`")
    with col_pc2:
        if word_count < 5:
            st.caption("💡 Tip: A longer, more specific prompt gives better results")
        elif word_count >= 5:
            st.caption("✅ Good prompt length!")

st.divider()

# ─── Generate Button ──────────────────────────────────────────────────────────
col_left, col_center, col_right = st.columns([1, 2, 1])
with col_center:
    generate_btn = st.button(
        "🚀 Generate Summary",
        use_container_width=True,
        help="Click to start the NLP summarization process"
    )

st.divider()

# ─── Main Processing Logic ────────────────────────────────────────────────────
if generate_btn:

    # ── Validation ────────────────────────────────────────────────────────────
    if not uploaded_files:
        st.warning("⚠️ Please upload at least one document before generating.")
        st.stop()

    if not prompt.strip():
        st.warning("⚠️ Please enter a prompt to guide the summarization.")
        st.stop()

    if len(prompt.strip().split()) < 3:
        st.warning(
            "⚠️ Prompt is too short. "
            "Please enter at least 3 words describing what to summarize."
        )
        st.stop()

    # ── Step 1: Extract Text ──────────────────────────────────────────────────
    st.markdown("### ⚙️ Processing...")
    progress_bar = st.progress(0, text="Starting text extraction...")

    extracted_texts  = {}
    extraction_errors = []
    total_files       = len(uploaded_files)

    for i, file in enumerate(uploaded_files):
        progress_bar.progress(
            int((i / total_files) * 40),
            text=f"🔍 Extracting: **{file.name}**"
        )
        try:
            text = extract_text(file, file.name)
            if text and len(text.strip()) > 5:
                extracted_texts[file.name] = text
                st.success(
                    f"✅ **{file.name}** — "
                    f"`{len(text):,}` characters extracted successfully"
                )
            else:
                extraction_errors.append(file.name)
                st.error(
                    f"❌ **{file.name}** — No readable text found. "
                    f"File may be empty or image-based."
                )
        except Exception as e:
            extraction_errors.append(file.name)
            st.error(f"❌ **{file.name}** — Extraction error: `{str(e)}`")

    progress_bar.progress(40, text="✅ Extraction complete")

    # Guard: nothing extracted
    if not extracted_texts:
        st.error(
            "🚫 No text could be extracted from any uploaded document.\n\n"
            "**Possible causes:**\n"
            "- PDF is scanned/image-based (no embedded text)\n"
            "- File is corrupted or password protected\n"
            "- Unsupported encoding in TXT/CSV file"
        )
        progress_bar.empty()
        st.stop()

    # Warn about partial failures
    if extraction_errors:
        st.warning(
            f"⚠️ **{len(extraction_errors)}** file(s) could not be processed: "
            f"`{'`, `'.join(extraction_errors)}`"
        )

    # ── Step 2: Merge Texts ───────────────────────────────────────────────────
    progress_bar.progress(50, text="🔗 Merging documents...")

    # Clean text → for NLP only (no headers)
    merged_internal    = merge_texts(extracted_texts)
    clean_text_for_nlp = get_clean_text_for_summarization(merged_internal)

    # Readable text → for download only (with headers)
    readable_merged = get_readable_merged_text(extracted_texts)
    merged_path     = save_merged_text(readable_merged)

    st.info(
        f"📋 **{len(extracted_texts)} document(s)** merged · "
        f"**{len(clean_text_for_nlp):,}** clean characters ready for NLP"
    )

    # ── Debug Panel ───────────────────────────────────────────────────────────
    if show_debug:
        with st.expander("🛠️ Debug Information", expanded=True):
            st.markdown(
                f"<div class='debug-box'>"
                f"<strong>📏 Clean text length:</strong> "
                f"<code>{len(clean_text_for_nlp):,}</code> characters<br><br>"
                f"<strong>📁 Documents processed:</strong> "
                f"<code>{len(extracted_texts)}</code><br><br>"
                f"<strong>💬 Prompt:</strong> "
                f"<code>{prompt[:200]}</code><br><br>"
                f"<strong>🎚️ Document threshold:</strong> "
                f"<code>{custom_doc_threshold}</code><br><br>"
                f"<strong>🎚️ Sentence threshold:</strong> "
                f"<code>{custom_sent_threshold}</code>"
                f"</div>",
                unsafe_allow_html=True
            )
            st.markdown("**🔎 First 600 characters of clean extracted text:**")
            st.code(clean_text_for_nlp[:600], language="text")

    # ── Merged Text Preview ───────────────────────────────────────────────────
    if show_merged:
        with st.expander("📋 View Merged Document Text", expanded=False):
            st.text_area(
                "Combined text (used for download)",
                readable_merged,
                height=300
            )

    # ── Step 3: NLP Summarization ─────────────────────────────────────────────
    progress_bar.progress(65, text="🧠 Running relevance check...")

    result = generate_summary(
        combined_text          = clean_text_for_nlp,
        prompt                 = prompt,
        top_n                  = top_n,
        doc_threshold          = custom_doc_threshold,
        sent_threshold         = custom_sent_threshold,
    )

    progress_bar.progress(100, text="✅ Done!")
    progress_bar.empty()

    # ── Results Header ────────────────────────────────────────────────────────
    st.divider()
    st.subheader("📝 Summary Result")

    # Metrics row
    dbg             = result.get("debug", {})
    total_chars     = dbg.get("total_chars",     len(clean_text_for_nlp))
    total_sentences = dbg.get("total_sentences", 0)
    selected        = result["sentence_count"]
    rel_score       = result.get("relevance_score", 0.0)

    col_s1, col_s2, col_s3, col_s4, col_s5 = st.columns(5)
    with col_s1:
        st.metric("📁 Documents",          len(extracted_texts))
    with col_s2:
        st.metric("📏 Characters",         f"{total_chars:,}")
    with col_s3:
        st.metric("📜 Sentences Found",    total_sentences)
    with col_s4:
        st.metric("✅ Sentences Selected", selected)
    with col_s5:
        st.metric("🎯 Relevance Score",    f"{rel_score:.4f}")

    st.markdown("")

    # ── Status: No sentences extracted ───────────────────────────────────────
    if result["status"] == "no_sentences":
        st.error(
            "### 🚫 No Sentences Found\n\n"
            "No readable sentences could be extracted from your documents.\n\n"
            "**Possible fixes:**\n"
            "- Make sure your PDF is not scanned/image-based\n"
            "- Try a DOCX or TXT version of the same document\n"
            "- Enable **🛠️ Show Debug Info** in the sidebar to inspect raw text"
        )

    # ── Status: Prompt not relevant to document ───────────────────────────────
    elif result["status"] == "low_relevance":

        st.warning("### ⚠️ Prompt Does Not Match Document Content")

        rel_msg = result.get("relevance_msg", "")

        st.markdown(
            f"""
<div class='warning-box'>
<strong>🔍 Relevance Gate — Failed</strong><br><br>
Your prompt does not appear to be related to the
content of the uploaded document(s).<br><br>
<strong>Relevance Score:</strong>
&nbsp;<code>{rel_score:.4f}</code>
&nbsp;(minimum required:
<code>{custom_doc_threshold:.2f}</code>)<br><br>
<strong>Details:</strong> {rel_msg}
</div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("")
        st.markdown("#### 💡 How to Fix This")
        st.markdown("""
- ✏️ **Rewrite your prompt** using words and topics
  that actually appear in the document
- 🔎 **Enable 🛠️ Debug Info** in the sidebar to
  see exactly what text was extracted
- 📄 **Check your document** — it may be an
  image-based PDF with no embedded text
- 📝 **Try a general prompt** like:
  *"Summarize the main ideas of this document"*
- 🎚️ **Lower the thresholds** using the sliders
  in the sidebar (try `0.03` for both)
""")

        # Threshold adjustment tip
        with st.expander(
            "⚙️ Understanding Relevance Thresholds", expanded=False
        ):
            st.markdown(f"""
| Threshold | Current Value | What It Controls |
|---|---|---|
| Document Threshold | `{custom_doc_threshold:.2f}` | Min avg score for prompt to be considered relevant to doc |
| Sentence Threshold | `{custom_sent_threshold:.2f}` | Min score for each sentence to enter the summary |

**Recommended values by use case:**

| Use Case | Doc Threshold | Sent Threshold |
|---|---|---|
| General documents | `0.06` | `0.08` |
| Technical / research papers | `0.04` | `0.05` |
| Short documents (< 1 page) | `0.03` | `0.03` |
| CSV / structured data | `0.03` | `0.03` |

Use the **sidebar sliders** to adjust without editing code.
            """)

    # ── Status: Success ───────────────────────────────────────────────────────
    elif result["status"] == "success":

        # Relevance indicator bar
        rel_color  = (
            "#4CAF50" if rel_score >= 0.15 else
            "#FF9800" if rel_score >= 0.06 else
            "#f44336"
        )
        rel_label  = (
            "🟢 Strong Match"   if rel_score >= 0.15 else
            "🟡 Moderate Match" if rel_score >= 0.06 else
            "🔴 Weak Match"
        )

        st.markdown(
            f"""
<div class='relevance-bar'
     style='border-left: 5px solid {rel_color};'>
🎯 <strong>Prompt Relevance:</strong>
&nbsp;<code>{rel_score:.4f}</code>
&nbsp;|&nbsp; {rel_label}
&nbsp;|&nbsp; {result.get("relevance_msg", "")}
</div>
            """,
            unsafe_allow_html=True
        )

        # Summary output box
        st.markdown("#### 📄 Generated Summary")
        st.markdown(
            f"<div class='summary-box'>{result['summary']}</div>",
            unsafe_allow_html=True
        )

        st.markdown("")

        # Save summary to file
        save_summary(result["summary"])

        # ── Download Buttons ──────────────────────────────────────────────────
        st.markdown("#### ⬇️ Download Outputs")
        dl_col1, dl_col2 = st.columns(2)

        with dl_col1:
            st.download_button(
                label="📥 Download Summary (.txt)",
                data=result["summary"].encode("utf-8"),
                file_name="summary.txt",
                mime="text/plain",
                use_container_width=True,
                help="Download the generated summary as a plain text file"
            )

        with dl_col2:
            st.download_button(
                label="📥 Download Merged Document (.txt)",
                data=readable_merged.encode("utf-8"),
                file_name="merged_documents.txt",
                mime="text/plain",
                use_container_width=True,
                help="Download the combined text from all uploaded documents"
            )

        # ── Sentence Scores Panel ─────────────────────────────────────────────
        if show_scores and result["top_sentences"]:
            st.divider()
            st.markdown("#### 🔍 Sentence Relevance Scores")
            st.caption(
                f"Only sentences scoring above "
                f"`{custom_sent_threshold}` are shown. "
                f"Higher score = stronger match with your prompt."
            )

            for rank, (orig_idx, sentence, score) in enumerate(
                result["top_sentences"], start=1
            ):
                bar       = format_score_bar(score)
                bar_color = (
                    "#4CAF50" if score >= 0.15 else
                    "#FF9800" if score >= 0.08 else
                    "#9E9E9E"
                )
                st.markdown(
                    f"<div class='score-item'"
                    f" style='border-left-color:{bar_color};'>"
                    f"<strong>#{rank}</strong>"
                    f" &nbsp;|&nbsp; "
                    f"Score: <code>{score:.4f}</code>"
                    f" &nbsp; {bar}"
                    f"<br><br>{sentence}"
                    f"</div>",
                    unsafe_allow_html=True
                )

        # ── Debug Panel (success state) ───────────────────────────────────────
        if show_debug:
            with st.expander("🛠️ Debug — Processing Stats", expanded=False):
                st.markdown(
                    f"<div class='debug-box'>"
                    f"<strong>Total chars processed:</strong> "
                    f"<code>{dbg.get('total_chars', 0):,}</code><br><br>"
                    f"<strong>Total sentences tokenized:</strong> "
                    f"<code>{dbg.get('total_sentences', 0)}</code><br><br>"
                    f"<strong>Sentences above threshold:</strong> "
                    f"<code>{dbg.get('sentences_above_threshold', 0)}</code><br><br>"
                    f"<strong>TF-IDF vocabulary size:</strong> "
                    f"<code>{dbg.get('vocab_size', 0):,}</code><br><br>"
                    f"<strong>Max sentence score:</strong> "
                    f"<code>{dbg.get('all_scores_max', 0):.4f}</code><br><br>"
                    f"<strong>Mean sentence score:</strong> "
                    f"<code>{dbg.get('all_scores_mean', 0):.4f}</code>"
                    f"</div>",
                    unsafe_allow_html=True
                )

    # ── Unknown status fallback ───────────────────────────────────────────────
    else:
        st.error(
            f"❌ Unexpected status: `{result.get('status', 'unknown')}`. "
            f"Please try again or check the debug panel."
        )

# ─── Empty State ──────────────────────────────────────────────────────────────
else:
    st.markdown("""
<div class='empty-state'>
    <h3>👆 Upload your documents and enter a prompt to get started</h3>
    <p>Supports PDF &nbsp;·&nbsp; DOCX &nbsp;·&nbsp; TXT &nbsp;·&nbsp; CSV</p>
    <p>Multiple files can be uploaded at once</p>
    <br>
    <p style='font-size:13px; color:#aaa;'>
        ✅ No internet required &nbsp;·&nbsp;
        ✅ No API keys needed &nbsp;·&nbsp;
        ✅ Fully local NLP
    </p>
</div>
    """, unsafe_allow_html=True)
