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
    .stat-badge {
        display: inline-block;
        background-color: #4CAF50;
        color: white;
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 13px;
        margin: 2px;
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
    3. NLP engine scores every sentence
       by relevance to your prompt
    4. Top sentences are returned
       as your summary
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
    help="Hold Ctrl / Cmd to select multiple files"
)

# Show file summary if files are uploaded
if uploaded_files:
    st.markdown(f"**📂 {len(uploaded_files)} file(s) selected:**")
    for uf in uploaded_files:
        ext = uf.name.split(".")[-1].upper()
        size_kb = round(uf.size / 1024, 1)
        st.markdown(f"- `{uf.name}` — **{ext}** · {size_kb} KB")

st.divider()

# ─── Step 2: Prompt Input ─────────────────────────────────────────────────────
st.subheader("💬 Step 2: Enter Your Summary Prompt")
st.markdown(
    "Be specific — the more focused your prompt, "
    "the more relevant your summary will be."
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

# Prompt character counter
if prompt:
    st.caption(f"📝 Prompt length: `{len(prompt)}` characters")

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

    if len(prompt.strip()) < 10:
        st.warning("⚠️ Prompt is too short. Please enter a more descriptive prompt.")
        st.stop()

    # ── Step 1: Extract Text ──────────────────────────────────────────────────
    st.markdown("### ⚙️ Processing...")
    extraction_bar = st.progress(0, text="Starting extraction...")

    extracted_texts = {}
    extraction_errors = []
    total_files = len(uploaded_files)

    for i, file in enumerate(uploaded_files):
        extraction_bar.progress(
            int((i / total_files) * 50),
            text=f"🔍 Extracting: {file.name}"
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
            st.error(f"❌ **{file.name}** — Error: `{str(e)}`")

    extraction_bar.progress(50, text="✅ Extraction complete")

    if not extracted_texts:
        st.error(
            "🚫 No text could be extracted from any uploaded document. "
            "Please check your files and try again."
        )
        extraction_bar.empty()
        st.stop()

    # Warn about any failed files
    if extraction_errors:
        st.warning(
            f"⚠️ {len(extraction_errors)} file(s) could not be processed: "
            f"{', '.join(extraction_errors)}"
        )

    # ── Step 2: Merge Texts ───────────────────────────────────────────────────
    extraction_bar.progress(60, text="🔗 Merging documents...")

    # Internal merge — clean, no headers — for NLP only
    merged_internal    = merge_texts(extracted_texts)
    clean_text_for_nlp = get_clean_text_for_summarization(merged_internal)

    # Readable merge — with headers — for download only
    readable_merged = get_readable_merged_text(extracted_texts)
    merged_path     = save_merged_text(readable_merged)

    st.info(
        f"📋 **{len(extracted_texts)} document(s)** merged successfully · "
        f"**{len(clean_text_for_nlp):,}** clean characters ready for NLP"
    )

    # ── Debug Panel ───────────────────────────────────────────────────────────
    if show_debug:
        with st.expander("🛠️ Debug Information", expanded=True):
            st.markdown(
                f"<div class='debug-box'>"
                f"<strong>📏 Clean text length:</strong> "
                f"<code>{len(clean_text_for_nlp):,}</code> characters<br>"
                f"<strong>📁 Documents processed:</strong> "
                f"<code>{len(extracted_texts)}</code><br>"
                f"<strong>💬 Prompt length:</strong> "
                f"<code>{len(prompt)}</code> characters"
                f"</div>",
                unsafe_allow_html=True
            )
            st.markdown("**🔎 First 600 characters of clean text:**")
            st.code(clean_text_for_nlp[:600], language="text")
            st.markdown("**💬 Your Prompt:**")
            st.code(prompt, language="text")

    # ── Merged Text Preview ───────────────────────────────────────────────────
    if show_merged:
        with st.expander("📋 View Merged Document Text", expanded=False):
            st.text_area(
                "Combined Text (used for download)",
                readable_merged,
                height=300
            )

    # ── Step 3: NLP Summarization ─────────────────────────────────────────────
    extraction_bar.progress(75, text="🧠 Running NLP summarization...")

    result = generate_summary(
        combined_text=clean_text_for_nlp,
        prompt=prompt,
        top_n=top_n
    )

    extraction_bar.progress(100, text="✅ Done!")
    extraction_bar.empty()

    # ── Results Section ───────────────────────────────────────────────────────
    st.divider()
    st.subheader("📝 Summary Result")

    # Stats row
    dbg = result.get("debug", {})
    total_chars     = dbg.get("total_chars",     len(clean_text_for_nlp))
    total_sentences = dbg.get("total_sentences", 0)
    selected        = result["sentence_count"]

    col_s1, col_s2, col_s3, col_s4 = st.columns(4)
    with col_s1:
        st.metric("📁 Documents", len(extracted_texts))
    with col_s2:
        st.metric("📏 Characters", f"{total_chars:,}")
    with col_s3:
        st.metric("📜 Sentences Found", total_sentences)
    with col_s4:
        st.metric("✅ Sentences Selected", selected)

    st.markdown("")

    # ── Summary Output ────────────────────────────────────────────────────────
    if result["sentence_count"] == 0:
        st.error(
            "⚠️ Could not generate a meaningful summary.\n\n"
            "**Try the following:**\n"
            "- Enable **🛠️ Show Debug Info** in the sidebar to inspect extracted text\n"
            "- Make sure your documents contain readable text (not scanned images)\n"
            "- Try a different or more general prompt\n"
            "- Increase the **Number of Summary Sentences** slider\n"
        )
    else:
        # Summary display
        st.markdown("#### 📄 Generated Summary")
        st.markdown(
            f"<div class='summary-box'>{result['summary']}</div>",
            unsafe_allow_html=True
        )

        st.markdown("")

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

        # ── Sentence Relevance Scores ──────────────────────────────────────────
        if show_scores and result["top_sentences"]:
            st.divider()
            st.markdown("#### 🔍 Sentence Relevance Scores")
            st.caption(
                "Each sentence ranked by TF-IDF cosine similarity "
                "with your prompt. Higher score = more relevant."
            )

            for rank, (orig_idx, sentence, score) in enumerate(
                result["top_sentences"], start=1
            ):
                bar = format_score_bar(score)
                st.markdown(
                    f"<div class='score-item'>"
                    f"<strong>#{rank}</strong> &nbsp;|&nbsp; "
                    f"Score: <code>{score:.4f}</code> &nbsp; {bar}"
                    f"<br><br>{sentence}"
                    f"</div>",
                    unsafe_allow_html=True
                )

# ─── Empty State (no button pressed yet) ─────────────────────────────────────
else:
    st.markdown("""
    <div style='text-align:center; padding: 40px 20px; color: #888;'>
        <h3>👆 Upload your documents and enter a prompt to get started</h3>
        <p>Supports PDF · DOCX · TXT · CSV — multiple files at once</p>
    </div>
    """, unsafe_allow_html=True)