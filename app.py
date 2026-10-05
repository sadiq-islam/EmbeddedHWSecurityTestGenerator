"""Streamlit UI for interactive hardware specification planning."""

import os
import re
import streamlit as st
from dotenv import load_dotenv
from src.chatbot import HardwarePlanner
from src.vector_store import VectorStore, load_embedder
from src.pdf_processor import PDFProcessor

load_dotenv()

st.set_page_config(page_title="Hardware Planner Chat", layout="wide")

if "planner" not in st.session_state:
    env_key = os.getenv("GROQ_API_KEY", "")
    st.session_state.planner = HardwarePlanner(api_key=env_key) if env_key else None

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []

if "current_draft" not in st.session_state:
    st.session_state.current_draft = ""

st.title("Embedded Hardware Security Test Generation and Verification Tool")

with st.sidebar:
    st.header("Document Setup")
    api_key_input = st.text_input("Groq API Key", type="password", value=os.getenv("GROQ_API_KEY", ""))

    if api_key_input and (st.session_state.planner is None or st.session_state.planner.client.api_key != api_key_input):
        st.session_state.planner = HardwarePlanner(api_key=api_key_input)
        st.success("API Key loaded.")

    uploaded_file = st.file_uploader("Upload PDF Specification", type=["pdf"])
    if uploaded_file and st.button("Process Document"):
        with st.spinner("Parsing Layout & OCR with Docling. This may take a moment..."):
            try:
                embedder = load_embedder()
                store = VectorStore(embedder)

                # Use the new Docling PDFProcessor
                with PDFProcessor(uploaded_file.read(), max_pages=100) as doc:
                    docling_doc, doc_chunks, total_pages = doc.extract_document()

                    store.build_index(doc_chunks)
                    st.session_state.vector_store = store

                # --- VISUAL EXTRACTION METRICS ---
                st.markdown("### Extraction Results")

                parsed_pages = len(docling_doc.pages)
                success_rate = parsed_pages / total_pages if total_pages > 0 else 0.0

                st.progress(success_rate, text=f"Extraction Coverage: {int(success_rate * 100)}%")

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Pages", f"{parsed_pages} / {total_pages}")
                with col2:
                    table_count = len(list(docling_doc.tables))
                    st.metric("Tables", table_count)
                with col3:
                    st.metric("Text Chunks", len(doc_chunks))

                if success_rate < 1.0:
                    st.warning("Some pages could not be fully parsed and were skipped.")
                else:
                    st.success("Document successfully parsed! You can now chat.")
            except Exception as e:
                st.error(f"Error processing document: {e}")

if st.session_state.current_draft:
    st.sidebar.markdown("### Active Draft")
    st.sidebar.download_button(
        label="Download Current Draft (.md)",
        data=st.session_state.current_draft,
        file_name="test_draft.md",
        mime="text/markdown"
    )

for msg in st.session_state.chat_messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["display_content"])
        if msg.get("sources"):
            with st.expander("🔍 Document Sources & Evidence"):
                for idx, src in enumerate(msg["sources"], 1):
                    score_info = f" (Score: {src['score']:.3f})" if "score" in src else ""
                    st.markdown(f"**Source {idx} — Page {src['page']}** [{src['kind']}]{score_info}")
                    st.code(src["text"], language=None)

if prompt := st.chat_input("Ask about the specs or challenge criteria..."):
    if not st.session_state.planner:
        st.error("Please provide a Groq API Key.")
        st.stop()

    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.chat_messages.append({"role": "user", "display_content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("Searching document & generating answer..."):
            response, hits = st.session_state.planner.chat(
                user_input=prompt,
                store=st.session_state.vector_store
            )

            if "[DOWNLOAD_READY]" in response:
                response = response.replace("[DOWNLOAD_READY]", "").strip()
                st.info("Your updated file is ready to download in the sidebar!")

            draft_match = re.search(r'```markdown(.*?)```', response, re.DOTALL)
            if draft_match:
                st.session_state.current_draft = draft_match.group(1).strip()
                st.success("Active draft updated.")

            st.markdown(response)

            if hits:
                with st.expander("🔍 Document Sources & Evidence"):
                    for idx, src in enumerate(hits, 1):
                        score_info = f" (Score: {src['score']:.3f})" if "score" in src else ""
                        st.markdown(f"**Source {idx} — Page {src['page']}** [{src['kind']}]{score_info}")
                        st.code(src["text"], language=None)

    st.session_state.chat_messages.append({
        "role": "assistant",
        "display_content": response,
        "sources": hits
    })