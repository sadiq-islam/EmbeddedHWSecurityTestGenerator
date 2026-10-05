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
        with st.spinner("Processing PDF and building vector index..."):
            embedder = load_embedder()
            store = VectorStore(embedder)

            with PDFProcessor(uploaded_file.read(), max_pages=100) as doc:
                pages = []
                for i in range(doc.pdf.page_count):
                    pages.append(doc.page(i, include_image=False))

                store.build_index(pages, [])
                st.session_state.vector_store = store
            st.success(f"Document ingested ({len(pages)} pages)! You can now chat.")

if st.session_state.current_draft:
    st.sidebar.markdown("### Active Draft")
    st.sidebar.download_button(
        label="Download Current Draft (.md)",
        data=st.session_state.current_draft,
        file_name="test_draft.md",
        mime="text/markdown"
    )

# Render conversation history with source inspection
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