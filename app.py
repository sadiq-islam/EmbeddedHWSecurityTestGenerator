"""Streamlit UI for interactive hardware specification planning."""

import os
import re
import streamlit as st
from dotenv import load_dotenv
from src.chatbot import HardwarePlanner
from src.vector_store import VectorStore, load_embedder
from src.pdf_processor import PDFProcessor

# Load environment variables (.env file)
load_dotenv()

# Configure the Streamlit page metadata and layout
st.set_page_config(page_title="Hardware Planner Chat", layout="wide")

# Initialize the Groq planner in session state if not already present
if "planner" not in st.session_state:
    env_key = os.getenv("GROQ_API_KEY", "")
    st.session_state.planner = HardwarePlanner(api_key=env_key) if env_key else None

# Initialize the vector store to None (populated after PDF upload)
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

# Initialize the conversation history list
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []

# Store the most recently generated markdown draft
if "current_draft" not in st.session_state:
    st.session_state.current_draft = ""

st.title("Embedded Hardware Security Test Generation and Verification Tool")

with st.sidebar:
    st.header("Document Setup")
    # API key input field
    api_key_input = st.text_input("Groq API Key", type="password", value=os.getenv("GROQ_API_KEY", ""))

    # Update the planner instance if a new API key is provided
    if api_key_input and (st.session_state.planner is None or st.session_state.planner.client.api_key != api_key_input):
        st.session_state.planner = HardwarePlanner(api_key=api_key_input)
        st.success("API Key loaded.")

    # PDF file uploader
    uploaded_file = st.file_uploader("Upload PDF Specification", type=["pdf"])
    if uploaded_file and st.button("Process Document"):
        with st.spinner("Processing PDF and building vector index..."):
            # Load the embedding model and initialize the FAISS vector store
            embedder = load_embedder()
            store = VectorStore(embedder)

            # Process the PDF document pages
            with PDFProcessor(uploaded_file.read(), max_pages=100) as doc:
                pages = []
                for i in range(doc.pdf.page_count):
                    # We skip image extraction in this loop since chat relies primarily on text RAG
                    pages.append(doc.page(i, include_image=False))

                # Build the text index
                store.build_index(pages, [])
                st.session_state.vector_store = store
            st.success(f"Document ingested ({len(pages)} pages)! You can now chat.")

# If a draft is generated, show a download button in the sidebar
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
        # Display collapsible source citations if they exist for this message
        if msg.get("sources"):
            with st.expander("🔍 Document Sources & Evidence"):
                for idx, src in enumerate(msg["sources"], 1):
                    score_info = f" (Score: {src['score']:.3f})" if "score" in src else ""
                    st.markdown(f"**Source {idx} — Page {src['page']}** [{src['kind']}]{score_info}")
                    st.code(src["text"], language=None)

# The chat input field at the bottom of the screen
if prompt := st.chat_input("Ask about the specs or challenge criteria..."):
    if not st.session_state.planner:
        st.error("Please provide a Groq API Key.")
        st.stop()

    # Immediately display the user's message
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.chat_messages.append({"role": "user", "display_content": prompt})

    # Generate and display the assistant's response
    with st.chat_message("assistant"):
        with st.spinner("Searching document & generating answer..."):
            # Query the planner with RAG enabled
            response, hits = st.session_state.planner.chat(
                user_input=prompt,
                store=st.session_state.vector_store
            )

            # Check for the special download keyword from the LLM
            if "[DOWNLOAD_READY]" in response:
                response = response.replace("[DOWNLOAD_READY]", "").strip()
                st.info("Your updated file is ready to download in the sidebar!")

            # Extract markdown blocks (like test drafts) using regex
            draft_match = re.search(r'```markdown(.*?)```', response, re.DOTALL)
            if draft_match:
                # Save the block into session state so the download button appears
                st.session_state.current_draft = draft_match.group(1).strip()
                st.success("Active draft updated.")

            # Display the text response
            st.markdown(response)

            # Display the source evidence
            if hits:
                with st.expander("🔍 Document Sources & Evidence"):
                    for idx, src in enumerate(hits, 1):
                        score_info = f" (Score: {src['score']:.3f})" if "score" in src else ""
                        st.markdown(f"**Source {idx} — Page {src['page']}** [{src['kind']}]{score_info}")
                        st.code(src["text"], language=None)

    # Save the assistant's response to history
    st.session_state.chat_messages.append({
        "role": "assistant",
        "display_content": response,
        "sources": hits
    })