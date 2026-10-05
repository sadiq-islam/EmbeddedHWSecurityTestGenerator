# Hardware Test Planner

Upload a hardware specification PDF, click **Process Document**, and interactively chat with an AI assistant to analyze specifications, challenge criteria, and draft verification procedures. The system uses Docling for AI-driven layout parsing and a two-stage retrieval-augmented generation (RAG) pipeline for complex manuals, 2-up booklets, and technical tables.

## Start on Windows

1. Install Python 3.11 or newer, if needed.
2. Extract this ZIP into its own folder.
3. Open PowerShell in the folder containing `run.py` and run:

```powershell
python run.py
```

Open [http://localhost:8501](http://localhost:8501) if the browser does not open automatically.

Paste your Groq API key in the sidebar, upload a PDF, and click **Process Document**. No virtual-environment activation command is needed. Keep the terminal open while using the app; press `Ctrl+C` to stop the server.

The first installation can take several minutes because Sentence Transformers installs PyTorch dependencies. The first retrieval-enabled run also downloads the MiniLM embedding model and MS-MARCO cross-encoder re-ranking weights. The app does not download a large generation model: generation is provided through the Groq API, while parsing and retrieval run locally.

Optional: copy `.env.example` to `.env` and add:

```dotenv
GROQ_API_KEY=your-key
```

No real key is included in the ZIP. A key entered in the sidebar overrides the saved key for the current run.

## Architecture: Who Does What

| Step | Component | Location |
| --- | --- | --- |
| Parse layout, tables, and OCR | Docling and RapidOCR | Local CPU |
| Semantic document chunking | `HierarchicalChunker` | Local CPU |
| Embed text and context | MiniLM | Local CPU |
| Broad retrieval | FAISS | Local RAM |
| High-accuracy re-ranking | MS-MARCO cross-encoder | Local CPU |
| Answer queries and draft tests | Groq Qwen Vision | Cloud API |

The default Groq model is `qwen/qwen3.8-27b`. Access and quotas depend on your account; this package does not promise unlimited free API use. PDF text is sent to Groq during chat generation.

## Features and Options

- **Visual extraction metrics:** The sidebar displays parsed pages, extracted tables, and structural chunks before chatting.
- **Two-stage retrieval:** FAISS performs broad vector search, followed by cross-encoder re-ranking for more precise technical matches.
- **Header injection:** Parent headings are added to chunks to preserve relationships such as a step's connection to its `Mode 3` section.
- **Interactive chat:** Ask about hardware behavior or request verification-test plans.
- **Sources and evidence:** Each response includes an expander showing retrieved text chunks and their relevance scores.
- **Maximum PDF pages:** PDFs larger than 100 pages are rejected before processing.

## Outputs and Interpretation

When asked to finalize a test draft, the assistant outputs a complete revised procedure in Markdown. The UI detects the draft and provides a **Download Current Draft (.md)** button in the sidebar.

The assistant treats the PDF as untrusted reference material and prefers read-only observations. Every proposed test is a draft for review, not evidence that hardware was tested. The assistant explicitly identifies missing information when a safe, measurable test cannot be created.

Docling preserves complex layouts and nested tables as Markdown grids using AI Table Structure Recognition (TSR). Generated quotes are checked against the supplied context, and cited pages are derived from Docling provenance metadata.

Re-ranker scores are displayed as raw similarity values. Results exist only in the current Streamlit session until downloaded; there is no persistent database. Changing or removing the PDF clears the previous result. Changing a key or option takes effect after submitting the next prompt.

## Comparing Experiments

For controlled research, use the same PDFs, prompts, and generation model. Chat history and retrieved chunks influence responses, so independently generated conversations may differ even at temperature 0. Save drafts and compare matched requirements; do not attribute every difference solely to retrieval.

## Code and Checks

| File | Purpose |
| --- | --- |
| `app.py` | Chat UI, state management, file uploading, visual metrics, and downloads |
| `src/pdf_processor.py` | AI layout analysis, TSR, RapidOCR, and hierarchical Markdown chunking |
| `src/vector_store.py` | MiniLM embeddings, FAISS retrieval, header injection, and cross-encoder re-ranking |
| `src/chatbot.py` | Conversational RAG loop and Groq API integration |
| `src/models.py` | Validated requirement and test-response shapes |
| `run.py` | Automatic local setup and launch |
| `test_workflow.py`, `test_app.py` | Offline regression tests |

After the launcher has installed dependencies, run:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -p "test_*.py" -v
```

Local checks use synthetic PDFs, fake model responses, and a real FAISS index. They do not consume API tokens or download embedding weights. A live Groq run is still required to evaluate generated tests.
