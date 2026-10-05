# Hardware Test Planner

Upload a hardware specification PDF, click **Process Document**, and interactively chat with an AI assistant to analyze specifications, challenge criteria, and draft verification procedures. The app creates its own retrieval queries; you do not need to hunt for exact pages.

## Start on Windows

1. Install Python 3.11 or newer if needed.
2. Extract this ZIP into its own folder.
3. Open PowerShell in the folder containing `run.py` and run:

```powershell
python run.py
```

The launcher creates `.venv`, installs dependencies once, and starts Streamlit.
Open **http://localhost:8501** if the browser does not open automatically.
Paste your Groq API key in the sidebar, upload a PDF, and click **Process Document**. No virtual environment activation command is needed. Keep the terminal open while using the app; Ctrl+C stops the server.

The first installation can take several minutes: Sentence Transformers brings
PyTorch dependencies. The first retrieval-enabled run also downloads MiniLM.
This app never downloads a large generation model. It serves only
on your local machine through the launcher.

Optional: copy `.env.example` to `.env` and enter `GROQ_API_KEY=your-key` there.
The app reads that file automatically. No real key is included in the ZIP.
The sidebar can override a saved key for a run.

## Who does what

| Step | Component | Location |
| --- | --- | --- |
| Extract text from PDF pages | PyMuPDF | Local |
| Embed selectable text | MiniLM | Local CPU |
| Retrieve relevant passages | FAISS | Local |
| Answer queries and draft tests | Groq Qwen | Cloud |

The default Groq model is `qwen/qwen3.8-27b`. Access and quotas depend on your account; this package does not promise unlimited
free API use. PDF text is sent to Groq during chat generation.

## Simple options

- **Interactive Chat:** Ask specific questions about hardware behaviors, or request test plans directly in the chat interface.
- **Document Sources & Evidence:** Every assistant response includes an expander to review the exact chunks of text retrieved by FAISS and their cosine similarity scores.
- **Maximum PDF pages:** Rejects a file larger than 100 pages before any processing to prevent memory overload.

## Outputs and interpretation

When you ask the assistant to finalize a test draft, it outputs the complete revised test procedure in standard Markdown. The UI detects this and provides a **Download Current Draft (.md)** button in the sidebar.

The AI is instructed to treat the PDF as untrusted reference material and prefer read-only observations. Every proposed test is a draft to review, not a result of testing hardware. The model will explicitly state if missing information prevents the creation of a safe, measurable test.

Text quotes are matched against extracted PDF text. Generated citation quotes
are checked against supplied context, and cited pages are derived from those
references. These checks **do not establish semantic correctness or coverage**.
Tables can still lose structure during selectable-text extraction.

Cosine scores are displayed as raw similarity values, not confidence percentages.
Results are in the current Streamlit session until downloaded; there is no persistent database. Changing/removing the PDF clears the previous result. Changing a key
or option applies when you press enter on your next prompt.

## Comparing experiments

For controlled research, use fixed PDFs, the same prompts, and the same generation model. The chat history and retrieved chunks dictate the model's responses, so independently generated conversations may differ even at temperature 0. Save your drafts and compare matched requirements; do not attribute every difference solely to retrieval.

## Code and checks

| File | Purpose |
| --- | --- |
| `app.py` | Chat UI, state management, file uploading, and downloads |
| `src/pdf_processor.py` | Symbol-preserving text extraction and token-bounded chunks |
| `src/vector_store.py` | MiniLM embeddings and FAISS retrieval |
| `src/chatbot.py` | Conversational RAG loop and Groq API integration |
| `src/models.py` | Validated requirement/test response shapes |
| `run.py` | Automatic local setup and launch |
| `test_workflow.py`, `test_app.py` | Offline regression tests |

After the launcher has installed dependencies:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -p "test_*.py" -v
```

Local checks use synthetic PDFs and fake model responses, with a real FAISS
index. They do not consume API tokens or download embedding weights. A live
Groq run is still needed to evaluate actual generated tests.