# Hardware Test Planner

Upload a hardware specification PDF, click **Generate test plan**, and review
the proposed procedures, pass/fail criteria, missing information, and evidence.
The app creates its own retrieval queries; you do not need to type questions.

## Start on Windows

1. Install Python 3.11 or 3.12 if needed.
2. Extract this ZIP into its own folder.
3. Open PowerShell in the folder containing `run.cmd` and run:

```powershell
.\run.cmd
```

The launcher creates `.venv`, installs dependencies once, and starts Streamlit.
Open **http://localhost:8501** if the browser does not open automatically.
Paste your Groq API key in the sidebar, upload a PDF, and click **Generate test
plan**. No virtual environment activation command is needed. Keep the terminal
open while using the app; Ctrl+C stops the server.

The first installation can take several minutes: Sentence Transformers brings
PyTorch dependencies. The first retrieval-enabled run also downloads MiniLM.
This app never downloads ColPali or a large generation model. It serves only
on your local machine through the launcher.

Optional: copy `.env.example` to `.env` and enter `GROQ_API_KEY=your-key` there.
The app reads that file automatically. No real key is included in the ZIP.
The sidebar can override a saved key for a run.

## Who does what

| Step | Component | Location |
| --- | --- | --- |
| Extract text and render page images | PyMuPDF | Local |
| Discover requirements, including visible diagrams when enabled | Groq Qwen vision | Cloud |
| Embed selectable text and extracted visual quotes | MiniLM | Local CPU |
| Retrieve additional evidence for each requirement | FAISS | Local |
| Draft a test or state insufficient information | Same Groq model | Cloud |

The default Groq model is `qwen/qwen3.8-27b`. The Options panel allows another
model, but it must support JSON output and, when enabled, image input. Access
and quotas depend on your account; this package does not promise unlimited
free API use. PDF text/images are sent to Groq during generation.

## Simple options

- **Read diagrams and scanned pages:** enabled by default; sends each rendered
  page to Groq during requirement extraction. Disable it for selectable-text-only work.
- **Retrieved passages per requirement:** defaults to 4. Set it to **0** for a
  source-page-only run, which skips MiniLM loading and FAISS retrieval.
- **Maximum PDF pages:** rejects a larger file before any cloud call.

The generation stage always receives the requirement's original selectable
source-page text, its visual quote when applicable, and any retrieved passages.
Page images are read during extraction, not embedded or sent a second time
during test drafting. Only extracted visual quotes enter the text index.
This is **text RAG with optional visual requirement extraction**, not ColPali.

## Outputs and interpretation

Use **Download JSON** for requirements, tests, supplied evidence, retrieval
scores, timing, token usage and settings. Use **Download report** for a readable
Markdown test plan. Downloads are generated in the browser; the app does not
automatically write your PDF or results to disk.

Every proposed test has approval `pending`. A `draft` is a proposal to review,
not a result of testing hardware. `insufficient_information` has no operational
steps or pass/fail decision. Requirements marked non-measurable are retained
in the JSON but do not receive tests.

Text quotes are matched against extracted PDF text. Generated citation quotes
are checked against supplied context, and cited pages are derived from those
references. These checks **do not establish semantic correctness or coverage**.
Visual transcriptions can be wrong and require checking against the PDF image.
Tables can still lose structure during selectable-text extraction; vision can
help, but it is not a guaranteed exact table parser. Dense diagrams or tiny
labels may be unreadable at the bounded rendering resolution.

Cosine scores are displayed as raw similarity values, not confidence percentages.
Cost remains `null` because the app does not know your billing terms. Token
usage comes from returned API responses. The SDK may retry, so failed-attempt
usage can be missing. Total pipeline time excludes initial model download/load.

If a provider error stops a run, available partial work is retained for download
and explicitly marked `failed`. A new run starts from the beginning. Results
are in the current Streamlit session until downloaded; there is no persistent
database. Changing/removing the PDF clears the previous result. Changing a key
or option applies when you press Generate again.

## Comparing experiments

The 0-passage option provides a simple source-only baseline; 4 passages adds
text retrieval. For controlled research, use fixed PDFs, manually labelled
requirements/evidence, the same extraction results, prompts and generation
model. The app currently re-extracts requirements each run, so independently
generated runs may differ even at temperature 0. Save the JSON and compare
matched requirements; do not attribute every difference solely to retrieval.
The earlier separate ColPali package also needs the same extraction/prompt
settings before making a controlled comparison.

## Code and checks

| File | Purpose |
| --- | --- |
| `app.py` | Upload, options, progress, review and downloads |
| `src/pdf_processor.py` | Symbol-preserving text, rendering and token-bounded chunks |
| `src/vector_store.py` | MiniLM embeddings and FAISS retrieval |
| `src/chatbot.py` | Requirement discovery, evidence checks, generation, reporting |
| `src/models.py` | Validated requirement/test response shapes |
| `run.py`, `run.cmd` | Automatic local setup and launch |
| `CHANGELOG.md` | Changes from the uploaded HR chatbot |

After the launcher has installed dependencies:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Local checks use synthetic PDFs and fake model responses, with a real FAISS
index. They do not consume API tokens or download embedding weights. A live
Groq run is still needed to evaluate actual generated tests.
