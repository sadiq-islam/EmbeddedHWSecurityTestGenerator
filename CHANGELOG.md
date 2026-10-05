# Changes from the uploaded HR Policy Assistant

## User workflow

- Replaced HR Q&A with automatic hardware requirement discovery and security
  test planning: upload PDF, enter key, click Generate.
- Kept a small Streamlit interface and the local MiniLM/FAISS retrieval approach.
- Removed company-name state and the HR classifier. One configurable Groq model
  handles extraction and drafting through the official SDK.
- Added `run.cmd`/`run.py` so setup no longer requires manual virtual environment
  activation. Dependency changes trigger an automatic install on the next launch.
- Added optional visual page reading using the same API, without adding ColPali.
- Added source-only mode by setting retrieved passages to 0.

## PDF and retrieval corrections

- Replaced destructive character filtering with whitespace cleanup that preserves
  newlines, comparison signs, units, percentages, brackets and other Unicode.
- Replaced pdfplumber with PyMuPDF to support both text extraction and rendering
  using a single PDF dependency.
- Read PDFs from uploaded bytes, eliminating temporary-file cleanup failures.
- Detect empty selectable text directly; no synthetic PAGE markers can make an
  empty page appear readable. In text-only mode the skipped page is reported.
- Use tokenizer offsets and a token budget instead of an unenforced character
  target. Long sentences are split; overlap is bounded.
- Cache the actual MiniLM instance and derive vector dimensions from its output.
- Search once per requirement; remove the duplicate query encoding and conflicting
  0.20/0.25 gates. Top-k evidence is still subject to sufficiency/grounding review.
- Preserve visual-only quotes separately from selectable text with their provenance.
- Removed unused pickle save/load methods. JSON downloads provide transparent
  experiment records; this version does not persist the vector index.

## Grounding and output

- Added Pydantic response schemas for requirements, citations and proposed tests.
- Check text requirement quotes against the source page. Reject unavailable
  citations and unmatched citation quotes; derive page numbers in code.
- Invalid test-schema or citation outputs become explicit abstentions, with a
  validation warning. Provider/extraction failures stop the run and retain partial work.
- Store `draft` or `insufficient_information`, plus pending human approval.
  Abstentions contain no operational steps or invented pass/fail criteria.
- Prompts treat document content as untrusted data and request non-destructive
  procedures. Prompts and literal checks are not a guarantee of semantic grounding.
- Removed the misleading `from_policy=True` grounding signal and percentage-like
  confidence display. FAISS scores are raw cosine similarities.
- Added full supplied-context records, generated quotes, model/settings, PDF hash,
  prompt version, timing and returned token counts. Unknown cost remains null.
- Added downloadable JSON and Markdown reports.

## UI and configuration repairs

- Load `.env` relative to the app directory; sidebar key entry remains available.
- Recreate the Groq client for every run so API key changes take effect.
- Identify uploaded documents by SHA-256 rather than filename and byte count.
- Clear stale results when the PDF changes or is removed.
- Render content through Streamlit text/Markdown components without raw HTML
  interpolation of user text, model output or filenames.
- Avoid nested expanders and remove decorative HTML/CSS that obscured the app logic.

## Validation and limits

- Twelve local checks passed, covering symbol preservation, long chunks, empty pages, PDF
  limits, real FAISS retrieval, source-only mode, automatic generation, rejected
  quotations, invalid citations, abstention and partial results.
- Streamlit's test runner was used to check initial and result screens.
- No live Groq requests, real embedding inference, or hardware execution were
  performed during these checks. The first model download and Windows launcher
  still need confirmation on the user's machine.
- Visual transcriptions are unverified; citations matching text do not prove a
  claim follows from it. No empirical grounding/recall/coverage score is claimed.
- Re-running currently repeats extraction and generation. A future controlled
  retrieval study should reuse fixed requirement/evidence fixtures across variants.
