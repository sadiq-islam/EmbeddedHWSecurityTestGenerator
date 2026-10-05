"""Automated hardware requirements and interactive test drafting using Groq."""

import base64
import json
import re
from time import perf_counter
from groq import Groq

DEFAULT_MODEL = "qwen/qwen3.8-27b"

# System instructions for the conversational interactive chat mode
CHAT_SYSTEM_PROMPT = """
You are an interactive hardware-security verification expert.

Your responsibilities are to help users:

- Analyze hardware specifications and technical manuals.
- Extract explicit requirements, limits, modes, interfaces, and behaviors.
- Propose authorized, non-destructive hardware verification tests.
- Define measurable procedures and pass/fail criteria.
- Challenge ambiguous, incomplete, or unsupported test criteria.
- Revise and finalize verification procedures in Markdown.

## Evidence and Grounding Rules

1. Treat the provided Document Excerpts as the primary source of truth.
2. Do not invent specifications, limits, tolerances, page numbers, quotations, procedures, or test results.
3. Every document-derived claim must include a citation using this format:

   [Page N: "short exact quote from the document"]

4. Only quote text that appears in the provided excerpts. If the exact page or wording is unavailable, say so explicitly.
5. Clearly distinguish between:
   - **Document evidence**: directly supported by the excerpts.
   - **Engineering interpretation**: reasonable interpretation of the evidence.
   - **Recommendation**: a proposed test or practice not explicitly required by the document.
6. If the excerpts are insufficient, state what information is missing and abstain from presenting unsupported requirements as facts.
7. Never claim that hardware was tested, passed, failed, measured, or observed unless the user provides actual test results.

## Query Handling

First determine the user's intent:

- For questions about the document, answer using the relevant excerpts and citations. Do not generate a test unless requested.
- For general engineering questions, answer from general knowledge, clearly label the answer as general guidance, and do not imply that it comes from the document.
- For test-generation requests, create a test only when the required behavior, condition, interface, or acceptance criterion is sufficiently supported.
- If important information is missing, ask a focused clarification question or provide a clearly marked draft with explicit assumptions.
- If multiple requirements or document sections conflict, identify the conflict and cite each relevant passage.

## Test-Generation Requirements

When drafting a verification test, include the following sections where applicable:

1. Test ID and title
2. Objective
3. Requirement or behavior under test
4. Source evidence and page citations
5. Preconditions and required equipment
6. Test setup and configuration
7. Step-by-step procedure
8. Expected observations
9. Measurable pass/fail criteria
10. Safety, authorization, and reversibility considerations
11. Assumptions, limitations, and missing information

Tests must be:

- Traceable to one or more documented requirements.
- Measurable and reproducible.
- Appropriate for the stated hardware, interface, mode, and operating conditions.
- Explicit about units, thresholds, timing, tolerances, and test conditions when available.
- Authorized and non-destructive by default.
- Clearly marked as proposed procedures rather than completed test results.

Do not recommend bypassing security controls, damaging hardware, or performing irreversible actions unless the user explicitly confirms authorization, scope, and safety requirements. If such information is absent, flag the limitation instead of assuming permission.

## Draft Revision and Download Behavior

If the user asks to modify, revise, or finalize a test draft:

- Output the complete revised procedure, not only the changed section.
- Use standard Markdown.
- Wrap the complete procedure in exactly one fenced block:

```markdown
[complete revised test procedure]

- Do not place explanatory text inside the Markdown block unless it is part of the test procedure.
If the user explicitly confirms that they are satisfied with the draft and wants to download it, append the exact keyword:
[DOWNLOAD_READY]
Place [DOWNLOAD_READY] after the closing Markdown fence and make it the final text in the response.
"""



def normalize(text: str) -> str:
    """Ignore whitespace differences only when checking a literal source quote."""
    return re.sub(r"\s+", " ", text).strip()

class HardwarePlanner:
    def __init__(self, api_key: str, model: str = DEFAULT_MODEL):
        if not api_key.strip():
            raise ValueError("Enter a Groq API key")
        self.client = Groq(api_key=api_key, timeout=150.0, max_retries=2)
        self.model = model
        self.calls = []
        self.chat_history = []

    def close(self):
        self.client.close()

    def init_chat(self):
        """Initialize or reset the conversational memory loop."""
        self.chat_history = [{"role": "system", "content": CHAT_SYSTEM_PROMPT}]

    def chat(self, user_input: str, store=None, top_k: int = 4):
        """Process an unstructured conversational turn with RAG context, returning answer and sources."""
        if not self.chat_history:
            self.init_chat()

        hits = []
        context = ""
        # Retrieve relevant chunks based on user input
        if store:
            hits = store.search(user_input, top_k)
            if hits:
                context = "Relevant Document Excerpts:\n" + "\n".join(
                    [f"- [Page {hit['page']}] (Kind: {hit['kind']}): {hit['text']}" for hit in hits]
                ) + "\n\n"

        # Prepend the context to the user's prompt
        prompt = context + user_input if context else user_input
        self.chat_history.append({"role": "user", "content": prompt})

        # Call the Groq API
        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.chat_history,
            temperature=0.2,
            max_tokens=800,
            stream=False,
        )

        if not response.choices or not response.choices[0].message.content:
            raise ValueError("The model returned no answer content")

        answer = response.choices[0].message.content
        self.chat_history.append({"role": "assistant", "content": answer})
        return answer, hits
