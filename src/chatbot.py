"""Automated hardware requirements and interactive test drafting using Groq."""

import base64
import json
import re
from time import perf_counter
from groq import Groq

DEFAULT_MODEL = "qwen/qwen3.8-27b"

# System instructions for the conversational interactive chat mode
CHAT_SYSTEM_PROMPT = """You are an interactive hardware security verification expert.
Your job is to help users analyze hardware specifications, suggest and generate test plans and test cases for HW testing with pass/fail criteria, challenge test criteria, and draft verification procedures.
Always ground your answers in the provided Document Excerpts.
When answering, explicitly cite the source page number and quote the relevant specification (e.g., "[Page 4: '...']"). If information is missing from the excerpts, state that clearly.
If the user asks to modify or finalize a test draft, output the complete revised test procedure in standard Markdown, wrapped in ```markdown ... ``` tags so the UI can extract it.
If the user explicitly states they are happy with the draft and want to download it, append the exact keyword [DOWNLOAD_READY] at the end of your response."""

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