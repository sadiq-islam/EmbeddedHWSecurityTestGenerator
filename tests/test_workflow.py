"""Offline regression checks; no API key, model download, or hardware is needed."""

import json
import unittest
import numpy as np
import pymupdf
from src.pdf_processor import PDFProcessor, create_chunks
from src.models import TestDraft
from src.chatbot import HardwarePlanner, validate_citations, render_report
from src.vector_store import VectorStore


class Tokenizer:
    # Character tokens let tests force boundaries inside very long sentences.
    # Simulates a tokenizer by returning 1-to-1 character mappings.
    def __call__(self, text, **kwargs):
        return {"offset_mapping": [(i, i + 1) for i in range(len(text))]}
    def encode(self, text, **kwargs):
        return list(text)
    def num_special_tokens_to_add(self, pair):
        return 2


class Embedder:
    # A fake embedder that generates deterministic float arrays instead of using a real ML model.
    max_seq_length = 256
    tokenizer = Tokenizer()
    def encode(self, texts, **kwargs):
        # Deterministic vectors allow a real FAISS round trip without a downloaded model.
        # It weights vectors purely based on counts of specific keywords ("debug" and "production").
        values = np.array([[1 + t.lower().count("debug"), 1 + t.lower().count("production")]
                           for t in texts], dtype=np.float32)
        # Normalize the vectors for cosine similarity
        return values / np.linalg.norm(values, axis=1, keepdims=True)


def pdf_bytes(blank=False):
    # Generates a minimal, synthetic PDF entirely in-memory using PyMuPDF.
    with pymupdf.open() as doc:
        page = doc.new_page()
        if not blank:
            page.insert_text((72, 72), "Debug access is disabled in production mode.")
        doc.new_page().insert_text((72, 72), "The production state can be observed read-only.")
        return doc.tobytes()


def proposed_test():
    # Returns a valid mock TestDraft dictionary to simulate LLM output.
    return {"title": "Observe production debug state", "status": "draft",
            "preconditions": ["Operator identifies production mode"],
            "steps": ["Observe debug access without writing"],
            "pass_criteria": "Debug access is disabled", "fail_criteria": "Debug access is enabled",
            "missing_information": [], "citations": [{"reference_id": "S1",
            "quote": "Debug access is disabled in production mode."}]}


class FakePlanner(HardwarePlanner):
    # Overrides the main HardwarePlanner to intercept network calls to the LLM.
    def __init__(self, invalid_citation=False, fail_generation=False):
        self.model = "fake-model"
        self.calls = []
        self.invalid_citation = invalid_citation
        self.fail_generation = fail_generation

    def close(self):
        pass

    def ask(self, rules, payload, schema, stage, image=None):
        # Simulate the extraction phase output
        if stage == "extraction":
            if payload["page"] != 1 or not payload["text"]:
                return {"requirements": []}
            return {"requirements": [
                {"statement": "Debug access must be disabled in production mode",
                 "source_kind": "text", "source_quote": "Debug access is disabled in production mode.",
                 "measurable": True},
                {"statement": "Fabricated", "source_kind": "text",
                 "source_quote": "Not present in the PDF", "measurable": True}]}

        # Simulate network failure
        if self.fail_generation:
            raise ValueError("Simulated API response failure")

        test = proposed_test()
        # Test validation logic by deliberately feeding it a bad citation
        if self.invalid_citation:
            test["citations"][0]["quote"] = "Invented quote"
        return test


class WorkflowTests(unittest.TestCase):
    def test_symbols_and_lines_preserved(self):
        # Ensure regex formatting does not destroy essential engineering symbols.
        text = "VDD ≤ 3.3 V ±5%\nRegister [7:0] = 0xFF; T ≥ -40°C"
        self.assertEqual(PDFProcessor.clean_text(text), text)

    def test_long_sentence_is_bounded_with_exact_offsets(self):
        # Ensure chunks don't overflow the token limit.
        text = "x" * 2000
        chunks = create_chunks(text, 1, Tokenizer(), 80, overlap=10)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(c.content) <= 80 for c in chunks))
        self.assertTrue(all(c.content == text[c.char_start:c.char_end] for c in chunks))
        self.assertEqual(chunks[-1].char_end, len(text))

    def test_blank_page_detection_and_rendering(self):
        # Verify empty/image pages process correctly
        with PDFProcessor(pdf_bytes(blank=True)) as doc:
            page = doc.page(0, True)
        self.assertEqual(page["text"], "")
        self.assertTrue(page["jpeg"].startswith(b"\xff\xd8")) # JPEG magic bytes
        with self.assertRaisesRegex(ValueError, "page limit"):
            PDFProcessor(pdf_bytes(), 1)

    def test_faiss_search_retains_page_and_raw_score(self):
        # Test the vector search interface logic.
        store = VectorStore(Embedder())
        store.build_index([{"number": 7, "text": "Debug production"}], [])
        hit = store.search("Debug production", 4)[0]
        self.assertEqual(hit["page"], 7)
        self.assertAlmostEqual(hit["score"], 1.0, places=5)

    def test_automated_plan_and_literal_evidence(self):
        # Run a full simulated end-to-end extraction and test generation
        result = FakePlanner().generate(pdf_bytes(), "manual.pdf", Embedder())
        self.assertEqual(result["status"], "completed")
        self.assertEqual(len(result["requirements"]), 1)
        self.assertEqual(result["tests"][0]["approval"], "pending")
        self.assertEqual(result["tests"][0]["citations"][0]["page"], 1)
        self.assertIn("Discarded unmatched quote", result["warnings"][0])
        # The exported record must remain plain JSON for later evaluation.
        self.assertEqual(json.loads(json.dumps(result))["sha256"], result["sha256"])
        self.assertIn("Source page 1", render_report(result))

    def test_bad_citation_becomes_explicit_abstention(self):
        # Test that tests fall back to 'insufficient_information' if citations fail validation.
        result = FakePlanner(invalid_citation=True).generate(pdf_bytes(), "manual.pdf", Embedder())
        test = result["tests"][0]
        self.assertEqual(test["status"], "insufficient_information")
        self.assertEqual(test["steps"], [])
        self.assertIsNone(test["pass_criteria"])

    def test_zero_retrieval_needs_no_embedder(self):
        # When top_k = 0, RAG is bypassed.
        result = FakePlanner().generate(pdf_bytes(), "manual.pdf", None, top_k=0)
        self.assertEqual(result["status"], "completed")
        self.assertEqual([e["reference_id"] for e in result["tests"][0]["supplied_evidence"]], ["S1"])
        self.assertEqual(result["metrics"]["retrieval_seconds"], 0)

    def test_incomplete_run_preserves_requirements(self):
        # Ensure progress is saved even if the LLM crashes midway.
        result = FakePlanner(fail_generation=True).generate(pdf_bytes(), "manual.pdf", Embedder())
        self.assertEqual(result["status"], "failed")
        self.assertEqual(len(result["requirements"]), 1)
        self.assertIn("Simulated", result["error"])

    def test_unknown_reference_rejected(self):
        # Test model validation throwing an error for an unknown evidence reference
        draft = TestDraft.model_validate(proposed_test())
        with self.assertRaises(ValueError):
            validate_citations(draft, [{"reference_id": "R1", "text": "unrelated", "page": 1, "kind": "text"}])

    def test_abstention_clears_unverified_steps(self):
        # Test that missing information nullifies any hallucinated steps
        data = proposed_test()
        data.update(status="insufficient_information", missing_information=["Interface not specified"])
        result = TestDraft.model_validate(data)
        self.assertEqual(result.steps, [])
        self.assertIsNone(result.fail_criteria)

    def test_visual_evidence_stays_flagged_for_review(self):
        # Ensure UI flagging of image-based assertions is functional.
        class VisualPlanner(FakePlanner):
            def ask(self, rules, payload, schema, stage, image=None):
                if stage == "extraction":
                    return {"requirements": [{"statement": "Debug lock observed",
                            "source_quote": "DEBUG LOCK = 1", "source_kind": "visual",
                            "measurable": True}]} if payload["page"] == 1 else {"requirements": []}
                test = proposed_test()
                test["citations"] = [{"reference_id": "V1", "quote": "DEBUG LOCK = 1"}]
                return test
        result = VisualPlanner().generate(pdf_bytes(blank=True), "scan.pdf", Embedder())
        self.assertEqual(result["tests"][0]["citations"][0]["source_kind"], "visual")
        self.assertTrue(result["tests"][0]["visual_review_required"])


if __name__ == "__main__":
    unittest.main()