"""Exercise the upload/generate UI using a fake API and local test embeddings."""
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from streamlit.testing.v1 import AppTest
from test_workflow import FakePlanner, Embedder, pdf_bytes


class InterfaceTests(unittest.TestCase):
    def test_upload_generate_and_remove_document(self):
        data = pdf_bytes()
        upload = SimpleNamespace(name="manual.pdf", getvalue=lambda: data)
        # All inference is replaced. This makes no network calls or model downloads.
        with patch.dict(os.environ, {"GROQ_API_KEY": "test-key"}), \
                patch("streamlit.file_uploader", return_value=upload) as uploader, \
                patch("src.vector_store.load_embedder", return_value=Embedder()), \
                patch("src.chatbot.HardwarePlanner", return_value=FakePlanner()):
            app = AppTest.from_file(str(Path(__file__).parents[1] / "app.py"), default_timeout=20).run()
            self.assertFalse(app.exception)
            self.assertFalse(app.button[0].disabled)
            app.button[0].click().run()
            self.assertFalse(app.exception)
            self.assertEqual(app.session_state["plan"]["status"], "completed")
            self.assertEqual([m.value for m in app.metric], ["1", "1", "0"])
            uploader.return_value = None
            app.run()
            self.assertFalse(app.exception)
            self.assertNotIn("plan", app.session_state)


if __name__ == "__main__":
    unittest.main()
