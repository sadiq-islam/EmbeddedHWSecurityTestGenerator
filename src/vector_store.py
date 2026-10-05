"""One local MiniLM encoder and exact FAISS cosine search over bounded text chunks."""

import numpy as np
from src.pdf_processor import create_chunks

# The embedding model used for similarity search
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def load_embedder():
    # Imported only when needed; Streamlit caches this model across reruns.
    # Prevents slow imports at script start time.
    from sentence_transformers import SentenceTransformer
    # Force CPU to avoid complex local GPU configuration for a small model
    return SentenceTransformer(MODEL_NAME, device="cpu")


class VectorStore:
    def __init__(self, model):
        self.model = model
        self.chunks = []
        self.index = None

    def build_index(self, pages: list[dict], requirements: list[dict]):
        # FAISS is used for efficient vector similarity search
        import faiss
        self.chunks = []

        # Reserve tokenizer special tokens so the encoder never silently truncates chunks.
        budget = self.model.max_seq_length - self.model.tokenizer.num_special_tokens_to_add(False)

        # Chunk all text pages
        for page in pages:
            self.chunks.extend(create_chunks(page["text"], page["number"],
                                            self.model.tokenizer, budget))

        # Also chunk any visual quotes that were extracted during the initial run
        for req in requirements:
            if req["source_kind"] == "visual":
                # Image-derived quotes remain explicitly marked as unverified transcriptions.
                self.chunks.extend(create_chunks(req["source_quote"], req["page"],
                                                self.model.tokenizer, budget, "visual"))

        if not self.chunks:
            raise ValueError("No usable evidence to index. Enable image reading for scanned PDFs.")

        # Embed all chunks into dense vectors
        vectors = self.model.encode([c.content for c in self.chunks], batch_size=32,
                                    normalize_embeddings=True, convert_to_numpy=True,
                                    show_progress_bar=False).astype(np.float32)

        # Initialize an Inner Product index (equivalent to Cosine Similarity since vectors are normalized)
        self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(vectors)

    def search(self, query: str, top_k: int = 4) -> list[dict]:
        if self.index is None:
            raise ValueError("Build the index before searching")

        # Search once per requirement; there is no separate duplicate relevance search.
        # Embed the search query
        vector = self.model.encode([query], normalize_embeddings=True,
                                   convert_to_numpy=True, show_progress_bar=False).astype(np.float32)

        # Query the FAISS index to find the closest `top_k` matches
        scores, indices = self.index.search(vector, min(top_k, len(self.chunks)))

        # Map the results back to the original text chunks and scores
        return [{"page": self.chunks[i].page_number, "text": self.chunks[i].content,
                 "kind": self.chunks[i].source_kind, "score": float(score)}
                for score, i in zip(scores[0], indices[0]) if i >= 0]