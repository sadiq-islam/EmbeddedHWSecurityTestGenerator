"""One local MiniLM encoder, FAISS cosine search, and Cross-Encoder Re-ranking."""

import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

def load_embedder():
    return SentenceTransformer(MODEL_NAME, device="cpu")

def load_reranker():
    return CrossEncoder(RERANKER_NAME, device="cpu")

class VectorStore:
    def __init__(self, model):
        self.model = model
        self.reranker = load_reranker()
        self.chunks = []
        self.index = None

    def build_index(self, doc_chunks):
        import faiss
        self.chunks = []

        for chunk in doc_chunks:
            page_num = 1
            if chunk.meta.doc_items and len(chunk.meta.doc_items) > 0:
                prov = chunk.meta.doc_items[0].prov
                if prov and len(prov) > 0:
                    page_num = prov[0].page_no

            # --- HEADER INJECTION ---
            # Extract the hierarchical headings Docling found for this specific chunk
            headings = chunk.meta.headings if hasattr(chunk.meta, 'headings') else []
            heading_prefix = ""
            if headings:
                # Prepend the structural path (e.g., "Section: Main Features > Mode 3")
                heading_prefix = f"Section: {' > '.join(headings)}\n\n"

            # Combine the heading context with the raw paragraph
            enriched_text = f"{heading_prefix}{chunk.text}"

            self.chunks.append({
                "content": enriched_text,
                "page_number": page_num,
                "source_kind": "text"
            })

        if not self.chunks:
            raise ValueError("No usable text found to index.")

        vectors = self.model.encode([c["content"] for c in self.chunks], batch_size=32,
                                    normalize_embeddings=True, convert_to_numpy=True,
                                    show_progress_bar=False).astype(np.float32)

        self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(vectors)

    def search(self, query: str, final_top_k: int = 4, initial_fetch: int = 15) -> list[dict]:
        """Performs a broad dense search, then re-ranks the results for exact relevance."""
        if self.index is None:
            raise ValueError("Build the index before searching")

        vector = self.model.encode([query], normalize_embeddings=True,
                                   convert_to_numpy=True, show_progress_bar=False).astype(np.float32)

        fetch_count = min(initial_fetch, len(self.chunks))
        scores, indices = self.index.search(vector, fetch_count)

        initial_hits = [{"page": self.chunks[i]["page_number"], "text": self.chunks[i]["content"],
                         "kind": self.chunks[i]["source_kind"], "initial_score": float(score)}
                        for score, i in zip(scores[0], indices[0]) if i >= 0]

        if not initial_hits:
            return []

        pairs = [[query, hit["text"]] for hit in initial_hits]
        rerank_scores = self.reranker.predict(pairs)

        for idx, hit in enumerate(initial_hits):
            hit["score"] = float(rerank_scores[idx])

        reranked_hits = sorted(initial_hits, key=lambda x: x["score"], reverse=True)
        return reranked_hits[:final_top_k]