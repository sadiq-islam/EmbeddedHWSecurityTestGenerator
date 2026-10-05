"""Read PDF text without deleting technical symbols; render images only when needed."""

from dataclasses import dataclass
import re
import pymupdf


@dataclass
class TextChunk:
    content: str
    page_number: int
    chunk_index: int
    char_start: int
    char_end: int
    source_kind: str = "text"


def normalize(text: str) -> str:
    """Ignore whitespace differences only when checking a literal source quote."""
    return re.sub(r"\s+", " ", text).strip()


class PDFProcessor:
    def __init__(self, pdf_bytes: bytes, max_pages: int = 100):
        self.pdf = pymupdf.open(stream=pdf_bytes, filetype="pdf")
        if self.pdf.needs_pass or not self.pdf.page_count:
            self.pdf.close()
            raise ValueError("Upload an unlocked PDF with at least one page")
        if self.pdf.page_count > max_pages:
            count = self.pdf.page_count
            self.pdf.close()
            raise ValueError(f"PDF has {count} pages; increase the page limit ({max_pages})")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.pdf.close()

    @staticmethod
    def clean_text(text: str) -> str:
        # Preserve newlines, Unicode, signs, units, brackets and numeric boundaries.
        lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]
        return "\n".join(lines).strip()

    def page(self, index: int, include_image: bool) -> dict:
        page = self.pdf.load_page(index)
        text = self.clean_text(page.get_text("text", sort=True))
        if len(text) > 24000:
            raise ValueError(f"Page {index + 1} exceeds the 24,000-character extraction limit")
        jpeg = None
        if include_image:
            # Render one bounded image at a time to keep laptop memory modest.
            scale = min(1.5, 1600 / max(page.rect.width, page.rect.height))
            jpeg = page.get_pixmap(matrix=pymupdf.Matrix(scale, scale),
                                   colorspace=pymupdf.csRGB, alpha=False).tobytes("jpeg")
        return {"number": index + 1, "text": text, "jpeg": jpeg}


def create_chunks(text: str, page: int, tokenizer, token_budget: int,
                  source_kind: str = "text", overlap: int = 32) -> list[TextChunk]:
    """Use tokenizer offsets so long sentences cannot overflow the embedding window."""
    if not 0 <= overlap < token_budget:
        raise ValueError("overlap must be smaller than token_budget")
    if not text.strip():
        return []
    offsets = tokenizer(text, add_special_tokens=False, truncation=False,
                        return_offsets_mapping=True)["offset_mapping"]
    chunks = []
    start = 0
    while start < len(offsets):
        end = min(start + token_budget, len(offsets))
        char_start, char_end = offsets[start][0], offsets[end - 1][1]
        piece = text[char_start:char_end]
        # Retokenizing a substring can introduce boundary tokens; shrink if needed.
        while len(tokenizer.encode(piece, add_special_tokens=False)) > token_budget:
            end -= 1
            if end <= start:
                raise ValueError("Could not create a bounded text chunk")
            char_end = offsets[end - 1][1]
            piece = text[char_start:char_end]
        chunks.append(TextChunk(piece, page, len(chunks), char_start, char_end, source_kind))
        if end == len(offsets):
            break
        start = max(start + 1, end - overlap)
    return chunks