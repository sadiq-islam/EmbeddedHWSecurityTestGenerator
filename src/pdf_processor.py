"""Parse PDF structure dynamically using Docling's AI Layout Analysis and RapidOCR."""

from io import BytesIO
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions, RapidOcrOptions
from docling.datamodel.base_models import InputFormat, DocumentStream
from docling.chunking import HierarchicalChunker
import pymupdf

class PDFProcessor:
    def __init__(self, pdf_bytes: bytes, max_pages: int = 100):
        self.pdf_bytes = pdf_bytes
        self.max_pages = max_pages

        # Configure Docling for maximum universal robustness
        pipeline_options = PdfPipelineOptions()

        # 1. Enable OCR for any scanned documents or text trapped in images
        pipeline_options.do_ocr = True
        pipeline_options.ocr_options = RapidOcrOptions()

        # 2. Enable AI Table Structure Recognition (TSR) to handle complex grids
        pipeline_options.do_table_structure = True

        # 3. Enable AI Layout Analysis to dynamically determine reading order
        # This automatically handles multi-column documents and 2-up booklets
        # without needing manual slicing or aspect-ratio heuristics.
        pipeline_options.generate_page_images = True

        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def extract_document(self):
        """Converts the PDF to a structured DoclingDocument and creates semantic chunks."""
        # Enforce page limits to prevent API/memory overload
        with pymupdf.open(stream=self.pdf_bytes, filetype="pdf") as temp_pdf:
            total_pages = temp_pdf.page_count
            if total_pages > self.max_pages:
                raise ValueError(f"PDF has {total_pages} pages; increase the page limit ({self.max_pages})")

        # Pass the raw PDF stream to the AI parser
        buf = BytesIO(self.pdf_bytes)
        stream = DocumentStream(name="uploaded_spec.pdf", stream=buf)
        result = self.converter.convert(stream)

        # Group text semantically based on detected headers rather than fixed token limits
        chunker = HierarchicalChunker()
        doc_chunks = list(chunker.chunk(result.document))

        return result.document, doc_chunks, total_pages