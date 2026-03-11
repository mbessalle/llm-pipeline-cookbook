"""Code examples from Chapter 02: document-ingestion"""

# --- Example 1 ---
import fitz  # PyMuPDF -- yes, the import name is different from the package name

def extract_pdf_pymupdf(file_path: str) -> str:
    doc = fitz.open(file_path)
    text_parts = []
    
    for page_num, page in enumerate(doc):
        text = page.get_text("text")
        if text.strip():
            text_parts.append(f"[Page {page_num + 1}]\n{text}")
    
    doc.close()
    return "\n\n".join(text_parts)

# --- Example 2 ---
import pdfplumber

def extract_pdf_pdfplumber(file_path: str) -> dict:
    results = {"text": [], "tables": []}
    
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                results["text"].append(text)
            
            tables = page.extract_tables()
            for table in tables:
                results["tables"].append(table)
    
    return results

# --- Example 3 ---
from unstructured.partition.pdf import partition_pdf

def extract_pdf_unstructured(file_path: str) -> list:
    elements = partition_pdf(
        filename=file_path,
        strategy="hi_res",  # "fast" if you're impatient
        infer_table_structure=True,
        extract_images_in_pdf=True,
    )
    
    return [
        {
            "type": element.category,
            "text": element.text,
            "metadata": element.metadata.to_dict()
        }
        for element in elements
    ]

# --- Example 4 ---
def extract_document(file_path: str) -> ExtractionResult:
    # Fast pass
    result = try_pymupdf(file_path)
    
    if result.quality_score > 0.8:
        return result
    
    # Tables detected? Use the specialist
    if has_tables(result):
        return try_pdfplumber(file_path)
    
    # Nothing else worked -- bring in the heavy machinery
    return try_unstructured(file_path)

# --- Example 5 ---
import pytesseract
from pdf2image import convert_from_path

def ocr_pdf(file_path: str) -> str:
    images = convert_from_path(file_path, dpi=300)  # 300 DPI matters -- don't go lower
    
    text_parts = []
    for i, image in enumerate(images):
        text = pytesseract.image_to_string(image)
        text_parts.append(f"[Page {i + 1}]\n{text}")
    
    return "\n\n".join(text_parts)

# --- Example 6 ---
from google.cloud import vision

def ocr_with_google(file_path: str) -> str:
    client = vision.ImageAnnotatorClient()
    
    with open(file_path, "rb") as f:
        content = f.read()
    
    image = vision.Image(content=content)
    response = client.document_text_detection(image=image)
    
    return response.full_text_annotation.text

# --- Example 7 ---
def smart_ocr(file_path: str) -> str:
    result = ocr_tesseract(file_path)
    
    # Simple quality gate
    if len(result) < 100 or result.count("\ufffd") > 10:
        return ocr_google(file_path)
    
    return result

# --- Example 8 ---
from pathlib import Path
from dataclasses import dataclass
from typing import Protocol

@dataclass
class Document:
    content: str
    metadata: dict
    source_path: str
    format: str

class Extractor(Protocol):
    def can_handle(self, file_path: Path) -> bool: ...
    def extract(self, file_path: Path) -> Document: ...

class IngestionPipeline:
    def __init__(self):
        self.extractors = [
            PDFExtractor(),
            WordExtractor(),
            HTMLExtractor(),
            PlainTextExtractor(),  # fallback -- always last
        ]
    
    def ingest(self, file_path: str) -> Document:
        path = Path(file_path)
        
        for extractor in self.extractors:
            if extractor.can_handle(path):
                return extractor.extract(path)
        
        raise ValueError(f"No extractor for {path.suffix}")

# --- Example 9 ---
def extract_metadata(file_path: str) -> dict:
    path = Path(file_path)
    stat = path.stat()
    
    metadata = {
        "filename": path.name,
        "extension": path.suffix,
        "size_bytes": stat.st_size,
        "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
    }
    
    if path.suffix.lower() == ".pdf":
        with fitz.open(file_path) as doc:
            pdf_meta = doc.metadata
            metadata.update({
                "title": pdf_meta.get("title"),
                "author": pdf_meta.get("author"),
                "pages": len(doc),
            })
    
    return metadata

# --- Example 10 ---
@dataclass
class IngestionResult:
    success: bool
    document: Document | None
    error: str | None
    file_path: str

def batch_ingest(pipeline: IngestionPipeline, files: list[str]) -> dict:
    results = {"success": [], "failed": []}
    
    for file_path in files:
        try:
            doc = pipeline.ingest(file_path)
            results["success"].append(IngestionResult(True, doc, None, file_path))
        except Exception as e:
            log.error(f"Failed: {file_path}: {e}")
            results["failed"].append(IngestionResult(False, None, str(e), file_path))
    
    success_rate = len(results["success"]) / len(files) * 100
    log.info(f"Batch complete: {success_rate:.1f}% success ({len(files)} total)")
    return results

