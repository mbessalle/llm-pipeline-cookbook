# Chapter 2: Document Ingestion

Here's a fun exercise: go ask an LLM tutorial what a "document" looks like. You'll get a nice clean markdown string, maybe a well-formatted JSON blob. Neat, tidy, ready to embed.

Now open your actual production inbox. You've got PDFs with embedded images from a scanner that was new in 2004. Word documents with seventeen levels of tracked changes. An Excel file where someone used merged cells as a layout tool. And my personal favorite -- a file with a .pdf extension that's actually just a JPEG someone renamed.

Welcome to document ingestion. This is where your pipeline meets reality, and reality doesn't care about your type hints.

---

## PDF Extraction: Pick Your Pain

PDFs are everywhere. They're also one of the worst formats ever designed for text extraction -- a PDF is really a set of instructions for drawing characters on a page, not a structured document. But we work with what we have.

### PyMuPDF -- The Fast One

This is my default. It's written in C under the hood, so it's quick, and it handles straightforward text-heavy PDFs without drama.

```python
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
```

It's fast. It has no external dependencies beyond the pip install. But it falls apart on complex layouts -- multi-column documents, forms with text boxes, anything where the visual layout doesn't match the reading order. I'd say it handles about 70% of the PDFs I encounter without issues.

### pdfplumber -- The Table Whisperer

When I need to extract tables, this is what I reach for. The table detection is genuinely good -- it understands cell boundaries, merged cells, the works.

```python
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
```

The downside is speed. On a batch of 500 documents, pdfplumber took about 4x longer than PyMuPDF. That adds up when you're processing thousands of files. And it eats memory on large PDFs -- I've seen it consume 2GB on a single 300-page document.

### Unstructured -- The Kitchen Sink

When nothing else works, there's Unstructured. It tries to handle everything: images in PDFs, layout detection, table inference, you name it.

```python
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
```

I won't sugarcoat it -- the dependency tree is gnarly. Expect to spend an afternoon getting it installed properly, especially if you need the hi_res strategy which pulls in detectron2 and a bunch of ML models. But for genuinely difficult documents, it's the best option I've found.

### What I Actually Do in Production

Tiered extraction. Try the fast thing first. If the output looks bad, escalate.

```python
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
```

That quality_score check is crude -- we basically look at the ratio of actual words vs garbage characters, check if the text length is reasonable for the page count, and flag anything suspicious. Not perfect, but it catches the obvious failures and saves us from running expensive extraction on documents that don't need it.

---

## OCR: When Your PDFs Are Just Pictures

Scanned documents. The bane of my existence. You open the PDF, it looks like text, you try to extract it and get an empty string back. Because it's an image of text. Someone printed a document, scanned it back in, and emailed you the result.

### Tesseract for the Common Cases

Open source, works surprisingly well for clean scans, falls apart on anything degraded.

```python
import pytesseract
from pdf2image import convert_from_path

def ocr_pdf(file_path: str) -> str:
    images = convert_from_path(file_path, dpi=300)  # 300 DPI matters -- don't go lower
    
    text_parts = []
    for i, image in enumerate(images):
        text = pytesseract.image_to_string(image)
        text_parts.append(f"[Page {i + 1}]\n{text}")
    
    return "\n\n".join(text_parts)
```

Install with `apt install tesseract-ocr` plus whatever language packs you need. For Dutch documents I also need `tesseract-ocr-nld`. The language pack thing tripped me up for an embarrassing amount of time -- Tesseract defaults to English and just silently produces garbage on non-English text.

### Cloud OCR When Accuracy Matters

For documents where getting the text wrong has actual consequences -- legal documents, contracts, anything someone might sue over -- I send it to Google Cloud Vision or AWS Textract. The accuracy difference is noticeable, especially on degraded scans, handwriting, or mixed-language documents.

```python
from google.cloud import vision

def ocr_with_google(file_path: str) -> str:
    client = vision.ImageAnnotatorClient()
    
    with open(file_path, "rb") as f:
        content = f.read()
    
    image = vision.Image(content=content)
    response = client.document_text_detection(image=image)
    
    return response.full_text_annotation.text
```

At about $1.50 per 1000 pages, it's not free, but it's cheap insurance against bad data flowing into your pipeline. Bad OCR means bad chunks means bad embeddings means bad retrieval. The error compounds downstream.

### The Hybrid I Actually Run

```python
def smart_ocr(file_path: str) -> str:
    result = ocr_tesseract(file_path)
    
    # Simple quality gate
    if len(result) < 100 or result.count("\ufffd") > 10:
        return ocr_google(file_path)
    
    return result
```

Tesseract handles maybe 80% of our scanned docs fine. The remaining 20% go to Google. Net cost is much lower than sending everything to the cloud.

---

## Handling Multiple Formats Without Losing Your Mind

In production you'll get PDFs, Word docs, plain text, HTML, maybe even the occasional PowerPoint. The cleanest pattern I've found is a simple extractor registry:

```python
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
```

Nothing revolutionary. But having that Protocol interface means adding a new format is just writing a new class that implements can_handle and extract. When we needed to support .msg (Outlook email) files last month, it took about 40 minutes to add.

---

## Don't Throw Away Metadata

This is a mistake I made early and regretted later. We were extracting text from documents and discarding everything else -- filename, creation date, page count, author. Then a user asked "show me all documents from before 2020" and we had no way to answer that without re-processing everything.

```python
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
```

Store this alongside your document content. You will need it eventually, and re-ingesting thousands of documents just to grab metadata is not a fun afternoon.

---

## Documents Will Fail. Plan For It.

I process about ten thousand documents a month. On any given batch, somewhere between 2-5% will fail extraction. Corrupted files, password-protected PDFs, formats we don't support, files that claim to be PDFs but aren't. You can't prevent this. You can handle it gracefully.

```python
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
```

The important bit: log the failures with enough context to debug them later. File path, error message, stack trace. We review failed ingestions weekly and either fix the extractor or add the file to a "known unsupported" list. Over time, the failure rate has dropped from about 8% to under 3%.

---

*Next up: you've got raw text. Now you need to chop it into pieces an LLM can actually work with. Chunking is where the subtlety lives.*
