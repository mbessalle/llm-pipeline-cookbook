"""Code examples from Chapter 03: chunking-strategies"""

# --- Example 1 ---
def fixed_size_chunks(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap
    
    return chunks

# --- Example 2 ---
import nltk
nltk.download('punkt')

def sentence_chunks(text: str, max_tokens: int = 500) -> list[str]:
    sentences = nltk.sent_tokenize(text)
    
    chunks = []
    current_chunk = []
    current_tokens = 0
    
    for sentence in sentences:
        sentence_tokens = len(sentence.split())  # rough count, close enough
        
        if current_tokens + sentence_tokens > max_tokens and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_tokens = 0
        
        current_chunk.append(sentence)
        current_tokens += sentence_tokens
    
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks

# --- Example 3 ---
import re

def semantic_chunks(text: str, max_tokens: int = 500) -> list[str]:
    # Split on paragraph breaks or markdown headers
    pattern = r'\n\n+|(?=^#{1,3}\s)'
    segments = re.split(pattern, text, flags=re.MULTILINE)
    segments = [s.strip() for s in segments if s.strip()]
    
    chunks = []
    current_chunk = []
    current_tokens = 0
    
    for segment in segments:
        segment_tokens = len(segment.split())
        
        # Oversized segment gets its own treatment
        if segment_tokens > max_tokens:
            if current_chunk:
                chunks.append("\n\n".join(current_chunk))
                current_chunk = []
                current_tokens = 0
            sub_chunks = sentence_chunks(segment, max_tokens)
            chunks.extend(sub_chunks)
            continue
        
        if current_tokens + segment_tokens > max_tokens and current_chunk:
            chunks.append("\n\n".join(current_chunk))
            current_chunk = []
            current_tokens = 0
        
        current_chunk.append(segment)
        current_tokens += segment_tokens
    
    if current_chunk:
        chunks.append("\n\n".join(current_chunk))
    
    return chunks

# --- Example 4 ---
from dataclasses import dataclass

@dataclass
class Chunk:
    text: str
    metadata: dict
    index: int

def structured_chunks(document: dict) -> list[Chunk]:
    chunks = []
    
    for section_idx, section in enumerate(document["sections"]):
        section_header = section.get("header", "")
        
        for para_idx, paragraph in enumerate(section["paragraphs"]):
            chunk = Chunk(
                text=paragraph,
                metadata={
                    "document_id": document["id"],
                    "section": section_header,
                    "section_index": section_idx,
                    "paragraph_index": para_idx,
                    "prev_section": document["sections"][section_idx - 1]["header"] if section_idx > 0 else None,
                    "next_section": document["sections"][section_idx + 1]["header"] if section_idx < len(document["sections"]) - 1 else None,
                },
                index=len(chunks)
            )
            chunks.append(chunk)
    
    return chunks

# --- Example 5 ---
def table_to_text(table: list[list[str]]) -> str:
    if not table:
        return ""
    
    headers = table[0]
    rows = table[1:]
    
    text_rows = []
    for row in rows:
        row_text = "; ".join(
            f"{headers[i]}: {cell}" 
            for i, cell in enumerate(row) 
            if i < len(headers)
        )
        text_rows.append(row_text)
    
    return "\n".join(text_rows)

# | Name | Age | City |
# | John | 30  | NYC  |
# becomes: "Name: John; Age: 30; City: NYC"

# --- Example 6 ---
def chunk_with_tables(text: str, tables: list[dict], max_tokens: int = 500) -> list[Chunk]:
    chunks = []
    
    # Tables get their own chunks, no splitting
    for table in tables:
        table_text = format_table_markdown(table["data"])
        chunks.append(Chunk(
            text=table_text,
            metadata={"type": "table", "table_id": table["id"]},
            index=len(chunks)
        ))
    
    # Everything else gets normal chunking
    non_table_text = remove_table_placeholders(text)
    text_chunks = semantic_chunks(non_table_text, max_tokens)
    
    for chunk_text in text_chunks:
        chunks.append(Chunk(
            text=chunk_text,
            metadata={"type": "text"},
            index=len(chunks)
        ))
    
    return chunks

# --- Example 7 ---
def chunks_with_smart_overlap(text: str, chunk_size: int = 500, overlap_size: int = 50) -> list[Chunk]:
    sentences = nltk.sent_tokenize(text)
    
    chunks = []
    current_chunk = []
    current_tokens = 0
    overlap_sentences = []
    
    for sentence in sentences:
        tokens = len(sentence.split())
        
        if current_tokens + tokens > chunk_size and current_chunk:
            chunks.append(Chunk(
                text=" ".join(current_chunk),
                metadata={"overlap_from_prev": len(overlap_sentences) > 0},
                index=len(chunks)
            ))
            
            # Keep the last few sentences as overlap
            overlap_sentences = []
            overlap_tokens = 0
            for s in reversed(current_chunk):
                s_tokens = len(s.split())
                if overlap_tokens + s_tokens > overlap_size:
                    break
                overlap_sentences.insert(0, s)
                overlap_tokens += s_tokens
            
            current_chunk = overlap_sentences.copy()
            current_tokens = overlap_tokens
        
        current_chunk.append(sentence)
        current_tokens += tokens
    
    if current_chunk:
        chunks.append(Chunk(
            text=" ".join(current_chunk),
            metadata={"overlap_from_prev": len(overlap_sentences) > 0},
            index=len(chunks)
        ))
    
    return chunks

# --- Example 8 ---
class ChunkingPipeline:
    def __init__(self, config: ChunkConfig):
        self.max_tokens = config.max_tokens
        self.overlap = config.overlap
        self.preserve_tables = config.preserve_tables
    
    def process(self, document: Document) -> list[Chunk]:
        chunks = []
        
        if self.preserve_tables and document.tables:
            for table in document.tables:
                chunks.append(self._table_chunk(table, document))
        
        text_chunks = semantic_chunks(
            document.text, 
            max_tokens=self.max_tokens
        )
        
        text_chunks = self._add_overlap(text_chunks)
        
        for i, text in enumerate(text_chunks):
            chunks.append(Chunk(
                text=text,
                metadata={
                    "document_id": document.id,
                    "source": document.source_path,
                    "chunk_index": i,
                    "total_chunks": len(text_chunks),
                    **document.metadata
                },
                index=len(chunks)
            ))
        
        return chunks

