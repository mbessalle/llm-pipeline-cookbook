# Chapter 3: Chunking Strategies

Chunking is one of those things that sounds trivial until you get it wrong. "Just split the text into pieces" -- sure, and just split a novel into random 500-word blocks and see how much sense it makes. I spent two weeks debugging retrieval quality issues before realizing the problem wasn't my embeddings or my prompts. It was my chunking. Garbage chunks in, garbage retrieval out.

The fundamental tension: LLMs have context windows. Your document collection doesn't care about your context window. Something has to give, and that something is your chunking strategy.

---

## The Tradeoff Nobody Tells You About

Chunks that are too small lose context. A 50-token fragment about "the termination clause in section 4.2" means nothing without sections 4.0 and 4.1. Chunks that are too big dilute relevance -- when a user asks about termination clauses, you don't want to return a 10,000-token chunk covering the entire contract. You want the specific section.

And then there's the boundary problem. Split at the wrong point and you get a chunk that ends mid-sentence. Or worse, you split a table in half. Half a table is worse than no table at all because the LLM will try to interpret it and hallucinate the missing data.

I've settled on a target of 300-500 tokens per chunk for most of our use cases. But honestly, the right size depends on your data. Legal documents with dense cross-references need bigger chunks. Log files with independent entries can go smaller.

---

## Fixed-Size: Quick and Dirty

If you're prototyping and just need something working, fixed-size chunks are fine. Not good, but fine.

```python
def fixed_size_chunks(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap
    
    return chunks
```

This will cut mid-sentence, mid-word even. It doesn't know or care about your document structure. I used this for exactly one prototype and then immediately replaced it. But it let me validate the rest of the pipeline in an afternoon, which was the point.

---

## Sentence-Based: The Minimum Viable Approach

At bare minimum, respect sentence boundaries. It's not that much more code and the quality difference is significant.

```python
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
```

Better. But it still doesn't understand your document. A paragraph about budget allocation gets merged with the next paragraph about project timelines just because they happen to be adjacent and under the token limit.

---

## Semantic Chunking: What I Actually Use

Split on natural boundaries -- paragraph breaks, section headers, thematic shifts. This is what makes retrieval actually work well.

```python
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
```

The key insight that took me too long to figure out: when a single segment exceeds your token limit, don't just truncate it. Fall back to sentence-level splitting for that segment only. The rest of your document keeps its semantic boundaries intact.

---

## Don't Throw Away Structure

This is probably the single most impactful thing I changed in our pipeline. When you chunk a document, you lose the surrounding context. But you can preserve it in metadata.

```python
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
```

Even if the chunk text is just a paragraph about zoning requirements, the metadata tells you it came from "Section 3: Land Use Regulations" and the next section is "Section 4: Building Permits." That context is gold during retrieval. When two chunks score similarly on embedding similarity, the metadata lets you pick the more relevant one.

We started storing section headers in metadata about three months in. Retrieval precision jumped noticeably. Should've done it from day one.

---

## Tables: The Chunking Nightmare

I have a special hatred for tables in documents. They carry dense, structured information that completely falls apart when you split them.

You have two options, and neither is perfect.

**Option 1: Linearize the table into text.** Each row becomes a sentence.

```python
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
```

Works okay for simple tables. Breaks down when you have a 50-row table with 12 columns.

**Option 2: Keep tables as single chunks,** even if they exceed your token limit.

This is what I do. A table is a semantic unit. Splitting it destroys meaning. Yes, sometimes you end up with a 2000-token chunk, and that's fine. The alternative -- half a table -- is worse in every way.

```python
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
```

---

## Overlap: Cheap Insurance

Chunk overlap means repeating some text at the boundaries between chunks. Think of it like the way book chapters sometimes recap the end of the previous chapter.

I use 10-15% overlap. So for a 500-token chunk, the last 50-75 tokens also appear at the start of the next chunk. It's wasteful in terms of storage and embedding cost, but it prevents the scenario where a key piece of information sits right at a chunk boundary and gets split between two chunks that individually make less sense.

The trick is to overlap at sentence boundaries, not arbitrary character positions:

```python
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
```

---

## Putting It Together

In production, our chunking pipeline looks roughly like this:

```python
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
```

Nothing fancy. Semantic splitting, table preservation, overlap, rich metadata. It handles about 97% of our documents without issues. The remaining 3% are edge cases we handle with document-specific extractors -- but that's a different problem.

---

*Next: now that you have chunks, you need to turn them into something searchable. Embeddings and retrieval is where things get interesting.*
