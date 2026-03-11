# Chapter 4: Embedding & Retrieval

You've got chunks. Now you need to find the right ones when someone asks a question. This sounds straightforward -- turn text into vectors, do a similarity search, done. And the basic version is exactly that simple. But the gap between "works in a demo" and "works reliably on ten thousand documents" is where most of this chapter lives.

---

## The Basic Pipeline

Three steps. Indexing: you convert your chunks into embedding vectors and store them. Querying: a user asks something, you convert their question into a vector too. Retrieval: you find the chunks whose vectors are closest to the query vector and hand them to the LLM.

That's it. Everything else is optimization.

---

## Picking an Embedding Model

I spent way too long agonizing over this early on. Here's what I've learned: for most use cases, it barely matters which model you pick, as long as it's not terrible. The difference between a good embedding model and a great one is maybe 5% on retrieval benchmarks. The difference between good chunking and bad chunking is 40%.

That said, here's what I use:

### OpenAI Embeddings (My Default)

```python
from openai import OpenAI

client = OpenAI()

def embed_openai(texts: list[str], model: str = "text-embedding-3-small") -> list[list[float]]:
    response = client.embeddings.create(
        input=texts,
        model=model
    )
    return [item.embedding for item in response.data]
```

`text-embedding-3-small` costs $0.02 per million tokens. For our 10,000 documents per month, that's about 40 cents. Not worth optimizing away. I use this unless there's a specific reason not to.

`text-embedding-3-large` is 3x the cost and maybe 3-5% better quality. I've never needed it.

### When You Want Free

If you're running locally or cost is a real constraint:

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('BAAI/bge-large-en-v1.5')

def embed_local(texts: list[str]) -> list[list[float]]:
    return model.encode(texts).tolist()
```

`bge-large-en-v1.5` is probably the best open-source option right now. Quality is close to OpenAI's offerings. The tradeoff is you need a GPU to run it at reasonable speed, or you're waiting a long time on CPU.

`all-MiniLM-L6-v2` is the other common choice -- much smaller, much faster, noticeably worse quality. Fine for prototyping.

---

## Vector Stores: Where to Put Your Embeddings

### Chroma -- For Prototyping

```python
import chromadb

client = chromadb.Client()
collection = client.create_collection("documents")

collection.add(
    documents=["chunk 1 text", "chunk 2 text"],
    metadatas=[{"source": "doc1"}, {"source": "doc1"}],
    ids=["chunk1", "chunk2"]
)

results = collection.query(
    query_texts=["user question"],
    n_results=5
)
```

Zero setup. pip install and go. I use this when I'm testing a new idea and don't want to think about infrastructure. Don't use it in production -- it's in-memory by default, not particularly fast at scale, and the API has some quirks.

### pgvector -- If You Already Have Postgres

This is what we run in production. If your application already uses PostgreSQL (and most do), adding vector search is just an extension.

```python
# One-time setup
"""
CREATE EXTENSION vector;
CREATE TABLE chunks (
    id SERIAL PRIMARY KEY,
    content TEXT,
    embedding vector(1536),
    metadata JSONB
);
CREATE INDEX ON chunks USING ivfflat (embedding vector_cosine_ops);
"""

def store_chunk(conn, content: str, embedding: list[float], metadata: dict):
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO chunks (content, embedding, metadata) VALUES (%s, %s, %s)",
            (content, embedding, json.dumps(metadata))
        )
    conn.commit()

def search(conn, query_embedding: list[float], limit: int = 5) -> list[dict]:
    with conn.cursor() as cur:
        cur.execute("""
            SELECT content, metadata, 1 - (embedding <=> %s) as similarity
            FROM chunks
            ORDER BY embedding <=> %s
            LIMIT %s
        """, (query_embedding, query_embedding, limit))
        
        return [
            {"content": row[0], "metadata": row[1], "similarity": row[2]}
            for row in cur.fetchall()
        ]
```

The `<=>` operator is cosine distance. The ivfflat index makes searches fast -- we query across 80,000 chunks in about 15ms. The one gotcha: you need to rebuild the index periodically as you add data. We do it nightly.

### Pinecone, Qdrant, Weaviate -- Managed Options

If you don't want to manage infrastructure, Pinecone is the easiest to get started with. Qdrant is my pick if you want something self-hostable with great performance. Weaviate is interesting if you want built-in hybrid search.

I've used all three. For a team of one or two, pgvector is usually enough. You're adding a managed service (and its cost) to avoid a problem you might not have yet.

---

## Hybrid Search: Why Vectors Aren't Enough

Here's something that surprised me. Pure vector search is good at finding semantically similar content. But it's terrible at exact matches. A user asks "what does section 4.2.1 say about setback requirements?" and vector search returns chunks about setback requirements from five different sections because they're all semantically similar. The specific section number? Vectors don't really capture that.

Enter hybrid search: combine vector similarity with old-school keyword matching (BM25).

```python
from rank_bm25 import BM25Okapi
import numpy as np

class HybridRetriever:
    def __init__(self, chunks: list, embeddings: list):
        self.chunks = chunks
        self.embeddings = np.array(embeddings)
        
        tokenized = [chunk.text.lower().split() for chunk in chunks]
        self.bm25 = BM25Okapi(tokenized)
    
    def search(
        self, 
        query: str, 
        query_embedding: list,
        top_k: int = 10,
        alpha: float = 0.5  # 0 = pure BM25, 1 = pure vector
    ) -> list:
        
        # BM25 scores
        bm25_scores = self.bm25.get_scores(query.lower().split())
        bm25_scores = bm25_scores / (bm25_scores.max() + 1e-6)
        
        # Vector similarity
        query_vec = np.array(query_embedding)
        vector_scores = np.dot(self.embeddings, query_vec)
        vector_scores = (vector_scores + 1) / 2
        
        # Blend
        combined = alpha * vector_scores + (1 - alpha) * bm25_scores
        
        top_indices = np.argsort(combined)[-top_k:][::-1]
        return [(self.chunks[i], combined[i]) for i in top_indices]
```

The `alpha` parameter controls the balance. I've found 0.5-0.7 works for most of our document types. Technical documents with lots of specific codes and section numbers benefit from more BM25 weight (lower alpha). Conversational content does better with more vector weight.

After adding hybrid search, our "exact reference" queries -- where users ask about a specific section or code number -- went from about 60% accuracy to over 90%. That alone justified the extra complexity.

---

## Reranking: The Second Pass

Initial retrieval is optimized for speed, not precision. You're comparing a query vector against thousands of chunk vectors using approximate nearest neighbor search. It's fast but it misses nuance.

Reranking is a second pass where you use a cross-encoder to score each candidate more carefully. Cross-encoders look at the query and document together (not separately), so they catch semantic relationships that embedding similarity misses.

```python
from sentence_transformers import CrossEncoder

reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

def rerank(query: str, chunks: list, top_k: int = 5) -> list:
    pairs = [[query, chunk.text] for chunk in chunks]
    scores = reranker.predict(pairs)
    
    ranked = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)
    return [chunk for chunk, score in ranked[:top_k]]
```

The pattern is: retrieve many, rerank to few. I typically retrieve 20 candidates and rerank down to 5. The reranker is slower -- it can't be as fast as a vector dot product -- so you only run it on a small candidate set.

```python
def retrieve_and_rerank(query: str, retriever, top_k: int = 5):
    candidates = retriever.search(query, embed(query), top_k=top_k * 4)
    chunks = [chunk for chunk, score in candidates]
    return rerank(query, chunks, top_k=top_k)
```

Is reranking worth the added latency? In our pipeline, it adds about 200ms. For batch processing, that's nothing. For real-time chat, it depends on your tolerance. We use it everywhere and nobody has complained about speed.

---

## Measuring Whether Any of This Works

You need test cases. No way around it. Build a set of queries with known relevant documents and measure:

```python
def evaluate_retrieval(retriever, test_cases: list) -> dict:
    metrics = {"recall@5": [], "mrr": []}
    
    for case in test_cases:
        results = retriever.retrieve(case["query"], top_k=5)
        result_ids = [r.id for r in results]
        relevant = set(case["relevant_ids"])
        
        # What fraction of relevant docs did we find?
        retrieved_relevant = len(set(result_ids) & relevant)
        metrics["recall@5"].append(retrieved_relevant / len(relevant))
        
        # How high does the first relevant result rank?
        for i, rid in enumerate(result_ids):
            if rid in relevant:
                metrics["mrr"].append(1 / (i + 1))
                break
        else:
            metrics["mrr"].append(0)
    
    return {
        "recall@5": np.mean(metrics["recall@5"]),
        "mrr": np.mean(metrics["mrr"])
    }
```

I aim for Recall@5 above 0.8 and MRR above 0.5. If you're below that, look at your chunking first, then your embedding model, then consider hybrid search and reranking. In that order -- I've seen people add reranking to fix problems that were actually chunking problems, and it doesn't help.

Building the test set is the annoying part. We manually labeled about 200 query-document pairs. Took a full day. Worth every minute because now every change to the pipeline gets measured against a real benchmark instead of vibes.

---

*Next up: getting the LLM to actually do what you want with the retrieved context. Prompt engineering for pipelines is a different game than prompt engineering for chat.*
