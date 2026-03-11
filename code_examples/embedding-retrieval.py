"""Code examples from Chapter 04: embedding-retrieval"""

# --- Example 1 ---
from openai import OpenAI

client = OpenAI()

def embed_openai(texts: list[str], model: str = "text-embedding-3-small") -> list[list[float]]:
    response = client.embeddings.create(
        input=texts,
        model=model
    )
    return [item.embedding for item in response.data]

# --- Example 2 ---
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('BAAI/bge-large-en-v1.5')

def embed_local(texts: list[str]) -> list[list[float]]:
    return model.encode(texts).tolist()

# --- Example 3 ---
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

# --- Example 4 ---
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

# --- Example 5 ---
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

# --- Example 6 ---
from sentence_transformers import CrossEncoder

reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

def rerank(query: str, chunks: list, top_k: int = 5) -> list:
    pairs = [[query, chunk.text] for chunk in chunks]
    scores = reranker.predict(pairs)
    
    ranked = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)
    return [chunk for chunk, score in ranked[:top_k]]

# --- Example 7 ---
def retrieve_and_rerank(query: str, retriever, top_k: int = 5):
    candidates = retriever.search(query, embed(query), top_k=top_k * 4)
    chunks = [chunk for chunk, score in candidates]
    return rerank(query, chunks, top_k=top_k)

# --- Example 8 ---
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

