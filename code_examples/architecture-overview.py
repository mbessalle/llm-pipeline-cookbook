"""Code examples from Chapter 01: architecture-overview"""

# --- Example 1 ---
# This pattern alone cut our API costs by 70%
def process_document(doc: Document) -> ProcessedResult:
    # Quick classification -- no LLM needed
    doc_type = classify_by_keywords(doc)
    
    # Simple cases get simple treatment
    if doc_type in SIMPLE_TYPES:
        return extract_with_rules(doc)
    
    # LLM only when we actually need it
    return extract_with_llm(doc)

# --- Example 2 ---
@app.post("/process")
def process_document(doc: UploadFile):
    result = pipeline.process(doc)
    return result

# --- Example 3 ---
@app.post("/process")
def submit_document(doc: UploadFile):
    job_id = queue.submit(doc)
    return {"job_id": job_id, "status_url": f"/jobs/{job_id}"}

@app.get("/jobs/{job_id}")
def get_status(job_id: str):
    return queue.get_status(job_id)

# --- Example 4 ---
# Actual calculator I use -- adjust the prices as they change
EMBEDDING_COST = 0.00002   # per 1K tokens, text-embedding-3-small
LLM_INPUT_COST = 0.00015   # per 1K tokens, gpt-4o-mini
LLM_OUTPUT_COST = 0.0006   # per 1K tokens, gpt-4o-mini

docs = 10_000
chunks_per_doc = 8
tokens_per_chunk = 250
llm_calls = 3
input_tokens = 1000
output_tokens = 500

# Embeddings
embed_total = docs * chunks_per_doc * tokens_per_chunk
embed_cost = (embed_total / 1000) * EMBEDDING_COST
# 20M tokens -> $0.40/month. Basically free.

# LLM
llm_in = docs * llm_calls * input_tokens
llm_out = docs * llm_calls * output_tokens
llm_cost = (llm_in / 1000 * LLM_INPUT_COST) + (llm_out / 1000 * LLM_OUTPUT_COST)
# $4.50 + $9.00 = $13.50/month

# Grand total: ~$14/month for 10K documents
# That's $0.0014 per document

