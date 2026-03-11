# Chapter 1: Architecture Overview

Look, I'm going to be honest with you. Every LLM demo you've ever seen is a lie. Not intentionally -- but the gap between a Jupyter notebook that summarizes a PDF and a production system processing ten thousand municipal documents a month? It's massive. I spent my first three weeks on the job thinking "this should be straightforward" before reality set in.

So let's talk architecture. Not the theoretical kind you find in whitepapers, but the kind that survives contact with real data.

---

## Do You Even Need an LLM?

Seriously. Ask yourself this before you write a single line of pipeline code.

I've watched teams burn through $40K in API credits because nobody stopped to ask whether regex would've worked. Keyword extraction, pattern matching, simple classification with a rules engine -- these things are boring but they're fast, cheap, and deterministic. An LLM gives you none of those properties.

Here's how I think about it:

Reach for traditional NLP when you need exact matches, when speed matters more than nuance, when the rules are clear-cut, or when cost per request needs to be basically zero. Regex isn't glamorous. It works.

Reach for LLMs when context actually matters, when the task involves reasoning or summarization that you can't encode in rules, when edge cases are the norm rather than the exception. Processing Dutch municipal zoning documents where half of them have inconsistent formatting and the other half are scanned PDFs from 2003? Yeah, you need an LLM for that.

But here's the thing most people miss -- the best approach is almost always hybrid:

```python
# This pattern alone cut our API costs by 70%
def process_document(doc: Document) -> ProcessedResult:
    # Quick classification -- no LLM needed
    doc_type = classify_by_keywords(doc)
    
    # Simple cases get simple treatment
    if doc_type in SIMPLE_TYPES:
        return extract_with_rules(doc)
    
    # LLM only when we actually need it
    return extract_with_llm(doc)
```

Traditional NLP as a gatekeeper, LLM as the specialist. We went from calling GPT-4 on every single document to calling it on maybe 30% of them. Same output quality. A fraction of the cost.

---

## Pipeline Patterns That Actually Work

### The Straight Line

```
[Ingest] -> [Chunk] -> [Embed] -> [Store] -> [Query] -> [Generate]
```

This is where everyone starts, and honestly? For a lot of use cases it's fine. Simple RAG app, document search, internal knowledge base. You can build this in an afternoon.

The problem shows up at scale. One stage chokes and the whole thing stalls. You can't scale the embedding step independently from ingestion. And debugging is easy right up until it isn't.

Start here. But know you'll outgrow it.

### Fan-Out / Fan-In

```
                 +-> [Process A] -+
[Ingest] -> [Split] -+-> [Process B] -+-> [Merge] -> [Store]
                 +-> [Process C] -+
```

This is what we moved to when processing times started creeping past acceptable limits. Split the work, process in parallel, merge results. A document with twelve sections gets twelve parallel LLM calls instead of one sequential monster prompt.

The catch -- and I learned this on a particularly frustrating Wednesday -- is that merge logic gets complicated fast. What happens when Process B fails but A and C succeed? Do you retry just B? Do you return partial results? What if the sections have dependencies on each other?

We ended up with a state machine tracking each chunk. Overkill for a prototype. Essential for production.

### Event-Driven (Where You'll Probably End Up)

```
[Source] -> [Queue] -> [Workers] -> [Queue] -> [Workers] -> [Sink]
```

Queues between every stage. Workers pull tasks, process them, push results. If the API rate-limits you, messages just pile up in the queue and get processed when capacity frees up. A worker crashes? Another one picks up the message. Load spike on Monday morning when everyone uploads their weekend backlog? The queue absorbs it.

We use Redis for the queue layer -- nothing fancy. RabbitMQ or SQS would work too. The specific technology matters less than the pattern: decouple your stages so they can fail independently.

My actual recommendation: prototype with the straight line, ship with event-driven. Don't try to architect the perfect system on day one. You don't know enough about your data yet.

---

## The Sync vs Async Decision

This one bit me early. We had a document upload endpoint that processed everything synchronously. Worked great in development with a three-page test PDF. First real user uploaded a 200-page zoning report and the request timed out after 30 seconds. Nginx returned a 504, the user retried, now we had two copies of the same job running, and -- well, you can imagine.

**Synchronous** processing is fine when you're dealing with small documents, chat-style interactions, anything where the user can reasonably wait a few seconds. Keep it simple:

```python
@app.post("/process")
def process_document(doc: UploadFile):
    result = pipeline.process(doc)
    return result
```

**Asynchronous** is what you want for anything that might take more than about ten seconds. Give the user a job ID, let them poll for status. Or use websockets if you're feeling ambitious.

```python
@app.post("/process")
def submit_document(doc: UploadFile):
    job_id = queue.submit(doc)
    return {"job_id": job_id, "status_url": f"/jobs/{job_id}"}

@app.get("/jobs/{job_id}")
def get_status(job_id: str):
    return queue.get_status(job_id)
```

The rule I go by now: if your P95 latency is over ten seconds, you need async. Users will close the tab, retry, complain. And you'll waste compute on abandoned requests. Just make it async from the start if there's any doubt.

---

## Estimating Costs Before You Build

I cannot overstate how important this is. Build a cost calculator before you build the pipeline. I've seen two different teams (one at a previous job, one a friend's startup) get blindsided by API costs because they did their napkin math wrong.

Here are the variables you need:

- Documents per month (let's say 10,000)
- Average tokens per document (around 2,000 for our municipal docs)
- Chunks per document (we average 8)
- Embedding model (text-embedding-3-small is the sweet spot right now)
- LLM model (gpt-4o-mini for most things, gpt-4o when quality matters)
- LLM calls per document (we do about 3 -- classify, extract, validate)

```python
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
```

Fourteen dollars a month. For ten thousand documents. If you're charging a client even five cents per document, your margins are absurd. But -- and this is the part people forget -- that's with gpt-4o-mini. Swap in gpt-4o and multiply the LLM cost by roughly 20x. Now you're at $270/month. Still reasonable, but the economics change.

Run these numbers before you quote a price to anyone. Trust me on this.

---

## What I'd Tell Myself Six Months Ago

Don't reach for the LLM by default. Use it strategically, where it actually earns its keep.

Start with a simple architecture. You'll refactor it later, and that's fine -- you'll know more about your data by then.

Go async earlier than you think you need to. The sync-to-async migration is painful and you'll wish you'd just done it from the start.

And build that cost calculator. Tape it to your monitor if you have to.

---

*Next up: getting data into your pipeline without losing your mind. Document ingestion is where things get messy.*
