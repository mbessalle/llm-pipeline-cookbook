# LLM Pipeline Cookbook -- Code Examples

Production patterns for building LLM pipelines that actually work.

This repo contains the **code examples** from the LLM Pipeline Cookbook. The full book with explanations, architecture decisions, and lessons learned is available on [Gumroad](https://mbessalle.gumroad.com/l/llm-pipeline-cookbook).

## Code Examples

| File | Chapter | What's Inside |
|------|---------|---------------|
| `architecture-overview.py` | 1 | Hybrid NLP/LLM routing, cost calculator |
| `document-ingestion.py` | 2 | PDF extraction (PyMuPDF, pdfplumber, Unstructured), OCR, multi-format pipeline |
| `chunking-strategies.py` | 3 | Semantic chunking, table handling, overlap strategies |
| `embedding-retrieval.py` | 4 | Vector stores, hybrid search (BM25 + vectors), reranking |
| `prompt-engineering.py` | 5 | Structured output, few-shot templates, prompt versioning |
| `api-patterns.py` | 6 | Retries with backoff, rate limiting, provider fallback |
| `error-handling.py` | 7 | Dead letter queues, partial failure handling, circuit breakers |
| `cost-optimization.py` | 8 | Model routing, token reduction, caching |
| `monitoring.py` | 9 | Logging, Prometheus metrics, quality checks, alerting |
| `deployment.py` | 10 | FastAPI app, Celery workers, Docker, health checks |

## Get the Book

The code here works on its own, but the book explains *why* these patterns exist, when to use them, and the mistakes I made figuring them out.

**[Get the LLM Pipeline Cookbook on Gumroad](https://mbessalle.gumroad.com/l/llm-pipeline-cookbook)** -- $29

## About

Built by an AI engineer processing 10,000+ municipal documents monthly in the Netherlands. These patterns are extracted from production systems handling real workloads.

## License

MIT -- use the code however you want.
