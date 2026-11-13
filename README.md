# LLM Pipeline Cookbook

[![CI](https://github.com/mbessalle/llm-pipeline-cookbook/actions/workflows/ci.yml/badge.svg)](https://github.com/mbessalle/llm-pipeline-cookbook/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

Production patterns for building LLM pipelines that actually work: what to do about rate limits, flaky
providers, bad inputs, runaway cost and missing visibility.

## Runnable patterns

[`patterns/`](patterns/) holds the core patterns as importable, tested code. Time and sleeping are injectable,
so the tests run in milliseconds with no API key and no network.

| Pattern | What it solves | Code | Test |
|---|---|---|---|
| Retry with backoff and jitter | Transient provider errors, without retrying bad requests | [`resilience.py`](patterns/resilience.py) | delays are exactly base × 2ⁿ, capped, and stop at the limit |
| Token-bucket rate limiter | Staying under requests-per-minute limits | [`resilience.py`](patterns/resilience.py) | spends the bucket, then waits for the refill |
| Provider fallback | One provider down should not stop the pipeline | [`resilience.py`](patterns/resilience.py) | next provider is used; a failing one is retired |
| Circuit breaker | Stop hammering a dependency that is down | [`resilience.py`](patterns/resilience.py) | opens, rejects without calling, recovers |
| Dead letter queue | One bad document must not block the batch | [`recovery.py`](patterns/recovery.py) | parked, retried, given up on after the limit |
| Idempotent processing | Reruns must not pay twice for the same content | [`recovery.py`](patterns/recovery.py) | runs once per content version |
| Response cache | Identical requests cost nothing the second time | [`cost.py`](patterns/cost.py) | second identical call is free |
| Model routing and cost tracking | Cheapest model that is good enough; cost per task | [`cost.py`](patterns/cost.py) | cost matches the price table |
| OpenTelemetry tracing | Where the time and the tokens went | [`tracing.py`](patterns/tracing.py) | each call is a span with model and token counts |

```bash
pip install ".[dev]"
pytest          # 13 tests
ruff check .
```

```python
from patterns.resilience import retry_with_backoff
from patterns.tracing import traced_completion


def call_model(model: str, prompt: str) -> dict:   # your provider call goes here
    return {"content": "invoice", "input_tokens": 812, "output_tokens": 9}


@retry_with_backoff(max_retries=3, retryable_exceptions=(TimeoutError, ConnectionError))
def classify(document: str) -> dict:
    return traced_completion(call_model, model="small-model", purpose="classify", prompt=document)


print(classify("Invoice 2026-0113 ...")["content"])
```

## Book excerpts

[`code_examples/`](code_examples/) contains the **code examples as printed** in the LLM Pipeline Cookbook, one file per chapter. They are excerpts that follow the text, not importable modules. The full book with explanations, architecture decisions, and lessons learned is available on [Gumroad](https://mbessalle.gumroad.com/l/llm-pipeline-cookbook).

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
