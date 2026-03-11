# LLM Pipeline Cookbook

Production patterns for building LLM pipelines that actually work.

This isn't another tutorial that stops at `openai.chat.completions.create()`. This is the stuff that comes after -- when you need to process ten thousand documents a month, handle failures gracefully, keep costs under control, and sleep at night knowing the pipeline won't break.

## What's Inside

1. **Architecture Overview** -- When to use LLMs vs traditional NLP, pipeline patterns, sync vs async
2. **Document Ingestion** -- PDF extraction, OCR, multi-format handling, failure recovery
3. **Chunking Strategies** -- Semantic chunking, table handling, overlap strategies
4. **Embedding & Retrieval** -- Vector stores, hybrid search, reranking
5. **Prompt Engineering for Pipelines** -- Structured output, few-shot templates, versioning
6. **LLM API Patterns** -- Retries, rate limiting, fallback providers, batching
7. **Error Handling & Recovery** -- Dead letter queues, partial failures, circuit breakers
8. **Cost Optimization** -- Model routing, token reduction, caching, cost tracking
9. **Monitoring & Observability** -- Logging, metrics, quality checks, alerting
10. **Deployment Patterns** -- Containers, queues, scaling, CI/CD

## Who This Is For

Engineers building production LLM pipelines. You know Python, you've used the OpenAI API, and now you need to make it work reliably at scale.

## Get the PDF

The full book with all code examples is available on [Gumroad](https://mbessalle.gumroad.com/l/llm-pipeline-cookbook).

## About the Author

Moises Bessalle is an AI engineer based in the Netherlands, building data engineering pipelines with LLMs for processing municipal documents. Previously a fullstack developer with a background in chemical engineering.

## License

The code examples in this repository are MIT licensed. The book content is copyrighted.
