# Chapter 9: Monitoring & Observability

About a month into running our pipeline, I noticed extraction quality had degraded. Not dramatically -- but enough that users started complaining about missing dates and garbled entity names. Turns out OpenAI had quietly updated the gpt-4o-mini model, and some of our prompts weren't producing clean JSON anymore. The update happened two weeks before anyone noticed.

That's when I added monitoring. Not after something catastrophic -- after something subtle. The subtle failures are worse because they erode trust slowly.

---

## What You Need to Watch

There's a framework called the Four Golden Signals (from Google's SRE book) that works well here: latency, traffic, errors, and saturation. For LLM pipelines, I add two more: token usage (because that's your money) and output quality (because a 200 OK response can still contain garbage).

Latency tells you if something is slowing down. Traffic tells you if usage patterns are changing. Errors tell you what's breaking. Saturation tells you if you're approaching limits. Token usage tells you if costs are spiking. And quality tells you if the LLM is actually doing its job.

Most people monitor the first four and forget the last two. Don't.

---

## Log Every LLM Call

Every single one. This is non-negotiable. When something goes wrong (and it will), you need to trace it back to the specific API call.

```python
import structlog
import time
from uuid import uuid4

log = structlog.get_logger()

class LoggingLLMClient:
    def __init__(self, client):
        self.client = client
    
    def complete(self, model, messages, **kwargs):
        call_id = str(uuid4())[:8]
        start = time.perf_counter()
        
        try:
            response = self.client.chat.completions.create(
                model=model, messages=messages, **kwargs
            )
            
            latency = (time.perf_counter() - start) * 1000
            
            log.info("llm_call",
                call_id=call_id,
                model=model,
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
                latency_ms=round(latency, 2),
                status="ok"
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            latency = (time.perf_counter() - start) * 1000
            log.error("llm_call",
                call_id=call_id,
                model=model,
                error=type(e).__name__,
                message=str(e),
                latency_ms=round(latency, 2),
                status="error"
            )
            raise
```

That call_id is small but essential. When a user reports a problem with a specific document, I can grep the logs by document ID, find the call IDs, and see exactly what went in and what came out. Without it, you're searching through thousands of log entries trying to match timestamps.

---

## Track Latency Properly

Average latency is useless. If nine requests take 500ms and one takes 30 seconds, your average is 3.4 seconds -- which describes none of your actual requests. Track percentiles.

```python
from collections import deque
import statistics

class LatencyTracker:
    def __init__(self, window_size=1000):
        self.latencies = deque(maxlen=window_size)
    
    def record(self, latency_ms):
        self.latencies.append(latency_ms)
    
    def stats(self):
        if not self.latencies:
            return {}
        
        s = sorted(self.latencies)
        n = len(s)
        
        return {
            "count": n,
            "mean": round(statistics.mean(s), 1),
            "median": round(statistics.median(s), 1),
            "p95": round(s[int(n * 0.95)], 1),
            "p99": round(s[int(n * 0.99)], 1),
            "max": round(max(s), 1),
        }
```

The P95 is the number I care about most. That's the latency experienced by your unluckiest-but-not-extremely-unlucky users. If P95 is creeping up, something is degrading -- maybe a model update, maybe your prompts are getting longer, maybe the API is having a bad day.

We check this daily. A P95 under 3 seconds is good for our pipeline. When it crosses 5 seconds, I start investigating.

---

## Prometheus Metrics (If You're Serious)

For production, I use Prometheus with Grafana dashboards. The setup takes about an hour but the visibility it gives you is worth days of debugging.

```python
from prometheus_client import Counter, Histogram, Gauge

llm_calls = Counter('llm_calls_total', 'Total LLM calls', ['model', 'status'])
tokens_used = Counter('tokens_total', 'Total tokens', ['model', 'direction'])
llm_latency = Histogram('llm_latency_seconds', 'Call latency', ['model'],
    buckets=[0.1, 0.5, 1, 2, 5, 10, 30])
active_requests = Gauge('llm_active_requests', 'Active requests')

class MetricsClient:
    def __init__(self, client):
        self.client = client
    
    def complete(self, model, messages, **kwargs):
        active_requests.inc()
        start = time.perf_counter()
        
        try:
            response = self.client.chat.completions.create(
                model=model, messages=messages, **kwargs
            )
            
            llm_calls.labels(model=model, status='ok').inc()
            tokens_used.labels(model=model, direction='input').inc(response.usage.prompt_tokens)
            tokens_used.labels(model=model, direction='output').inc(response.usage.completion_tokens)
            
            return response.choices[0].message.content
        except Exception as e:
            llm_calls.labels(model=model, status='error').inc()
            raise
        finally:
            llm_latency.labels(model=model).observe(time.perf_counter() - start)
            active_requests.dec()
```

The Grafana dashboard shows me at a glance: requests per second, error rate, P95 latency, token burn rate, and cost estimate. I check it first thing every morning and again before leaving. Takes 30 seconds and catches problems before users notice them.

---

## Quality Monitoring: The Hard Part

This is where most people give up, because measuring output quality automatically is genuinely hard. But even basic checks catch a lot.

```python
import json

def check_output_quality(response, expected_schema):
    issues = []
    
    # Is it valid JSON?
    try:
        parsed = json.loads(response)
    except json.JSONDecodeError as e:
        return {"valid": False, "issues": [f"Invalid JSON: {e}"]}
    
    # Required fields present?
    for field in expected_schema.get("required_fields", []):
        if field not in parsed:
            issues.append(f"Missing: {field}")
    
    # Types correct?
    for field, expected_type in expected_schema.get("field_types", {}).items():
        if field in parsed and not isinstance(parsed[field], expected_type):
            issues.append(f"Wrong type: {field}")
    
    # Non-empty?
    if not response.strip():
        issues.append("Empty response")
    
    return {"valid": len(issues) == 0, "issues": issues}
```

We run this on every extraction result. The percentage of results that pass all quality checks is our primary quality metric. When that number drops below 95%, something changed and I need to investigate.

For deeper quality assessment, I periodically sample 50 random outputs and manually review them. Takes about 30 minutes, and it catches the kinds of quality issues that automated checks miss -- things like "the entity extraction technically returned valid JSON, but it completely missed the primary party in the contract."

---

## Alerting: Know Before Your Users Do

Simple alerts that have saved me multiple times:

```python
from dataclasses import dataclass
from typing import Callable

@dataclass
class Alert:
    name: str
    condition: Callable
    message: str
    severity: str  # "warning" or "critical"

class AlertManager:
    def __init__(self, notify_func):
        self.alerts = []
        self.notify = notify_func
    
    def add(self, alert):
        self.alerts.append(alert)
    
    def check(self):
        for alert in self.alerts:
            if alert.condition():
                self.notify(alert.severity, f"{alert.name}: {alert.message}")
```

The three alerts I'd set up first:

Error rate above 10% in the last hour. This catches API outages and broken prompts.

P95 latency above 5 seconds. This catches API degradation and runaway requests.

Hourly cost exceeding 2x the expected rate. This catches prompt changes that accidentally explode token usage.

I run these checks every 5 minutes via a cron job. Notifications go to Slack. The cost alert alone has paid for itself three times over -- once when a developer pushed a prompt change that removed the max_tokens constraint, and the LLM started generating 4000-token responses instead of 200-token ones.

---

## Distributed Tracing for Complex Pipelines

If your pipeline has multiple stages (ingestion, chunking, embedding, extraction, storage), you need to trace a document's journey through all of them. OpenTelemetry does this well.

```python
from opentelemetry import trace

tracer = trace.get_tracer("llm-pipeline")

def process_document(doc):
    with tracer.start_as_current_span("process_document") as span:
        span.set_attribute("document.id", doc.id)
        
        with tracer.start_as_current_span("ingest"):
            text = ingest(doc)
        
        with tracer.start_as_current_span("chunk") as cs:
            chunks = chunk(text)
            cs.set_attribute("chunk.count", len(chunks))
        
        with tracer.start_as_current_span("extract") as es:
            result = extract(chunks)
            es.set_attribute("tokens.total", result.total_tokens)
        
        return result
```

This gives you a waterfall view of each document's processing. When a document takes 45 seconds instead of the usual 3, you can see exactly which stage was slow. For us, it's almost always the LLM extraction stage, but occasionally it's ingestion (huge PDFs) or embedding (batch too large).

---

## The Dashboard You Actually Need

I've seen teams build elaborate dashboards with 30 panels that nobody looks at. Here's what I actually check daily:

Real-time: active requests, requests per second, current error rate, queue depth.

Hourly: total requests processed, total tokens consumed (split by input/output), estimated cost, error breakdown by type.

Quality: schema validation pass rate, average completeness score, any parsing failures.

That's it. Three sections, maybe ten graphs total. If something is wrong, one of these will show it. Everything else is noise.

---

*Last chapter: getting all of this deployed and running reliably. Deployment patterns for LLM pipelines.*
