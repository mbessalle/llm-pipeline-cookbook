"""Code examples from Chapter 09: monitoring"""

# --- Example 1 ---
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

# --- Example 2 ---
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

# --- Example 3 ---
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

# --- Example 4 ---
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

# --- Example 5 ---
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

# --- Example 6 ---
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

