# Chapter 6: LLM API Patterns

Every LLM tutorial shows you this:

```python
response = client.chat.completions.create(...)
```

And it works. In your notebook. On your laptop. With one document. Then you deploy it and discover that LLM APIs are, to put it diplomatically, unreliable.

I kept a log for one month of every API error our pipeline hit. Rate limits: 847 times. Timeouts: 203 times. Server errors (500/503): 68 times. Random connection resets: 31 times. That's over a thousand failures in a month, on a pipeline processing about ten thousand documents. Roughly one in ten calls fails on the first try.

You need to build for this.

---

## Retries: The Non-Negotiable

If you take nothing else from this chapter, take this: wrap every API call in a retry with exponential backoff. Not linear backoff, not fixed delays. Exponential, with jitter.

```python
import time
import random
from functools import wraps

def retry_with_backoff(
    max_retries=3,
    base_delay=1.0,
    max_delay=60.0,
    retryable_exceptions=(Exception,)
):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except retryable_exceptions as e:
                    last_exception = e
                    if attempt == max_retries:
                        raise
                    
                    delay = min(base_delay * (2 ** attempt), max_delay)
                    delay = delay * (0.5 + random.random())  # jitter
                    
                    log.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay:.1f}s")
                    time.sleep(delay)
            
            raise last_exception
        return wrapper
    return decorator
```

The jitter is important. Without it, when a rate limit hits and all your workers retry at the same time, they'll all retry at the same time again. And again. It's called the thundering herd problem and it will make your rate limiting worse, not better. Random jitter spreads the retries out.

I use max_retries=3 for most calls. First retry after ~1 second, second after ~2 seconds, third after ~4 seconds. If it's still failing after that, something is genuinely broken and retrying won't help.

---

## Rate Limiting: Be a Good Citizen

OpenAI's rate limits are per-minute. Hit them and you get a 429 response. Hit them hard enough and they'll throttle you for longer. Better to limit yourself proactively.

```python
import threading

class RateLimiter:
    def __init__(self, requests_per_minute: int):
        self.capacity = requests_per_minute
        self.tokens = requests_per_minute
        self.last_refill = time.time()
        self.lock = threading.Lock()
        self.refill_rate = requests_per_minute / 60.0
    
    def acquire(self, timeout: float = 60.0) -> bool:
        deadline = time.time() + timeout
        
        while True:
            with self.lock:
                self._refill()
                if self.tokens >= 1:
                    self.tokens -= 1
                    return True
            
            if time.time() >= deadline:
                return False
            time.sleep(0.1)
    
    def _refill(self):
        now = time.time()
        elapsed = now - self.last_refill
        self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
        self.last_refill = now
```

We run at about 80% of our actual rate limit. Leaves headroom for bursts and prevents the pipeline from constantly bouncing off the ceiling.

Even better -- adapt to the rate limit headers the API sends back:

```python
class AdaptiveRateLimiter:
    def __init__(self):
        self.retry_after = 0
        self.remaining = float('inf')
        self.reset_at = 0
    
    def update_from_response(self, headers: dict):
        if 'x-ratelimit-remaining' in headers:
            self.remaining = int(headers['x-ratelimit-remaining'])
        if 'x-ratelimit-reset' in headers:
            self.reset_at = float(headers['x-ratelimit-reset'])
        if 'retry-after' in headers:
            self.retry_after = int(headers['retry-after'])
    
    def wait_if_needed(self):
        if self.retry_after > 0:
            time.sleep(self.retry_after)
            self.retry_after = 0
        elif self.remaining == 0 and self.reset_at > time.time():
            time.sleep(self.reset_at - time.time())
```

This way the limiter learns from the API's actual responses instead of relying on hardcoded numbers that might change.

---

## Batching: Parallel But Controlled

Processing documents one at a time is slow. Processing them all at once will get you rate-limited instantly. The answer is controlled concurrency.

```python
from concurrent.futures import ThreadPoolExecutor, as_completed

def batch_process(items, processor, batch_size=10, max_workers=5):
    results = [None] * len(items)
    
    def process_item(index, item):
        result = processor(item)
        return index, result
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        
        for i, item in enumerate(items):
            future = executor.submit(process_item, i, item)
            futures.append(future)
            
            if (i + 1) % batch_size == 0:
                time.sleep(1)  # breathe between batches
        
        for future in as_completed(futures):
            try:
                index, result = future.result()
                results[index] = result
            except Exception as e:
                log.error(f"Item failed: {e}")
    
    return results
```

Five concurrent workers with a one-second pause every ten submissions. It's not optimal in the theoretical sense -- you could squeeze more throughput with adaptive concurrency. But it's predictable, easy to reason about, and stays well within rate limits.

The `results = [None] * len(items)` pattern preserves ordering. When futures complete out of order (they will), the index tracks which result goes where. I learned this after spending an afternoon debugging why document results were shuffled.

---

## Fallback Providers

Last February, OpenAI had a four-hour outage. Our pipeline just... stopped. Ten thousand documents queued up, nothing processing. After that, I built a fallback system.

```python
from dataclasses import dataclass

@dataclass
class ProviderConfig:
    name: str
    client: object
    model: str
    priority: int
    healthy: bool = True
    failure_count: int = 0

class FallbackClient:
    def __init__(self, providers: list[ProviderConfig]):
        self.providers = sorted(providers, key=lambda p: p.priority)
        self.max_failures = 3
    
    def complete(self, messages: list, **kwargs) -> str:
        errors = []
        
        for provider in self.providers:
            if not provider.healthy:
                continue
            
            try:
                result = provider.client.complete(
                    model=provider.model,
                    messages=messages,
                    **kwargs
                )
                provider.failure_count = 0  # reset on success
                return result
                
            except Exception as e:
                errors.append((provider.name, e))
                provider.failure_count += 1
                
                if provider.failure_count >= self.max_failures:
                    provider.healthy = False
                    log.error(f"{provider.name} marked unhealthy after {self.max_failures} failures")
        
        raise Exception(f"All providers failed: {errors}")
```

Primary: OpenAI gpt-4o-mini. Fallback: Anthropic claude-3-haiku. Emergency fallback: a local Llama model that's slower but never goes down. In practice, the fallback has triggered maybe five times in six months. But those five times, the pipeline kept running instead of dying.

One subtlety: different models produce slightly different output formats. Even with the same prompt, gpt-4o-mini and claude-3-haiku will structure JSON differently sometimes. Make sure your parsing is robust enough to handle minor variations, or normalize the output in a post-processing step.

---

## Stream for Users, Batch for Pipelines

Quick rule. If a human is watching, stream the response so they see tokens appear. If a machine is consuming the output, just wait for the complete response. Streaming adds complexity (partial JSON, incomplete sentences) that you don't need in a pipeline.

```python
# For pipelines: just get the whole thing
def pipeline_call(prompt: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        stream=False
    )
    return response.choices[0].message.content

# For users: stream it
def user_facing_call(prompt: str):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        stream=True
    )
    for chunk in response:
        if chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content
```

---

## Putting It All Together

Here's roughly what our production client looks like. Not the prettiest code, but it's survived six months of daily use without major incidents.

```python
class ResilientLLMClient:
    def __init__(self, config):
        self.fallback = FallbackClient(config.providers)
        self.rate_limiter = RateLimiter(config.requests_per_minute)
        self.request_count = 0
        self.total_tokens = 0
    
    def complete(self, messages, max_tokens=1000, temperature=0.0):
        if not self.rate_limiter.acquire(timeout=30):
            raise Exception("Rate limit timeout -- queue is full")
        
        last_error = None
        for attempt in range(4):  # 1 try + 3 retries
            try:
                start = time.time()
                
                response = self.fallback.complete(
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                
                elapsed = time.time() - start
                self.request_count += 1
                self.total_tokens += response.usage.total_tokens
                
                return {
                    "content": response.choices[0].message.content,
                    "usage": response.usage,
                    "latency": elapsed
                }
                
            except Exception as e:
                last_error = e
                delay = min(1.0 * (2 ** attempt), 30)
                delay *= (0.5 + random.random())
                time.sleep(delay)
        
        raise last_error
```

It's retries wrapping fallbacks wrapping rate limiting. Three layers of defense against the chaos of distributed systems. Boring? Absolutely. Reliable? Also absolutely.

---

*Next: what happens when things go wrong despite all your defenses. Error handling and recovery for when the inevitable happens.*
