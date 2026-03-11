# Chapter 7: Error Handling & Recovery

Three weeks into running our pipeline in production, I woke up to 847 failed documents in the queue. An API outage had cascaded through the system, and because I hadn't built proper error handling, half of those documents were simply gone. No record of what failed, no way to retry, no dead letter queue. Just a log file full of stack traces and a sinking feeling.

That was the week I rebuilt our entire error handling approach. Here's what I learned.

---

## Not All Errors Are Equal

This sounds obvious but it took me embarrassingly long to internalize. A rate limit error and a corrupted document are completely different problems that need completely different solutions.

Transient errors -- API timeouts, rate limits, network blips -- fix themselves. You just need to wait and try again. I see these constantly. Some weeks it's dozens per day.

Permanent errors -- a PDF that's actually a JPEG, a document in a language your pipeline doesn't support, malformed data that no amount of retrying will fix -- these need to be logged and set aside. Retrying them is just burning money.

Partial failures are the tricky ones. You extracted the entities and the dates but the summary generation timed out. Do you throw everything away and retry? Or save what you got? We save the partial results. Something is almost always better than nothing.

And then there are poison documents. Files that don't just fail -- they crash your processor. An XML bomb that expands to gigabytes. A PDF with circular references that sends the parser into an infinite loop. These need to be quarantined so they don't keep crashing your pipeline.

---

## Retries That Actually Work

The tenacity library makes this almost too easy:

```python
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type((RateLimitError, TimeoutError, ConnectionError))
)
def process_document(doc):
    return llm_client.extract(doc.content)
```

But here's the thing most people miss: you need to be selective about what you retry. I had a bug early on where we were retrying on ValidationError -- which meant every time the LLM returned malformed JSON, we'd call the API three more times, get the same malformed JSON, and waste tokens.

```python
def should_retry(exception):
    if isinstance(exception, RateLimitError):
        return True
    if isinstance(exception, APIError):
        return exception.status_code >= 500  # server errors only
    if isinstance(exception, TimeoutError):
        return True
    # Don't retry validation errors, permission errors, etc.
    return False
```

Also -- set a timeout on individual retries. I had a document that caused the LLM to generate an endlessly long response. The API call never timed out, the retry logic never kicked in, and one worker was stuck for 45 minutes on a single document while everything else queued up behind it.

---

## Dead Letter Queues: Stop Losing Documents

When a document fails all retries, it goes into a dead letter queue. Not the trash. Not /dev/null. A queue where I can inspect it, understand why it failed, and reprocess it later when the issue is fixed.

```python
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
import json

@dataclass
class DeadLetter:
    document_id: str
    error_type: str
    error_message: str
    attempt_count: int
    first_attempt: str
    last_attempt: str
    document_path: str

class DeadLetterQueue:
    def __init__(self, storage_path: str):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
    
    def add(self, doc, error, attempt):
        dl = DeadLetter(
            document_id=doc.id,
            error_type=type(error).__name__,
            error_message=str(error),
            attempt_count=attempt,
            first_attempt=datetime.utcnow().isoformat(),
            last_attempt=datetime.utcnow().isoformat(),
            document_path=doc.source_path
        )
        
        file_path = self.storage_path / f"{doc.id}.json"
        with open(file_path, 'w') as f:
            json.dump(asdict(dl), f, default=str)
    
    def get_all(self):
        letters = []
        for fp in self.storage_path.glob("*.json"):
            with open(fp) as f:
                letters.append(DeadLetter(**json.load(f)))
        return letters
    
    def remove(self, document_id):
        fp = self.storage_path / f"{document_id}.json"
        if fp.exists():
            fp.unlink()
```

Every Monday morning I review the dead letter queue. Usually it's 20-40 documents from the previous week. Most are genuinely broken files -- password-protected PDFs, corrupted downloads, things that no code change will fix. But occasionally there's a pattern: a bunch of documents from the same source failing with the same error. That's a bug to fix.

We also run a weekly reprocessing job:

```python
def reprocess_dead_letters(dlq, processor):
    for letter in dlq.get_all():
        if letter.attempt_count >= 5:
            log.error(f"Giving up on {letter.document_id} after 5 attempts")
            continue
        
        try:
            doc = load_document(letter.document_path)
            result = processor.process(doc)
            dlq.remove(letter.document_id)
            log.info(f"Reprocessed {letter.document_id} successfully")
        except Exception as e:
            log.warning(f"Still failing: {letter.document_id}: {e}")
```

Sometimes documents fail because of temporary API issues. A week later, the same document processes fine. The dead letter queue gives you that second chance without any manual intervention.

---

## Partial Results: Don't Throw the Baby Out

A document has five extraction tasks: entities, dates, classification, summary, and sentiment. The summary generation times out. Do you discard everything?

No. You keep what you got.

```python
@dataclass
class ExtractionResult:
    success: bool
    data: dict
    errors: list
    partial: bool = False

def extract_with_partial_failure(doc):
    results = {}
    errors = []
    
    extractors = [
        ("entities", extract_entities),
        ("dates", extract_dates),
        ("summary", extract_summary),
        ("classification", classify_document),
    ]
    
    for field_name, extractor in extractors:
        try:
            results[field_name] = extractor(doc.content)
        except Exception as e:
            errors.append(f"{field_name}: {str(e)}")
            results[field_name] = None
    
    return ExtractionResult(
        success=len(errors) == 0,
        data=results,
        errors=errors,
        partial=len(errors) > 0 and len(errors) < len(extractors)
    )
```

Then decide what's acceptable:

```python
def is_result_acceptable(result, required_fields):
    if result.success:
        return True
    
    for field in required_fields:
        if result.data.get(field) is None:
            return False
    return True

# We require entities and classification. Summary is nice-to-have.
result = extract_with_partial_failure(doc)
if is_result_acceptable(result, ["entities", "classification"]):
    save_result(result)  # partial is fine
else:
    dead_letter_queue.add(doc, PartialExtractionError(result.errors), 1)
```

About 4% of our documents end up as partial results. We flag them in the database so users know the extraction isn't complete, and we reprocess them during off-peak hours.

---

## Idempotency: Safe to Rerun

Pipelines crash. Servers restart. Workers die mid-processing. When you restart, you'll reprocess some documents. Make sure that's safe.

```python
import hashlib

class IdempotentProcessor:
    def __init__(self, storage):
        self.storage = storage
    
    def process(self, doc, force=False):
        content_hash = hashlib.sha256(doc.content.encode()).hexdigest()
        doc_hash = f"{doc.id}:{content_hash}"
        
        if not force:
            existing = self.storage.get_by_hash(doc_hash)
            if existing:
                return existing  # already processed, return cached
        
        result = self._do_process(doc)
        result.content_hash = doc_hash
        self.storage.save(result)
        return result
```

The content hash means we detect if a document has changed. Same document ID but different content? Reprocess. Same content? Return the cached result. This saved us during a particularly bad incident where a restart caused our queue to replay 3,000 documents -- most had already been processed, and the idempotency check skipped them in milliseconds.

---

## Circuit Breakers: Fail Fast

When OpenAI is down, you don't want your pipeline to slowly timeout on every single request. That's thousands of 30-second timeouts, all burning memory and threads. A circuit breaker detects the pattern early and fails fast.

```python
from datetime import datetime, timedelta
from enum import Enum

class CircuitState(Enum):
    CLOSED = "closed"       # normal
    OPEN = "open"           # failing, reject immediately
    HALF_OPEN = "half_open" # testing recovery

class CircuitBreaker:
    def __init__(self, failure_threshold=5, reset_timeout=timedelta(seconds=30)):
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.failures = 0
        self.last_failure = None
        self.state = CircuitState.CLOSED
    
    def call(self, func, *args, **kwargs):
        if self.state == CircuitState.OPEN:
            if datetime.utcnow() - self.last_failure > self.reset_timeout:
                self.state = CircuitState.HALF_OPEN
            else:
                raise CircuitOpenError("Circuit breaker is open -- dependency is down")
        
        try:
            result = func(*args, **kwargs)
            self.failures = 0
            self.state = CircuitState.CLOSED
            return result
        except Exception as e:
            self.failures += 1
            self.last_failure = datetime.utcnow()
            if self.failures >= self.failure_threshold:
                self.state = CircuitState.OPEN
            raise
```

Five failures in a row and the circuit opens. All subsequent calls fail immediately -- no waiting for timeouts. After 30 seconds, it lets one request through (half-open state). If that succeeds, circuit closes and normal operation resumes. If it fails, circuit stays open.

In practice, this means when OpenAI goes down, we get five slow failures and then everything else fails fast. Documents go into the dead letter queue immediately instead of timing out one by one. When the API comes back, the circuit breaker detects it automatically and processing resumes.

---

## Logging: Structured, Not Storytelling

Early on, my logs looked like: `"Error processing document: something went wrong"`. Very helpful. Now I use structured logging with consistent fields.

```python
import structlog

log = structlog.get_logger()

def process_document(doc):
    log_ctx = log.bind(
        document_id=doc.id,
        source=doc.source_path,
        size=len(doc.content)
    )
    
    try:
        result = extract(doc)
        log_ctx.info("processed",
            entities=len(result.entities),
            tokens=result.token_count
        )
        return result
    except APIError as e:
        log_ctx.error("api_failed",
            status=e.status_code,
            message=str(e)
        )
        raise
    except Exception as e:
        log_ctx.exception("unexpected_error")
        raise
```

Every log entry has the document ID and source path. I can grep for a specific document and see its entire processing history. When something fails at 3 AM, structured logs are the difference between a 5-minute fix and a 2-hour investigation.

---

*Next: you've built a reliable pipeline. Now let's make sure you're not spending more than you need to. Cost optimization is where the real money savings live.*
