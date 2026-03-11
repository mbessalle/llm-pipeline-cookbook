"""Code examples from Chapter 07: error-handling"""

# --- Example 1 ---
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type((RateLimitError, TimeoutError, ConnectionError))
)
def process_document(doc):
    return llm_client.extract(doc.content)

# --- Example 2 ---
def should_retry(exception):
    if isinstance(exception, RateLimitError):
        return True
    if isinstance(exception, APIError):
        return exception.status_code >= 500  # server errors only
    if isinstance(exception, TimeoutError):
        return True
    # Don't retry validation errors, permission errors, etc.
    return False

# --- Example 3 ---
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

# --- Example 4 ---
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

# --- Example 5 ---
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

# --- Example 6 ---
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

# --- Example 7 ---
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

# --- Example 8 ---
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

# --- Example 9 ---
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

