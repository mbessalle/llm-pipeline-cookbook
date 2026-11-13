"""Chapter 7 (error handling): dead letter queue and idempotent processing."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path


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
    """Failed documents are parked on disk with the reason, so one bad input never blocks the batch."""

    def __init__(self, storage_path: str | Path):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

    def add(self, document_id: str, document_path: str, error: Exception, attempt: int) -> DeadLetter:
        now = datetime.now(timezone.utc).isoformat()
        existing = self._read(document_id)
        letter = DeadLetter(
            document_id=document_id, error_type=type(error).__name__, error_message=str(error), attempt_count=attempt,
            first_attempt=existing.first_attempt if existing else now, last_attempt=now, document_path=document_path)
        (self.storage_path / f"{document_id}.json").write_text(json.dumps(asdict(letter)))
        return letter

    def get_all(self) -> list[DeadLetter]:
        return [DeadLetter(**json.loads(path.read_text())) for path in sorted(self.storage_path.glob("*.json"))]

    def remove(self, document_id: str) -> None:
        (self.storage_path / f"{document_id}.json").unlink(missing_ok=True)

    def _read(self, document_id: str) -> DeadLetter | None:
        path = self.storage_path / f"{document_id}.json"
        return DeadLetter(**json.loads(path.read_text())) if path.exists() else None


def reprocess_dead_letters(queue: DeadLetterQueue, process, max_attempts: int = 5) -> dict:
    """Retry parked documents. `process(document_path)` raises on failure. Returns what happened to each."""
    outcome = {"recovered": [], "still_failing": [], "given_up": []}
    for letter in queue.get_all():
        if letter.attempt_count >= max_attempts:
            outcome["given_up"].append(letter.document_id)
            continue
        try:
            process(letter.document_path)
        except Exception as error:  # noqa: BLE001 - any failure of this step is handled the same way
            queue.add(letter.document_id, letter.document_path, error, letter.attempt_count + 1)
            outcome["still_failing"].append(letter.document_id)
        else:
            queue.remove(letter.document_id)
            outcome["recovered"].append(letter.document_id)
    return outcome


class IdempotentProcessor:
    """Process a document once per content version: a rerun with the same content returns the stored result."""

    def __init__(self, process, results: dict | None = None):
        self._process = process
        self.results = results if results is not None else {}

    def process(self, document_id: str, content: str, force: bool = False):
        key = f"{document_id}:{hashlib.sha256(content.encode()).hexdigest()}"
        if not force and key in self.results:
            return self.results[key]
        self.results[key] = self._process(content)
        return self.results[key]
