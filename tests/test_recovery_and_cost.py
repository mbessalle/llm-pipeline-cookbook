from datetime import date

import pytest

from patterns.cost import CostTracker, LLMCache, TaskComplexity, route_model
from patterns.recovery import DeadLetterQueue, IdempotentProcessor, reprocess_dead_letters


def test_dead_letters_are_parked_retried_and_given_up_on(tmp_path):
    queue = DeadLetterQueue(tmp_path / "dlq")
    queue.add("doc-1", "in/doc-1.pdf", ValueError("bad table"), attempt=1)
    queue.add("doc-2", "in/doc-2.pdf", TimeoutError("ocr timeout"), attempt=1)
    queue.add("doc-3", "in/doc-3.pdf", ValueError("corrupt"), attempt=5)
    assert [letter.error_type for letter in queue.get_all()] == ["ValueError", "TimeoutError", "ValueError"]

    def process(path):
        if "doc-2" in path:
            raise TimeoutError("still slow")

    outcome = reprocess_dead_letters(queue, process, max_attempts=5)
    assert outcome == {"recovered": ["doc-1"], "still_failing": ["doc-2"], "given_up": ["doc-3"]}
    remaining = {letter.document_id: letter for letter in queue.get_all()}
    assert set(remaining) == {"doc-2", "doc-3"} and remaining["doc-2"].attempt_count == 2
    assert remaining["doc-2"].first_attempt <= remaining["doc-2"].last_attempt


def test_idempotent_processing_runs_once_per_content_version():
    runs = []
    processor = IdempotentProcessor(lambda content: runs.append(content) or content.upper())
    assert processor.process("doc-1", "hello") == "HELLO"
    assert processor.process("doc-1", "hello") == "HELLO" and len(runs) == 1      # same content: stored result
    assert processor.process("doc-1", "hello v2") == "HELLO V2" and len(runs) == 2  # changed content: reprocessed
    processor.process("doc-1", "hello", force=True)
    assert len(runs) == 3


def test_cache_makes_the_second_identical_call_free(tmp_path):
    cache, calls = LLMCache(tmp_path / "cache"), []

    def call(prompt, model):
        calls.append((prompt, model))
        return {"label": "invoice"}

    assert cache.cached_call("classify this", "small", call) == {"label": "invoice"}
    assert cache.cached_call("classify this", "small", call) == {"label": "invoice"}
    assert len(calls) == 1 and (cache.hits, cache.misses) == (1, 1)
    cache.cached_call("classify this", "large", call)                 # another model is another entry
    assert len(calls) == 2


def test_cost_tracking_and_routing():
    prices = {"small": (0.10, 0.50), "large": (2.00, 10.00)}          # USD per million tokens (input, output)
    model_map = {TaskComplexity.SIMPLE: "small", TaskComplexity.MEDIUM: "small", TaskComplexity.COMPLEX: "large"}
    tracker = CostTracker(prices, today=lambda: date(2026, 1, 1))

    assert route_model(TaskComplexity.SIMPLE, model_map) == "small"
    assert tracker.record("small", 1_000_000, 200_000, task="classify") == pytest.approx(0.20)
    assert tracker.record(route_model(TaskComplexity.COMPLEX, model_map), 100_000, 50_000, task="analyse") == pytest.approx(0.70)
    report = tracker.daily_report()
    assert report["total_cost"] == pytest.approx(0.90) and report["total_tokens"] == 1_350_000
    assert report["by_task"] == pytest.approx({"classify": 0.20, "analyse": 0.70})
    assert report["by_model"] == pytest.approx({"small": 0.20, "large": 0.70})
