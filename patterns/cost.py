"""Chapter 8 (cost optimisation): model routing, response caching, cost tracking."""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime, timezone
from enum import Enum
from pathlib import Path


class TaskComplexity(Enum):
    SIMPLE = "simple"      # classification, extraction
    MEDIUM = "medium"      # summaries, moderate extraction
    COMPLEX = "complex"    # reasoning, nuanced analysis


def route_model(complexity: TaskComplexity, model_map: dict[TaskComplexity, str]) -> str:
    """Send each task to the cheapest model that is good enough for it. The map is a setting, not a constant."""
    return model_map[complexity]


class LLMCache:
    """Exact-match cache on (model, prompt). Identical requests cost nothing the second time."""

    def __init__(self, storage_path: str | Path):
        self.path = Path(storage_path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.hits = self.misses = 0

    def _file(self, prompt: str, model: str) -> Path:
        return self.path / (hashlib.sha256(f"{model}:{prompt}".encode()).hexdigest() + ".json")

    def get(self, prompt: str, model: str):
        file = self._file(prompt, model)
        if file.exists():
            self.hits += 1
            return json.loads(file.read_text())["response"]
        self.misses += 1
        return None

    def set(self, prompt: str, model: str, response) -> None:
        self._file(prompt, model).write_text(
            json.dumps({"response": response, "cached_at": datetime.now(timezone.utc).isoformat()}))

    def cached_call(self, prompt: str, model: str, call: Callable[[str, str], object]):
        cached = self.get(prompt, model)
        if cached is not None:
            return cached
        response = call(prompt, model)
        self.set(prompt, model, response)
        return response


@dataclass
class UsageRecord:
    day: date
    model: str
    input_tokens: int
    output_tokens: int
    cost: float
    task: str


class CostTracker:
    """Cost per call from a price table (USD per million tokens), with a daily report by task and by model."""

    def __init__(self, prices: dict[str, tuple[float, float]], today: Callable[[], date] = date.today):
        self.prices, self._today = prices, today
        self.records: list[UsageRecord] = []

    def record(self, model: str, input_tokens: int, output_tokens: int, task: str) -> float:
        input_price, output_price = self.prices[model]
        cost = (input_tokens * input_price + output_tokens * output_price) / 1_000_000
        self.records.append(UsageRecord(self._today(), model, input_tokens, output_tokens, cost, task))
        return cost

    def daily_report(self) -> dict:
        records = [r for r in self.records if r.day == self._today()]
        by_task: dict[str, float] = defaultdict(float)
        by_model: dict[str, float] = defaultdict(float)
        for r in records:
            by_task[r.task] += r.cost
            by_model[r.model] += r.cost
        return {"total_cost": sum(r.cost for r in records),
                "total_tokens": sum(r.input_tokens + r.output_tokens for r in records),
                "by_task": dict(by_task), "by_model": dict(by_model)}
