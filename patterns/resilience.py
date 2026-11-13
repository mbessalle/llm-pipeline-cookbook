"""Chapter 6 (API patterns) and Chapter 7 (error handling): retries, rate limiting, provider fallback,
circuit breaking."""
from __future__ import annotations

import logging
import random
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from functools import wraps

log = logging.getLogger("patterns")


def retry_with_backoff(max_retries: int = 3, base_delay: float = 1.0, max_delay: float = 60.0,
                       retryable_exceptions: tuple = (Exception,), sleep: Callable[[float], None] = time.sleep,
                       jitter: Callable[[], float] = random.random):
    """Retry a call with exponential backoff and jitter. Non-retryable exceptions pass straight through."""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except retryable_exceptions as error:
                    if attempt == max_retries:
                        raise
                    delay = min(base_delay * (2 ** attempt), max_delay) * (0.5 + jitter())
                    log.warning("Attempt %d failed: %s. Retrying in %.1fs", attempt + 1, error, delay)
                    sleep(delay)

        return wrapper

    return decorator


class RateLimiter:
    """Token bucket: `requests_per_minute` tokens, refilled continuously."""

    def __init__(self, requests_per_minute: int, clock: Callable[[], float] = time.time,
                 sleep: Callable[[float], None] = time.sleep):
        self.capacity = requests_per_minute
        self.tokens = float(requests_per_minute)
        self.refill_rate = requests_per_minute / 60.0
        self._clock, self._sleep = clock, sleep
        self.last_refill = clock()
        self.lock = threading.Lock()

    def acquire(self, timeout: float = 60.0) -> bool:
        deadline = self._clock() + timeout
        while True:
            with self.lock:
                self._refill()
                if self.tokens >= 1:
                    self.tokens -= 1
                    return True
            if self._clock() >= deadline:
                return False
            self._sleep(0.1)

    def _refill(self) -> None:
        now = self._clock()
        self.tokens = min(self.capacity, self.tokens + (now - self.last_refill) * self.refill_rate)
        self.last_refill = now


@dataclass
class ProviderConfig:
    name: str
    client: object          # anything with .complete(model=..., messages=..., **kwargs)
    model: str
    priority: int
    healthy: bool = True
    failure_count: int = 0


class AllProvidersFailed(Exception):
    def __init__(self, errors: list[tuple[str, Exception]]):
        super().__init__(f"All providers failed: {[(name, str(error)) for name, error in errors]}")
        self.errors = errors


class FallbackClient:
    """Try providers in priority order; take one out of rotation after repeated failures."""

    def __init__(self, providers: list[ProviderConfig], max_failures: int = 3):
        self.providers = sorted(providers, key=lambda p: p.priority)
        self.max_failures = max_failures

    def complete(self, messages: list, **kwargs):
        errors = []
        for provider in self.providers:
            if not provider.healthy:
                continue
            try:
                result = provider.client.complete(model=provider.model, messages=messages, **kwargs)
                provider.failure_count = 0
                return result
            except Exception as error:  # noqa: BLE001 - any provider failure means "try the next one"
                errors.append((provider.name, error))
                provider.failure_count += 1
                if provider.failure_count >= self.max_failures:
                    provider.healthy = False
                    log.error("%s marked unhealthy after %d failures", provider.name, self.max_failures)
        raise AllProvidersFailed(errors)


class CircuitState(Enum):
    CLOSED = "closed"        # normal
    OPEN = "open"            # failing: reject immediately
    HALF_OPEN = "half_open"  # testing recovery


class CircuitOpenError(Exception):
    pass


class CircuitBreaker:
    """Stop calling a dependency that keeps failing; let one call through after `reset_timeout` to test it."""

    def __init__(self, failure_threshold: int = 5, reset_timeout: timedelta = timedelta(seconds=30),
                 now: Callable[[], datetime] = lambda: datetime.now(timezone.utc)):
        self.failure_threshold, self.reset_timeout, self._now = failure_threshold, reset_timeout, now
        self.failures = 0
        self.last_failure: datetime | None = None
        self.state = CircuitState.CLOSED

    def call(self, func, *args, **kwargs):
        if self.state == CircuitState.OPEN:
            if self._now() - self.last_failure > self.reset_timeout:
                self.state = CircuitState.HALF_OPEN
            else:
                raise CircuitOpenError("Circuit breaker is open: the dependency is down")
        try:
            result = func(*args, **kwargs)
        except Exception:
            self.failures += 1
            self.last_failure = self._now()
            if self.failures >= self.failure_threshold or self.state == CircuitState.HALF_OPEN:
                self.state = CircuitState.OPEN
            raise
        self.failures = 0
        self.state = CircuitState.CLOSED
        return result
