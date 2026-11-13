from datetime import datetime, timedelta, timezone

import pytest

from patterns.resilience import (
    AllProvidersFailed,
    CircuitBreaker,
    CircuitOpenError,
    CircuitState,
    FallbackClient,
    ProviderConfig,
    RateLimiter,
    retry_with_backoff,
)


def test_retry_succeeds_after_transient_failures_with_exponential_delays():
    slept, calls = [], []

    @retry_with_backoff(max_retries=3, base_delay=1.0, sleep=slept.append, jitter=lambda: 0.5)
    def flaky():
        calls.append(1)
        if len(calls) < 3:
            raise TimeoutError("slow")
        return "ok"

    assert flaky() == "ok" and len(calls) == 3
    assert slept == [1.0, 2.0]                        # base * 2^attempt, jitter factor 1.0


def test_retry_gives_up_after_the_limit_and_caps_the_delay():
    slept = []

    @retry_with_backoff(max_retries=4, base_delay=10.0, max_delay=25.0, sleep=slept.append, jitter=lambda: 0.5)
    def always_fails():
        raise TimeoutError("down")

    with pytest.raises(TimeoutError):
        always_fails()
    assert slept == [10.0, 20.0, 25.0, 25.0]          # four retries, capped at max_delay


def test_non_retryable_errors_are_not_retried():
    calls = []

    @retry_with_backoff(retryable_exceptions=(TimeoutError,), sleep=lambda s: None)
    def bad_request():
        calls.append(1)
        raise ValueError("invalid prompt")

    with pytest.raises(ValueError):
        bad_request()
    assert len(calls) == 1


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def time(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


def test_rate_limiter_spends_its_bucket_then_waits_for_refill():
    clock = FakeClock()
    limiter = RateLimiter(requests_per_minute=60, clock=clock.time, sleep=clock.sleep)   # 1 token per second
    assert all(limiter.acquire(timeout=0) for _ in range(60))        # the full bucket, instantly
    assert limiter.acquire(timeout=0) is False                       # empty
    assert limiter.acquire(timeout=5) is True and 0.9 <= clock.now <= 1.2   # waited about one second


class Provider:
    def __init__(self, reply=None, error=None):
        self.reply, self.error, self.calls = reply, error, 0

    def complete(self, model, messages, **kwargs):
        self.calls += 1
        if self.error:
            raise self.error
        return f"{model}: {self.reply}"


def test_fallback_uses_the_next_provider_and_retires_a_failing_one():
    primary, backup = Provider(error=ConnectionError("503")), Provider(reply="hello")
    client = FallbackClient([ProviderConfig("backup", backup, "small", priority=2),
                             ProviderConfig("primary", primary, "large", priority=1)], max_failures=2)
    assert client.complete([{"role": "user", "content": "hi"}]) == "small: hello"
    assert client.complete([]) == "small: hello"
    assert primary.calls == 2 and client.providers[0].healthy is False
    client.complete([])
    assert primary.calls == 2                                        # no longer tried


def test_fallback_reports_every_provider_error():
    client = FallbackClient([ProviderConfig("a", Provider(error=ConnectionError("a down")), "m", 1),
                             ProviderConfig("b", Provider(error=TimeoutError("b slow")), "m", 2)])
    with pytest.raises(AllProvidersFailed) as failure:
        client.complete([])
    assert [name for name, _ in failure.value.errors] == ["a", "b"]


def test_circuit_breaker_opens_rejects_then_recovers():
    now = [datetime(2026, 1, 1, tzinfo=timezone.utc)]
    breaker = CircuitBreaker(failure_threshold=2, reset_timeout=timedelta(seconds=30), now=lambda: now[0])

    def down():
        raise ConnectionError("down")

    for _ in range(2):
        with pytest.raises(ConnectionError):
            breaker.call(down)
    assert breaker.state == CircuitState.OPEN
    with pytest.raises(CircuitOpenError):
        breaker.call(lambda: "never runs")                           # rejected without calling

    now[0] += timedelta(seconds=31)
    with pytest.raises(ConnectionError):
        breaker.call(down)                                           # one trial call, still down
    assert breaker.state == CircuitState.OPEN

    now[0] += timedelta(seconds=31)
    assert breaker.call(lambda: "up") == "up" and breaker.state == CircuitState.CLOSED
