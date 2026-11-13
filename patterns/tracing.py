"""Chapter 9 (monitoring), extended: OpenTelemetry tracing of model calls.

Logs say that something happened; a trace says where the time and the tokens went. Each model call becomes a
span with the GenAI semantic-convention attributes, so any OTLP backend (Jaeger, Tempo, Langfuse, ...) can show
cost and latency per step.
"""
from __future__ import annotations

from collections.abc import Callable

from opentelemetry import trace
from opentelemetry.trace import StatusCode

tracer = trace.get_tracer("llm-pipeline-cookbook")


def traced_completion(call: Callable[..., dict], *, model: str, purpose: str, provider: str = "openai", **kwargs) -> dict:
    """Run `call(model=model, **kwargs)` inside a span.

    `call` returns a dict with "content" and, when the provider reports it, "input_tokens" and "output_tokens".
    """
    with tracer.start_as_current_span(f"chat {model}") as span:
        span.set_attribute("gen_ai.operation.name", "chat")
        span.set_attribute("gen_ai.provider.name", provider)
        span.set_attribute("gen_ai.request.model", model)
        span.set_attribute("app.llm.purpose", purpose)
        try:
            result = call(model=model, **kwargs)
        except Exception as error:
            span.record_exception(error)
            span.set_status(StatusCode.ERROR)
            raise
        for key, attribute in (("input_tokens", "gen_ai.usage.input_tokens"), ("output_tokens", "gen_ai.usage.output_tokens")):
            if key in result:
                span.set_attribute(attribute, result[key])
        return result
