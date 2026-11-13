import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

exporter = InMemorySpanExporter()
provider = TracerProvider()
provider.add_span_processor(SimpleSpanProcessor(exporter))
trace.set_tracer_provider(provider)

from patterns.tracing import traced_completion


@pytest.fixture(autouse=True)
def clear_spans():
    exporter.clear()


def test_a_model_call_becomes_a_span_with_genai_attributes():
    def call(model, prompt):
        return {"content": "invoice", "input_tokens": 812, "output_tokens": 9}

    assert traced_completion(call, model="small", purpose="classify", prompt="...")["content"] == "invoice"
    (span,) = exporter.get_finished_spans()
    assert span.name == "chat small"
    assert dict(span.attributes) == {
        "gen_ai.operation.name": "chat", "gen_ai.provider.name": "openai", "gen_ai.request.model": "small",
        "app.llm.purpose": "classify", "gen_ai.usage.input_tokens": 812, "gen_ai.usage.output_tokens": 9}


def test_a_failed_call_is_recorded_on_the_span_and_re_raised():
    def call(model):
        raise TimeoutError("provider timeout")

    with pytest.raises(TimeoutError):
        traced_completion(call, model="small", purpose="classify")
    (span,) = exporter.get_finished_spans()
    assert span.status.status_code.name == "ERROR" and span.events[0].name == "exception"
