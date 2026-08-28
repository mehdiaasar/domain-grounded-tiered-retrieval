"""A scripted chat model, so the graph can be exercised with no network.

The nodes only ever use two things from a chat model: piping it inside a
LangChain chain, and ``with_structured_output``. Implementing just those keeps
the stub small while still driving the real graph, the real routers, and the
real scoring arithmetic.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable, RunnableLambda


class StubLLM(Runnable):
    """Returns scripted text, and scripted objects for structured output."""

    def __init__(
        self,
        text: str | Callable[[Any], str] = "stub answer",
        structured: dict[type, Any] | None = None,
    ):
        self.text = text
        self.structured = structured or {}
        self.calls: list[Any] = []

    def invoke(self, input: Any, config: Any = None, **kwargs: Any) -> AIMessage:
        self.calls.append(input)
        text = self.text(input) if callable(self.text) else self.text
        return AIMessage(content=text)

    def with_structured_output(self, schema: type, **kwargs: Any) -> Runnable:
        value = self.structured.get(schema)
        if value is None:
            raise AssertionError(f"No stubbed structured output for {schema.__name__}")

        def _call(payload: Any) -> Any:
            self.calls.append(payload)
            return value(payload) if callable(value) else value

        return RunnableLambda(_call)


class ExplodingSearch:
    """A search tool whose every call fails, to exercise the degraded paths."""

    def __init__(self, error: str = "quota exceeded"):
        self.error = error

    def invoke(self, *_args: Any, **_kwargs: Any):
        raise RuntimeError(self.error)


class ScriptedSearch:
    """Returns a fixed Tavily-shaped payload per call, in order."""

    def __init__(self, *payloads: Any):
        self.payloads = list(payloads)
        self.queries: list[Any] = []

    def invoke(self, payload: Any, *_a: Any, **_k: Any):
        self.queries.append(payload)
        if not self.payloads:
            return {"results": []}
        return self.payloads.pop(0) if len(self.payloads) > 1 else self.payloads[0]
