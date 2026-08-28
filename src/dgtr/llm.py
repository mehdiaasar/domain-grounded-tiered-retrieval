"""Chat-model construction across the supported providers."""

from __future__ import annotations

from typing import Any

from dgtr.config import Settings


class ProviderUnavailable(RuntimeError):
    """Raised when a provider's package is missing or its key is absent."""


def build_chat_model(settings: Settings, model: str | None = None) -> Any:
    """Return a LangChain chat model for ``settings``.

    All four providers expose ``with_structured_output``, which the critic,
    router, and grader nodes depend on.
    """
    name = model or settings.model
    provider = settings.provider

    if settings.needs_api_key and not settings.api_key:
        raise ProviderUnavailable(
            f"{provider} needs an API key. Set it in .env or paste one in the sidebar."
        )

    if provider == "groq":
        try:
            from langchain_groq import ChatGroq
        except ImportError as exc:  # pragma: no cover - depends on install extras
            raise ProviderUnavailable("pip install langchain-groq") from exc
        return ChatGroq(
            model=name,
            api_key=settings.api_key,
            temperature=settings.temperature,
            timeout=settings.request_timeout,
        )

    if provider == "openai":
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:  # pragma: no cover
            raise ProviderUnavailable("pip install langchain-openai") from exc
        return ChatOpenAI(
            model=name,
            api_key=settings.api_key,
            temperature=settings.temperature,
            timeout=settings.request_timeout,
        )

    if provider == "google":
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError as exc:  # pragma: no cover
            raise ProviderUnavailable("pip install langchain-google-genai") from exc
        return ChatGoogleGenerativeAI(
            model=name,
            google_api_key=settings.api_key,
            temperature=settings.temperature,
        )

    if provider == "ollama":
        try:
            from langchain_ollama import ChatOllama
        except ImportError as exc:  # pragma: no cover
            raise ProviderUnavailable("pip install langchain-ollama") from exc
        return ChatOllama(
            model=name,
            base_url=settings.ollama_base_url,
            temperature=settings.temperature,
        )

    raise ProviderUnavailable(f"Unknown provider: {provider}")
