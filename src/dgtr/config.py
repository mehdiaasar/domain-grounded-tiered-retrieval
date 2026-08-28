"""Runtime configuration and provider resolution.

The pipeline is provider-agnostic: the same graph runs against OpenAI, Groq,
Google Gemini, or a local Ollama daemon. Which one is used is decided here,
from environment variables or from keys typed into the web UI at request time.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from typing import Literal

from dotenv import load_dotenv

load_dotenv()

Provider = Literal["groq", "openai", "google", "ollama"]

#: Default model per provider. Groq is first because its free tier serves
#: llama-3.1-8b-instant -- the same 8B generator the paper evaluates.
DEFAULT_MODELS: dict[str, str] = {
    "groq": "llama-3.1-8b-instant",
    "openai": "gpt-4o-mini",
    "google": "gemini-2.0-flash",
    "ollama": "llama3.1:8b",
}

#: Environment variable holding each provider's API key. Ollama runs locally
#: and needs none.
PROVIDER_KEY_ENV: dict[str, str | None] = {
    "groq": "GROQ_API_KEY",
    "openai": "OPENAI_API_KEY",
    "google": "GOOGLE_API_KEY",
    "ollama": None,
}

#: Order in which providers are auto-detected when LLM_PROVIDER is unset.
PROVIDER_PRIORITY: tuple[str, ...] = ("groq", "openai", "google", "ollama")

#: Composite claim score at or above which an answer is accepted without
#: escalating to the next retrieval tier. Section 3.4 of the paper.
ACCEPT_THRESHOLD = 70.0


@dataclass(frozen=True)
class Settings:
    """Resolved configuration for one pipeline instance."""

    provider: Provider = "groq"
    model: str = ""
    judge_model: str = ""
    api_key: str | None = None
    tavily_api_key: str | None = None
    ollama_base_url: str = "http://localhost:11434"
    temperature: float = 0.0
    accept_threshold: float = ACCEPT_THRESHOLD
    max_search_results: int = 5
    request_timeout: float = 60.0
    tags: tuple[str, ...] = field(default=())

    def __post_init__(self) -> None:
        if not self.model:
            object.__setattr__(self, "model", DEFAULT_MODELS[self.provider])
        if not self.judge_model:
            object.__setattr__(self, "judge_model", self.model)

    # -- construction ----------------------------------------------------

    @classmethod
    def from_env(cls, **overrides) -> Settings:
        """Build settings from the environment, with keyword overrides.

        ``LLM_PROVIDER`` picks the provider explicitly. When it is unset the
        first provider in :data:`PROVIDER_PRIORITY` with a key present wins,
        falling back to Ollama (which needs no key) so a fully local install
        works out of the box.
        """
        requested = overrides.pop("provider", None) or os.getenv("LLM_PROVIDER") or ""
        provider = requested.strip().lower()
        if provider not in DEFAULT_MODELS:
            provider = _autodetect_provider()

        key_env = PROVIDER_KEY_ENV[provider]
        settings = cls(
            provider=provider,  # type: ignore[arg-type]
            model=os.getenv("LLM_MODEL", ""),
            judge_model=os.getenv("JUDGE_MODEL", ""),
            api_key=os.getenv(key_env) if key_env else None,
            tavily_api_key=os.getenv("TAVILY_API_KEY"),
            ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            accept_threshold=float(os.getenv("ACCEPT_THRESHOLD", ACCEPT_THRESHOLD)),
            max_search_results=int(os.getenv("MAX_SEARCH_RESULTS", 5)),
        )
        return replace(settings, **overrides) if overrides else settings

    def with_keys(self, api_key: str | None, tavily_api_key: str | None) -> Settings:
        """Return a copy carrying visitor-supplied keys, ignoring blanks."""
        return replace(
            self,
            api_key=api_key.strip() if api_key and api_key.strip() else self.api_key,
            tavily_api_key=(
                tavily_api_key.strip()
                if tavily_api_key and tavily_api_key.strip()
                else self.tavily_api_key
            ),
        )

    # -- readiness -------------------------------------------------------

    @property
    def needs_api_key(self) -> bool:
        return PROVIDER_KEY_ENV[self.provider] is not None

    @property
    def has_llm(self) -> bool:
        return bool(self.api_key) or not self.needs_api_key

    @property
    def has_search(self) -> bool:
        return bool(self.tavily_api_key)

    def missing(self) -> list[str]:
        """Human-readable names of the credentials still required."""
        gaps = []
        if not self.has_llm:
            gaps.append(PROVIDER_KEY_ENV[self.provider] or "")
        if not self.has_search:
            gaps.append("TAVILY_API_KEY")
        return [g for g in gaps if g]


def _autodetect_provider() -> str:
    """Pick the first provider whose key is present.

    With no keys at all we land on Groq rather than Ollama: a hosted demo has
    no local daemon, and the UI can point the visitor at Groq's free tier.
    Local-only users set ``LLM_PROVIDER=ollama``.
    """
    for candidate in PROVIDER_PRIORITY:
        key_env = PROVIDER_KEY_ENV[candidate]
        if key_env and os.getenv(key_env):
            return candidate
    return "groq"
