import pytest

from dgtr.config import DEFAULT_MODELS, Settings


def test_provider_default_model(monkeypatch):
    monkeypatch.delenv("LLM_MODEL", raising=False)
    assert Settings(provider="openai").model == DEFAULT_MODELS["openai"]


def test_explicit_provider_wins_over_env(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("LLM_MODEL", "")
    assert Settings.from_env(provider="google").provider == "google"


def test_autodetect_prefers_the_first_key_present(monkeypatch):
    for var in ("LLM_PROVIDER", "LLM_MODEL", "GROQ_API_KEY", "OPENAI_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    assert Settings.from_env().provider == "openai"


def test_autodetect_falls_back_to_the_free_tier(monkeypatch):
    for var in ("LLM_PROVIDER", "LLM_MODEL", "GROQ_API_KEY", "OPENAI_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    assert Settings.from_env().provider == "groq"


def test_ollama_needs_no_key():
    settings = Settings(provider="ollama")
    assert settings.has_llm
    assert settings.missing() == ["TAVILY_API_KEY"]


def test_missing_lists_both_credentials():
    assert Settings(provider="groq").missing() == ["GROQ_API_KEY", "TAVILY_API_KEY"]


def test_with_keys_ignores_blanks():
    settings = Settings(provider="groq", api_key="original", tavily_api_key="tv")
    assert settings.with_keys("  ", None).api_key == "original"
    assert settings.with_keys("replacement", " ").api_key == "replacement"


@pytest.mark.parametrize("provider", sorted(DEFAULT_MODELS))
def test_every_provider_has_a_default_model(provider):
    assert Settings(provider=provider).model
