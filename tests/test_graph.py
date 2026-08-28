"""End-to-end graph runs against a scripted model. No network, no keys.

These assert the behaviour the paper claims: retrieval is skipped when
closed-book confidence is high, the trusted tier is tried before the open web,
and the pipeline declines rather than answering ungrounded.
"""

from __future__ import annotations

import pytest

from dgtr import nodes as N
from dgtr.config import Settings
from dgtr.graph import Pipeline
from dgtr.schemas import (
    BatchVerificationResult,
    ClaimJudgment,
    ConstraintCheckResult,
    ConstraintVerdict,
    DocumentGrader,
    DomainClassification,
    SearchQuery,
)
from tests.stubs import ExplodingSearch, ScriptedSearch, StubLLM

ANSWER = "1. Insulin lowers blood glucose.\n2. Insulin is made in the pancreas."


def judgments(*labels: str) -> BatchVerificationResult:
    return BatchVerificationResult(
        judgments=[
            ClaimJudgment(claim_id=i, reasoning=f"stub {label}", label=label)
            for i, label in enumerate(labels)
        ]
    )


def constraints(*verdicts: str) -> ConstraintCheckResult:
    return ConstraintCheckResult(
        checks=[
            ConstraintVerdict(claim_id=i, verdict=v, reason="stub") for i, v in enumerate(verdicts)
        ]
    )


def tavily(*urls: str):
    return {"results": [{"url": u, "content": f"Body text for {u}."} for u in urls]}


@pytest.fixture
def settings():
    return Settings(provider="ollama", tavily_api_key="stub")


def make_pipeline(monkeypatch, settings, *, llm: StubLLM, search=None) -> Pipeline:
    """Build a real Pipeline whose model and search tool are stubs."""
    monkeypatch.setattr(N, "build_chat_model", lambda *a, **k: llm)
    if search is not None:
        monkeypatch.setattr(N, "build_search_tool", lambda *a, **k: search)
    return Pipeline(settings)


class TestConfidentAnswerBypassesRetrieval:
    """A claim set the critic fully vouches for should never hit the network."""

    def test_high_closed_book_score_ends_the_run(self, monkeypatch, settings):
        llm = StubLLM(
            text=ANSWER,
            structured={
                ConstraintCheckResult: constraints("PASS", "PASS"),
                BatchVerificationResult: judgments("CORRECT", "CORRECT"),
            },
        )
        search = ExplodingSearch("search must not be called")
        state = make_pipeline(monkeypatch, settings, llm=llm, search=search).run("q")

        assert state["trace"] == [
            "generate",
            "extract_claims_initial",
            "check_constraints",
            "score_intrinsic",
        ]
        assert state["score"] == 100.0
        assert state["answer"] == state["baseline_answer"]

    def test_partial_confidence_below_threshold_escalates(self, monkeypatch, settings):
        # One of two claims correct = 50, under the default 70.
        llm = StubLLM(
            text=ANSWER,
            structured={
                ConstraintCheckResult: constraints("PASS", "PASS"),
                BatchVerificationResult: judgments("CORRECT", "INCORRECT"),
                DomainClassification: DomainClassification(domain="medical"),
                SearchQuery: SearchQuery(optimized_query="insulin glucose"),
                DocumentGrader: DocumentGrader(binary_score="no", reasoning="off topic"),
            },
        )
        search = ScriptedSearch(tavily("https://who.int/x"))
        state = make_pipeline(monkeypatch, settings, llm=llm, search=search).run("q")

        assert "detect_domain" in state["trace"]
        assert state["domain"] == "medical"


class TestTieredEscalation:
    def test_trusted_tier_is_searched_before_the_open_web(self, monkeypatch, settings):
        # Closed-book scoring fails, then open-book scoring on the trusted
        # documents succeeds -- so the run should stop without the open web.
        scores = iter([judgments("INCORRECT", "INCORRECT"), judgments("CORRECT", "CORRECT")])
        llm = StubLLM(
            text=ANSWER,
            structured={
                ConstraintCheckResult: constraints("PASS", "PASS"),
                BatchVerificationResult: lambda _payload: next(scores),
                DomainClassification: DomainClassification(domain="medical"),
                SearchQuery: SearchQuery(optimized_query="insulin"),
                DocumentGrader: DocumentGrader(binary_score="yes", reasoning="relevant"),
            },
        )
        search = ScriptedSearch(tavily("https://who.int/insulin"))
        state = make_pipeline(monkeypatch, settings, llm=llm, search=search).run("q")

        trace = state["trace"]
        assert trace.index("trusted_search") < trace.index("grade_documents")
        assert "general_web_search" not in trace
        assert state["is_fallback"] is False
        assert state["groundedness_score"] == 100.0
        # The trusted call is domain-restricted; the open-web call is not.
        assert search.queries[0]["include_domains"] == N.allowlist_for("medical")

    def test_open_web_is_used_only_after_the_trusted_tier_is_rejected(
        self, monkeypatch, settings
    ):
        grades = iter(
            [
                DocumentGrader(binary_score="no", reasoning="paywall"),
                DocumentGrader(binary_score="yes", reasoning="relevant"),
            ]
        )
        llm = StubLLM(
            text=ANSWER,
            structured={
                ConstraintCheckResult: constraints("PASS", "PASS"),
                BatchVerificationResult: judgments("INCORRECT", "INCORRECT"),
                DomainClassification: DomainClassification(domain="medical"),
                SearchQuery: SearchQuery(optimized_query="insulin"),
                DocumentGrader: lambda _payload: next(grades),
            },
        )
        search = ScriptedSearch(
            tavily("https://who.int/paywalled"), tavily("https://example.test/open")
        )
        state = make_pipeline(monkeypatch, settings, llm=llm, search=search).run("q")

        trace = state["trace"]
        assert trace.index("trusted_search") < trace.index("general_web_search")
        assert state["is_fallback"] is True

    def test_constraint_violation_skips_closed_book_scoring(self, monkeypatch, settings):
        llm = StubLLM(
            text=ANSWER,
            structured={
                ConstraintCheckResult: constraints("VIOLATION", "PASS"),
                BatchVerificationResult: judgments("CORRECT", "CORRECT"),
                DomainClassification: DomainClassification(domain="history"),
                SearchQuery: SearchQuery(optimized_query="q"),
                DocumentGrader: DocumentGrader(binary_score="yes", reasoning="ok"),
            },
        )
        search = ScriptedSearch(tavily("https://britannica.com/a"))
        state = make_pipeline(monkeypatch, settings, llm=llm, search=search).run("q")

        assert "score_intrinsic" not in state["trace"]
        assert state["trace"][3] == "detect_domain"


class TestCircuitBreaker:
    def test_declines_when_every_tier_yields_nothing_usable(self, monkeypatch, settings):
        llm = StubLLM(
            text=ANSWER,
            structured={
                ConstraintCheckResult: constraints("PASS", "PASS"),
                BatchVerificationResult: judgments("INCORRECT", "INCORRECT"),
                DomainClassification: DomainClassification(domain="general"),
                SearchQuery: SearchQuery(optimized_query="q"),
                DocumentGrader: DocumentGrader(binary_score="no", reasoning="garbage"),
            },
        )
        search = ScriptedSearch(tavily("https://a.test/1"))
        state = make_pipeline(monkeypatch, settings, llm=llm, search=search).run("q")

        assert state["trace"][-1] == "generate_apology"
        assert state["answer"].startswith("I cannot confidently answer")

    def test_search_outage_degrades_to_declining_not_crashing(self, monkeypatch, settings):
        llm = StubLLM(
            text=ANSWER,
            structured={
                ConstraintCheckResult: constraints("PASS", "PASS"),
                BatchVerificationResult: judgments("INCORRECT", "INCORRECT"),
                DomainClassification: DomainClassification(domain="general"),
                SearchQuery: SearchQuery(optimized_query="q"),
            },
        )
        state = make_pipeline(
            monkeypatch, settings, llm=llm, search=ExplodingSearch()
        ).run("q")

        assert state["trace"][-1] == "generate_apology"
        # No retrieved prose to be cited as if it were evidence.
        assert state["retrieved_docs"] == ""
