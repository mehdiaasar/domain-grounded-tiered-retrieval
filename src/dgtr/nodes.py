"""Graph nodes and routers.

The notebook kept its models in module globals; here they are held on a
``Nodes`` instance so several pipelines (different providers, different
visitor keys) can coexist in one process -- which is what a web server needs.

Node logic is otherwise unchanged from the notebooks, including the failure
handling: a node that cannot do its job returns empty context and a
conservative label rather than raising, so a single flaky API call degrades
the answer instead of killing the run.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from dgtr import prompts
from dgtr.claims import parse_claims
from dgtr.config import Settings
from dgtr.llm import build_chat_model
from dgtr.schemas import (
    BatchVerificationResult,
    ConstraintCheckResult,
    DocumentGrader,
    DomainClassification,
    GraphState,
    SearchQuery,
)
from dgtr.search import build_search_tool, format_results, split_documents
from dgtr.sources import TRUSTED_SOURCES, allowlist_for

log = logging.getLogger(__name__)


class Nodes:
    """Bound node implementations for one configuration."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.gen_llm = build_chat_model(settings)
        # The notebook used three separate handles to the same model; one
        # shared instance is equivalent and a third of the connections.
        self.critic_llm = self.gen_llm
        self.rag_llm = self.gen_llm
        self._search = None

        self.generation_chain = (
            ChatPromptTemplate.from_template("Answer the query:\n{query}")
            | self.gen_llm
            | StrOutputParser()
        )
        self.claim_chain = (
            ChatPromptTemplate.from_template(prompts.CLAIM_EXTRACTION)
            | self.critic_llm
            | StrOutputParser()
        )
        self.regenerate_chain = (
            ChatPromptTemplate.from_messages(
                [("system", prompts.REGENERATE_SYSTEM), ("human", prompts.REGENERATE_USER)]
            )
            | self.rag_llm
            | StrOutputParser()
        )

    @property
    def search_tool(self) -> Any:
        if self._search is None:
            self._search = build_search_tool(self.settings)
        return self._search

    # -- generation ------------------------------------------------------

    def generate(self, state: GraphState) -> dict:
        """Zero-shot answer. Also recorded as the baseline for comparison."""
        answer = self.generation_chain.invoke({"query": state["query"]})
        return {"answer": answer, "baseline_answer": answer, "iteration": 0}

    def extract_claims(self, state: GraphState) -> dict:
        raw = self.claim_chain.invoke({"answer": state["answer"]})
        return {"claims": parse_claims(raw)}

    # -- verification ----------------------------------------------------

    def check_constraints(self, state: GraphState) -> dict:
        """Label each claim PASS/VIOLATION against the query's constraints."""
        claims = state.get("claims", [])
        if not claims:
            return {"constraint_labels": [], "constraint_reasons": []}

        prompt = ChatPromptTemplate.from_messages(
            [("system", prompts.CONSTRAINT_SYSTEM), ("human", prompts.CONSTRAINT_USER)]
        )
        chain = prompt | self.critic_llm.with_structured_output(ConstraintCheckResult)

        try:
            result = chain.invoke(
                {"query": state.get("query", ""), "claims": _numbered(claims)}
            )
            verdicts = result.checks
        except Exception as exc:
            # Defaulting to PASS keeps a flaky critic from forcing every query
            # down the retrieval ladder.
            log.warning("Constraint check failed: %s", exc)
            return {
                "constraint_labels": ["PASS"] * len(claims),
                "constraint_reasons": ["constraint check unavailable"] * len(claims),
            }

        verdict_map = {v.claim_id: v for v in verdicts}
        labels, reasons = [], []
        for i in range(len(claims)):
            verdict = verdict_map.get(i)
            labels.append(verdict.verdict if verdict else "PASS")
            reasons.append(verdict.reason if verdict else "not returned by critic")
        return {"constraint_labels": labels, "constraint_reasons": reasons}

    def score_intrinsic(self, state: GraphState) -> dict:
        """Closed-book grading: is the model's own knowledge good enough?

        Deliberately ruthless -- anything the critic cannot vouch for without
        a search engine scores 0, which routes the query into retrieval.
        """
        claims = state.get("claims", [])
        if not claims:
            return _empty_scores()

        prompt = ChatPromptTemplate.from_messages(
            [("system", prompts.INTRINSIC_SYSTEM), ("human", prompts.INTRINSIC_USER)]
        )
        chain = prompt | self.critic_llm.with_structured_output(BatchVerificationResult)

        try:
            judgments = chain.invoke({"claims": _numbered(claims)}).judgments
        except Exception as exc:
            log.warning("Intrinsic scoring failed: %s", exc)
            judgments = []

        scores, reasons = _map_judgments(judgments, len(claims), partial_credit=False)
        total = len(scores)
        final = (sum(scores) / total) * 100 if total else 0.0
        return {
            "score": final,
            "labels": scores,
            "claim_reasons": reasons,
            "hallucination_rate": (scores.count(0.0) / total) * 100 if total else 0.0,
            "unsupported_rate": 0.0,
            "groundedness_score": final,
        }

    def score_retrieved(self, state: GraphState) -> dict:
        """Open-book grading against retrieved context, with partial credit."""
        claims = state.get("claims", [])
        if not claims:
            return _empty_scores()

        prompt = ChatPromptTemplate.from_messages(
            [("system", prompts.RETRIEVED_SYSTEM), ("human", prompts.RETRIEVED_USER)]
        )
        chain = prompt | self.critic_llm.with_structured_output(BatchVerificationResult)

        try:
            judgments = chain.invoke(
                {"docs": state.get("retrieved_docs", ""), "claims": _numbered(claims)}
            ).judgments
        except Exception as exc:
            log.warning("Retrieved scoring failed: %s", exc)
            judgments = []


        scores, reasons = _map_judgments(judgments, len(claims), partial_credit=True)
        total = len(scores)
        return {
            "score": (sum(scores) / total) * 100 if total else 0.0,
            "labels": scores,
            "claim_reasons": reasons,
            "hallucination_rate": (scores.count(0.0) / total) * 100 if total else 0.0,
            "unsupported_rate": (scores.count(0.5) / total) * 100 if total else 0.0,
            "groundedness_score": (scores.count(1.0) / total) * 100 if total else 0.0,
        }

    # -- routing and retrieval -------------------------------------------

    def detect_domain(self, state: GraphState) -> dict:
        query = state.get("query", "")
        if not query:
            return {"domain": "general"}

        prompt = ChatPromptTemplate.from_messages(
            [("system", prompts.DOMAIN_SYSTEM), ("human", "Query: {query}")]
        )
        try:
            chain = prompt | self.gen_llm.with_structured_output(DomainClassification)
            result = chain.invoke({"options": ", ".join(TRUSTED_SOURCES), "query": query})
            domain = result.domain.strip().lower()
            if domain not in TRUSTED_SOURCES:
                domain = "general"
        except Exception as exc:
            log.warning("Domain classification failed: %s", exc)
            domain = "general"
        return {"domain": domain}

    def trusted_search(self, state: GraphState) -> dict:
        """Tier 1: search restricted to the domain's curated allowlist."""
        query = state.get("query", "")
        allowed = allowlist_for(state.get("domain"))
        base = {
            "trusted_sources_used": allowed,
            "trusted_search_done": True,
            "is_fallback": False,
            "regeneration_attempts": 0,
        }
        if not query or not allowed:
            return {**base, "retrieved_docs": "", "source_map": []}

        optimized = self._optimize_query(query)
        try:
            results = self.search_tool.invoke(
                {"query": optimized, "include_domains": allowed}
            )
        except Exception as exc:
            log.warning("Trusted search failed: %s", exc)
            return {**base, "retrieved_docs": "", "source_map": [], "optimized_query": optimized}

        context, urls = format_results(results)
        return {**base, "retrieved_docs": context, "source_map": urls, "optimized_query": optimized}

    def general_web_search(self, state: GraphState) -> dict:
        """Tier 2: the open web, consulted only after tier 1 yields nothing."""
        query = state.get("query", "")
        base = {"is_fallback": True, "general_search_done": True, "regeneration_attempts": 0}
        if not query:
            return {**base, "retrieved_docs": "", "source_map": []}

        try:
            results = self.search_tool.invoke(query)
        except Exception as exc:
            log.warning("General web search failed: %s", exc)
            return {**base, "retrieved_docs": "", "source_map": []}

        context, urls = format_results(results)
        return {**base, "retrieved_docs": context, "source_map": urls}

    def grade_documents(self, state: GraphState) -> dict:
        """Binary relevance filter over each retrieved document."""
        docs = split_documents(state.get("retrieved_docs", ""))
        if not docs:
            return {
                "retrieved_docs": "",
                "relevant_docs_found": False,
                "grader_logs": "No documents to grade.",
            }

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", prompts.GRADER_SYSTEM),
                ("human", "User query: {query}\n\nRetrieved document:\n{document}"),
            ]
        )
        grader = prompt | self.critic_llm.with_structured_output(DocumentGrader)
        query = state.get("query", "")

        try:
            results = grader.batch(
                [{"query": query, "document": doc} for doc in docs], return_exceptions=True
            )
        except Exception as exc:
            # A grader outage should not discard retrieved evidence.
            log.warning("Grader batch failed: %s", exc)
            return {
                "retrieved_docs": "\n\n".join(docs),
                "relevant_docs_found": True,
                "grader_logs": f"Grader failed, keeping all documents: {exc}",
            }

        kept, logs = [], []
        for doc, result in zip(docs, results, strict=False):
            preview = doc.split("\n")[0][:60]
            if isinstance(result, Exception):
                kept.append(doc)
                logs.append(f"KEPT (grader error): {preview}")
            elif result.binary_score.lower() == "yes":
                kept.append(doc)
                logs.append(f"KEPT: {result.reasoning} ({preview})")
            else:
                logs.append(f"REJECTED: {result.reasoning} ({preview})")

        return {
            "retrieved_docs": "\n\n".join(kept),
            "relevant_docs_found": bool(kept),
            "grader_logs": "\n".join(logs),
        }

    def regenerate(self, state: GraphState) -> dict:
        """Rewrite the answer from graded context only."""
        query = state.get("query", "")
        docs = state.get("retrieved_docs", "")
        attempts = state.get("regeneration_attempts", 0)

        if not query:
            return {"answer": "No question was provided."}
        if not docs:
            return {
                "answer": "I cannot answer this query because no verified context was provided."
            }

        try:
            answer = self.regenerate_chain.invoke({"docs": docs, "query": query})
        except Exception as exc:
            log.warning("Regeneration failed: %s", exc)
            answer = "I encountered a system error while trying to formulate the final answer."
        return {"answer": answer, "regeneration_attempts": attempts + 1}

    def generate_apology(self, state: GraphState) -> dict:
        """Circuit breaker: decline rather than emit an ungrounded answer."""
        return {"answer": prompts.APOLOGY}

    # -- helpers ---------------------------------------------------------

    def _optimize_query(self, query: str) -> str:
        """Rewrite a conversational query into search keywords.

        Short queries are already keyword-dense; rewriting them costs a call
        and risks dropping a constraint.
        """
        if len(query.split()) < 6:
            return query
        prompt = ChatPromptTemplate.from_messages(
            [("system", prompts.REWRITE_SYSTEM), ("human", "Original Query: {query}")]
        )
        try:
            chain = prompt | self.critic_llm.with_structured_output(SearchQuery)
            return chain.invoke({"query": query}).optimized_query
        except Exception as exc:
            log.warning("Query rewrite failed: %s", exc)
            return query


# -- routers -------------------------------------------------------------
# Routers are pure functions of state, so they are testable without any model.


def route_after_constraints(state: GraphState) -> str:
    """A constraint violation skips closed-book scoring entirely.

    If the answer already breaks the query's own constraints, asking the model
    whether it believes itself is pointless -- go straight to retrieval.
    """
    labels = state.get("constraint_labels", []) or []
    if state.get("trusted_search_done") and state.get("general_search_done"):
        return "stop"
    if "VIOLATION" in labels:
        return "detect_domain"
    return "score_intrinsic"


def route_on_score_intrinsic(state: GraphState, threshold: float) -> str:
    """High parametric confidence bypasses retrieval altogether."""
    return "generate_answer" if state.get("score", 0.0) >= threshold else "detect_domain"


def route_after_grading(state: GraphState) -> str:
    if state.get("relevant_docs_found"):
        return "regenerate"
    if not state.get("general_search_done"):
        return "general_web_search"
    return "generate_apology"


def route_after_constraints_regen(state: GraphState) -> str:
    labels = state.get("constraint_labels", []) or []
    if "VIOLATION" not in labels:
        return "score_retrieved"
    if not state.get("general_search_done"):
        return "general_web_search"
    return "stop"


def route_on_score(state: GraphState, threshold: float) -> str:
    if state.get("score", 0) >= threshold:
        return "generate_answer"
    if not state.get("general_search_done"):
        return "general_web_search"
    return "stop"


def _numbered(claims: list[str]) -> str:
    return "\n".join(f"{i}. {claim}" for i, claim in enumerate(claims))


def _empty_scores() -> dict:
    return {
        "score": 0.0,
        "hallucination_rate": 0.0,
        "unsupported_rate": 0.0,
        "groundedness_score": 0.0,
        "labels": [],
        "claim_reasons": [],
    }


def _map_judgments(
    judgments: list, count: int, *, partial_credit: bool
) -> tuple[list[float], list[str]]:
    """Turn critic labels into numeric scores, defaulting missing ones to 0.

    An unjudged claim counts as a failure: silently treating it as correct
    would let a truncated critic response inflate the score.
    """
    by_id = {j.claim_id: j for j in judgments}
    scores, reasons = [], []
    for i in range(count):
        judgment = by_id.get(i)
        label = judgment.label.upper() if judgment else "INCORRECT"
        if label == "CORRECT":
            scores.append(1.0)
        elif label == "PARTIAL" and partial_credit:
            scores.append(0.5)
        else:
            scores.append(0.0)
        reasons.append(judgment.reasoning if judgment else "not judged")
    return scores, reasons
