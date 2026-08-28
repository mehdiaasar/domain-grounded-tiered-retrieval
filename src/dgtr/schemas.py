"""Structured-output schemas and the LangGraph state definition."""

from __future__ import annotations

from typing import Literal, TypedDict

from pydantic import BaseModel, Field


class ConstraintVerdict(BaseModel):
    claim_id: int = Field(description="The index of the claim being checked")
    verdict: Literal["PASS", "VIOLATION"] = Field(
        description="'VIOLATION' if the claim breaks a query constraint, otherwise 'PASS'"
    )
    reason: str = Field(description="Brief reason for the verdict")


class ConstraintCheckResult(BaseModel):
    checks: list[ConstraintVerdict] = Field(description="List of verdicts for all claims")


class ClaimJudgment(BaseModel):
    claim_id: int = Field(description="The index of the claim being evaluated (0-based)")
    reasoning: str = Field(description="A brief 1-sentence explanation.")
    label: Literal["CORRECT", "INCORRECT", "PARTIAL"] = Field(
        description="Strictly CORRECT, INCORRECT, or PARTIAL."
    )


class BatchVerificationResult(BaseModel):
    judgments: list[ClaimJudgment] = Field(
        description="List of rigorous judgments for all claims"
    )


class DomainClassification(BaseModel):
    domain: str = Field(
        description="The classified domain name. Must perfectly match one of the provided options."
    )


class SearchQuery(BaseModel):
    optimized_query: str = Field(description="The optimized keyword search string")


class DocumentGrader(BaseModel):
    binary_score: str = Field(description="Relevance score 'yes' or 'no'")
    reasoning: str = Field(description="1 sentence explaining why it is relevant or irrelevant")


class GraphState(TypedDict, total=False):
    """State threaded through every node of the graph.

    ``score`` is the composite claim score; the three rate fields break it
    down so a run can be audited claim-by-claim rather than only at the
    answer level.
    """

    query: str
    answer: str
    baseline_answer: str
    claims: list[str]
    domain: str | None
    retrieved_docs: str | None
    score: float
    labels: list[float]
    claim_reasons: list[str]
    iteration: int
    trusted_sources_used: list[str] | None
    hallucination_rate: float
    unsupported_rate: float
    groundedness_score: float
    source_map: list[str]
    constraint_labels: list[str] | None
    constraint_reasons: list[str] | None
    relevant_docs_found: bool
    is_fallback: bool
    trusted_search_done: bool
    general_search_done: bool
    grader_logs: str
    regeneration_attempts: int
    optimized_query: str
