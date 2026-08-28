"""Graph assembly and the public ``Pipeline`` entry point."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from langgraph.graph import END, StateGraph

from dgtr import nodes as N
from dgtr.config import Settings
from dgtr.schemas import GraphState

#: Ordered node labels, used by the UI to render the trace in pipeline order.
NODE_LABELS: dict[str, str] = {
    "generate": "Zero-shot answer",
    "extract_claims_initial": "Decompose into atomic claims",
    "check_constraints": "Check query constraints",
    "score_intrinsic": "Closed-book claim scoring",
    "detect_domain": "Classify domain",
    "trusted_search": "Trusted-tier retrieval",
    "general_web_search": "Open-web fallback retrieval",
    "grade_documents": "Grade document relevance",
    "regenerate": "Regenerate from context",
    "extract_claims_regen": "Decompose regenerated answer",
    "check_constraints_regen": "Re-check query constraints",
    "score_retrieved": "Open-book claim scoring",
    "generate_apology": "Circuit breaker: decline",
}


def build_graph(settings: Settings) -> Any:
    """Compile the tiered-retrieval graph for ``settings``."""
    impl = N.Nodes(settings)
    threshold = settings.accept_threshold
    builder = StateGraph(GraphState)

    builder.add_node("generate", impl.generate)
    builder.add_node("extract_claims_initial", impl.extract_claims)
    builder.add_node("check_constraints", impl.check_constraints)
    builder.add_node("score_intrinsic", impl.score_intrinsic)
    builder.add_node("detect_domain", impl.detect_domain)
    builder.add_node("trusted_search", impl.trusted_search)
    builder.add_node("general_web_search", impl.general_web_search)
    builder.add_node("grade_documents", impl.grade_documents)
    builder.add_node("regenerate", impl.regenerate)
    builder.add_node("extract_claims_regen", impl.extract_claims)
    builder.add_node("check_constraints_regen", impl.check_constraints)
    builder.add_node("score_retrieved", impl.score_retrieved)
    builder.add_node("generate_apology", impl.generate_apology)

    builder.set_entry_point("generate")

    # Initial pass: answer, decompose, validate constraints.
    builder.add_edge("generate", "extract_claims_initial")
    builder.add_edge("extract_claims_initial", "check_constraints")
    builder.add_conditional_edges(
        "check_constraints",
        N.route_after_constraints,
        {
            "score_intrinsic": "score_intrinsic",
            "detect_domain": "detect_domain",
            "stop": "generate_apology",
        },
    )
    builder.add_conditional_edges(
        "score_intrinsic",
        lambda s: N.route_on_score_intrinsic(s, threshold),
        {"generate_answer": END, "detect_domain": "detect_domain"},
    )

    # Retrieval ladder: trusted allowlist first, open web only as fallback.
    builder.add_edge("detect_domain", "trusted_search")
    builder.add_edge("trusted_search", "grade_documents")
    builder.add_conditional_edges(
        "grade_documents",
        N.route_after_grading,
        {
            "regenerate": "regenerate",
            "general_web_search": "general_web_search",
            "generate_apology": "generate_apology",
        },
    )
    builder.add_edge("general_web_search", "grade_documents")

    # Regeneration and extrinsic verification. The trusted tier is exhausted
    # by this point, so failures fall straight through to the open web.
    builder.add_edge("regenerate", "extract_claims_regen")
    builder.add_edge("extract_claims_regen", "check_constraints_regen")
    builder.add_conditional_edges(
        "check_constraints_regen",
        N.route_after_constraints_regen,
        {
            "score_retrieved": "score_retrieved",
            "general_web_search": "general_web_search",
            "stop": "generate_apology",
        },
    )
    builder.add_conditional_edges(
        "score_retrieved",
        lambda s: N.route_on_score(s, threshold),
        {
            "generate_answer": END,
            "general_web_search": "general_web_search",
            "stop": "generate_apology",
        },
    )

    return builder.compile()


class Pipeline:
    """Convenience wrapper: run a query and collect its execution trace."""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or Settings.from_env()
        self.graph = build_graph(self.settings)

    def stream(self, query: str, recursion_limit: int = 40) -> Iterator[tuple[str, dict]]:
        """Yield ``(node_name, state_update)`` for each executed node."""
        stream = self.graph.stream(
            {"query": query},
            stream_mode="updates",
            config={"recursion_limit": recursion_limit},
        )
        for step in stream:
            yield from step.items()

    def run(self, query: str, recursion_limit: int = 40) -> dict:
        """Run to completion and return the accumulated final state.

        ``trace`` holds the ordered node names, which is what makes a run
        auditable -- which tier fired, and where it stopped.
        """
        state: dict = {"query": query, "trace": []}
        for node_name, update in self.stream(query, recursion_limit):
            state["trace"].append(node_name)
            state.update(update)
        return state
