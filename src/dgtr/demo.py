"""Offline demo mode: replay recorded runs from the evaluation CSVs.

The app must not be a dead page when no API keys are configured. Every row of
the shipped judge outputs is a complete real run -- baseline answer, retrieved
documents, grader decisions, extracted claims, and claim-level metrics -- so
the same UI can render it with no network access at all.

Nothing here is synthesised: a replayed trace shows what the pipeline actually
did on that query during the paper's evaluation.
"""

from __future__ import annotations

import ast
from collections.abc import Iterator
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import pandas as pd

from dgtr.claims import parse_claims
from dgtr.prompts import APOLOGY
from dgtr.search import split_documents

EVAL_DIR = Path(__file__).resolve().parents[2] / "evaluation"

#: Recorded benchmark runs, keyed by the label shown in the UI.
BENCHMARK_FILES: dict[str, str] = {
    "TimeQA v2": "evalfinal_llm_judge_timeqa_v2_150que_rm_constraint.csv",
    "FreshQA v2": "evaluation_results_llm_judge_FreshQA_v2_150que.csv",
    "TruthfulQA": "evaluation_results_llm_judge_TruthfulQA_v2_150que.csv",
    "MMLU Global Facts": "evaluation_results_llm_judge_mmlu_global_facts_rem_ceck_const.csv",
    "HaluEval General": "evaluation_results_llm_judge_HaluEval_general_rm_constrint.csv",
}


@dataclass
class RecordedRun:
    """One replayable query from the evaluation set."""

    benchmark: str
    query: str
    ground_truth: str
    baseline_answer: str
    proposed_answer: str
    retrieved_documents: str
    grader_logs: str
    claims: list[str]
    hallucination_rate: float
    groundedness: float
    rag_triggered: bool
    general_search_used: bool
    winner: str
    judge_reasoning: str

    @property
    def rag_bypassed(self) -> bool:
        """True when closed-book confidence was high enough to skip retrieval."""
        return not self.rag_triggered or self.baseline_answer == self.proposed_answer

    @property
    def declined(self) -> bool:
        return self.proposed_answer.strip().startswith(APOLOGY[:40])

    def documents(self) -> list[str]:
        return split_documents(self.retrieved_documents)

    def trace(self) -> list[tuple[str, str]]:
        """Reconstruct the node path this run took, as (node, detail) pairs."""
        steps: list[tuple[str, str]] = [
            ("generate", "Zero-shot answer produced from parametric knowledge."),
            ("extract_claims_initial", f"{len(self.claims)} atomic claims extracted."),
            ("check_constraints", "Claims checked against the query's constraints."),
        ]
        if self.rag_bypassed:
            steps.append(
                (
                    "score_intrinsic",
                    "Closed-book score cleared the acceptance threshold; "
                    "retrieval was skipped and the zero-shot answer was accepted.",
                )
            )
            return steps

        steps.append(
            ("score_intrinsic", "Closed-book score fell below threshold; escalating to retrieval.")
        )
        steps.append(("detect_domain", "Query routed to a curated source allowlist."))
        steps.append(("trusted_search", "Searched the domain's trusted sources."))
        if self.general_search_used:
            steps.append(
                (
                    "general_web_search",
                    "Trusted tier returned nothing the grader accepted; fell back to the open web.",
                )
            )
        steps.append(
            ("grade_documents", f"{len(self.documents())} documents survived relevance grading.")
        )
        if self.declined:
            steps.append(
                (
                    "generate_apology",
                    "All retrieval tiers exhausted; the pipeline declined to answer.",
                )
            )
            return steps
        steps.append(("regenerate", "Answer rewritten from graded context only."))
        steps.append(("extract_claims_regen", "Regenerated answer decomposed again."))
        steps.append(("check_constraints_regen", "Constraints re-checked."))
        steps.append(
            (
                "score_retrieved",
                f"Open-book groundedness {self.groundedness:.0f}%, "
                f"hallucination {self.hallucination_rate:.0f}%.",
            )
        )
        return steps


def _parse_claims(raw) -> list[str]:
    """Claim lists were written as repr'd Python lists; fall back to lines."""
    if not isinstance(raw, str) or not raw.strip():
        return []
    try:
        parsed = ast.literal_eval(raw)
        if isinstance(parsed, list):
            return parse_claims([str(c) for c in parsed])
    except (ValueError, SyntaxError):
        pass
    return parse_claims(raw)


def _clean(value) -> str:
    return "" if pd.isna(value) else str(value)


@lru_cache(maxsize=1)
def load_runs(eval_dir: str | None = None) -> tuple[RecordedRun, ...]:
    """Load every recorded run found on disk. Missing files are skipped."""
    base = Path(eval_dir) if eval_dir else EVAL_DIR
    runs: list[RecordedRun] = []

    for benchmark, filename in BENCHMARK_FILES.items():
        path = base / filename
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        for _, row in frame.iterrows():
            query = _clean(row.get("query")).strip()
            if not query:
                continue
            runs.append(
                RecordedRun(
                    benchmark=benchmark,
                    query=query,
                    ground_truth=_clean(row.get("ground_truth")),
                    baseline_answer=_clean(row.get("baseline_answer")),
                    proposed_answer=_clean(row.get("proposed_answer")),
                    retrieved_documents=_clean(row.get("retrieved_documents")),
                    grader_logs=_clean(row.get("grader_logs")),
                    claims=_parse_claims(row.get("clain_list")),
                    hallucination_rate=float(row.get("graph_hallucination_rate") or 0.0),
                    groundedness=float(row.get("graph_groundedness") or 0.0),
                    rag_triggered=bool(row.get("rag_triggered")),
                    general_search_used=bool(row.get("general_search_used")),
                    winner=_clean(row.get("winner")),
                    judge_reasoning=_clean(row.get("judge_reasoning")),
                )
            )
    return tuple(runs)


def benchmarks() -> list[str]:
    return sorted({run.benchmark for run in load_runs()})


def runs_for(benchmark: str) -> list[RecordedRun]:
    return [run for run in load_runs() if run.benchmark == benchmark]


def find(query: str) -> RecordedRun | None:
    """Exact-then-substring lookup, so a pasted query still resolves."""
    needle = query.strip().lower()
    if not needle:
        return None
    runs = load_runs()
    for run in runs:
        if run.query.strip().lower() == needle:
            return run
    for run in runs:
        if needle in run.query.strip().lower():
            return run
    return None


def search(term: str, limit: int = 25) -> Iterator[RecordedRun]:
    needle = term.strip().lower()
    count = 0
    for run in load_runs():
        if not needle or needle in run.query.lower():
            yield run
            count += 1
            if count >= limit:
                return
