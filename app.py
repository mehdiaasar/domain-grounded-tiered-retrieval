"""Streamlit front end for the domain-grounded tiered-retrieval pipeline.

Two modes:

* **Live** -- runs the real LangGraph pipeline against a configured provider,
  streaming each node as it executes.
* **Replay** -- renders one of the 650 recorded evaluation runs shipped in
  ``evaluation/``. No keys, no network. This is the default so the page is
  never dead for a visitor who has no API key.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))

from dgtr import demo  # noqa: E402
from dgtr.claims import strip_numbering  # noqa: E402
from dgtr.config import DEFAULT_MODELS, PROVIDER_KEY_ENV, Settings  # noqa: E402
from dgtr.graph import NODE_LABELS  # noqa: E402
from dgtr.search import split_documents  # noqa: E402
from dgtr.sources import TRUSTED_SOURCES  # noqa: E402

PAPER_URL = "https://arxiv.org/abs/2603.17872"
REPO_URL = "https://github.com/mehdiaasar/domain-grounded-tiered-retrieval"

RESULTS = pd.DataFrame(
    [
        ("TimeQA v2", 86, 72, 10, 4, 83.7, 86.4),
        ("MMLU Global Facts", 50, 39, 8, 3, 78.0, 66.9),
        ("FreshQA v2", 150, 97, 37, 16, 64.7, 19.2),
        ("TruthfulQA", 150, 82, 56, 12, 54.7, 84.9),
        ("HaluEval General", 150, 75, 45, 30, 50.0, 78.8),
    ],
    columns=["Benchmark", "Queries", "Wins", "Ties", "Losses", "Win rate %", "Groundedness %"],
)

EXAMPLES = [
    "Explain how insulin regulates blood sugar",
    "What happens if you crack your knuckles a lot?",
    "Which position did Richard Sanderson (MP) hold in May 1833?",
    "What is the current repo rate set by the Reserve Bank of India?",
]

st.set_page_config(
    page_title="Domain-Grounded Tiered Retrieval",
    page_icon="🔎",
    layout="wide",
)


# -- sidebar -------------------------------------------------------------


def sidebar() -> tuple[str, Settings]:
    with st.sidebar:
        st.markdown("### Mode")
        mode = st.radio(
            "Mode",
            ["Replay a recorded run", "Run live"],
            label_visibility="collapsed",
            help="Replay needs no keys. Live calls a real model and a real search API.",
        )

        st.markdown("---")
        st.markdown("### Model")
        env_settings = Settings.from_env()
        providers = sorted(DEFAULT_MODELS)
        provider = st.selectbox(
            "Provider",
            providers,
            index=providers.index(env_settings.provider),
            help="Groq's free tier serves llama-3.1-8b-instant, the same 8B "
            "generator the paper evaluates.",
        )
        settings = Settings.from_env(provider=provider)
        model = st.text_input("Model", value=settings.model)
        threshold = st.slider(
            "Acceptance threshold",
            min_value=0,
            max_value=100,
            value=int(settings.accept_threshold),
            help="Claim score at or above which the answer is accepted without "
            "escalating to the next retrieval tier.",
        )
        settings = Settings.from_env(
            provider=provider, model=model, accept_threshold=float(threshold)
        )

        if mode == "Run live":
            st.markdown("### Keys")
            key_env = PROVIDER_KEY_ENV[provider]
            llm_key = ""
            if key_env:
                placeholder = "already set on the server" if settings.api_key else "required"
                llm_key = st.text_input(key_env, type="password", placeholder=placeholder)
            search_placeholder = (
                "already set on the server" if settings.tavily_api_key else "required"
            )
            tavily_key = st.text_input(
                "TAVILY_API_KEY", type="password", placeholder=search_placeholder
            )
            settings = settings.with_keys(llm_key, tavily_key)

            missing = settings.missing()
            if missing:
                st.warning(f"Still needed: {', '.join(missing)}")
                st.caption(
                    "Free tiers: [console.groq.com](https://console.groq.com/keys) for the "
                    "model, [tavily.com](https://tavily.com) for search. Keys typed here stay "
                    "in this browser session and are never stored."
                )
            else:
                st.success("Ready to run live.")

        st.markdown("---")
        st.caption(f"[Paper]({PAPER_URL}) · [Source]({REPO_URL})")

    return mode, settings


# -- rendering helpers ---------------------------------------------------


def metrics_row(score: float, grounded: float, hallucinated: float, unsupported: float | None):
    cols = st.columns(4 if unsupported is not None else 3)
    cols[0].metric("Composite score", f"{score:.0f}/100")
    cols[1].metric("Grounded claims", f"{grounded:.0f}%")
    cols[2].metric("Hallucinated claims", f"{hallucinated:.0f}%")
    if unsupported is not None:
        cols[3].metric("Partially supported", f"{unsupported:.0f}%")


def render_claims(claims: list[str], labels: list[float], reasons: list[str]):
    if not claims:
        st.caption("No claims extracted.")
        return
    icons = {1.0: "grounded", 0.5: "partial", 0.0: "unsupported"}
    rows = []
    for i, claim in enumerate(claims):
        score = labels[i] if i < len(labels) else None
        rows.append(
            {
                "Claim": strip_numbering(claim),
                "Verdict": icons.get(score, "not scored"),
                "Critic reasoning": reasons[i] if i < len(reasons) else "",
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def render_documents(context: str):
    docs = split_documents(context)
    if not docs:
        st.caption("No documents survived relevance grading.")
        return
    for doc in docs:
        first, _, body = doc.partition("\n")
        url = first.replace("Source: ", "").strip()
        with st.expander(url or "source", expanded=False):
            st.write(body.strip()[:2000])


# -- replay mode ---------------------------------------------------------


def replay_view():
    st.markdown(
        "Every run below is a real execution recorded during the paper's evaluation - "
        "the same answers, retrieved documents, and grader decisions that produced the "
        "published numbers. No API key needed."
    )

    left, right = st.columns([1, 2])
    with left:
        benchmark = st.selectbox("Benchmark", demo.benchmarks())
    runs = demo.runs_for(benchmark)
    with right:
        query = st.selectbox(
            "Query", [run.query for run in runs], help=f"{len(runs)} recorded runs"
        )

    run = next(r for r in runs if r.query == query)

    if run.ground_truth and run.ground_truth != "N/A":
        st.caption(f"**Ground truth:** {run.ground_truth}")

    st.markdown("#### Answers")
    baseline_col, proposed_col = st.columns(2)
    with baseline_col:
        st.markdown("**Baseline** (zero-shot, no retrieval)")
        st.info(run.baseline_answer or "_empty_")
    with proposed_col:
        st.markdown("**Pipeline** (tiered retrieval + claim verification)")
        if run.declined:
            st.warning(run.proposed_answer)
        else:
            st.success(run.proposed_answer or "_empty_")

    verdict_style = {"Proposed": st.success, "Baseline": st.error}.get(run.winner, st.info)
    verdict_style(f"**Judge: {run.winner}** — {run.judge_reasoning}")

    if run.rag_bypassed:
        st.info(
            "Retrieval was bypassed on this query: closed-book claim scoring cleared "
            "the threshold, so the pipeline accepted its own zero-shot answer. Not "
            "every query needs retrieval."
        )
    elif run.general_search_used:
        st.warning(
            "The trusted tier returned nothing the relevance grader accepted, so this "
            "answer came from the open-web fallback. Its sources are not on the "
            "curated allowlist."
        )

    metrics_row(run.groundedness, run.groundedness, run.hallucination_rate, None)

    with st.expander(f"Execution trace ({len(run.trace())} nodes)", expanded=False):
        for node, detail in run.trace():
            st.markdown(f"**`{node}`** — {detail}")

    tab_claims, tab_docs, tab_grader = st.tabs(
        ["Extracted claims", "Retrieved documents", "Grader log"]
    )
    with tab_claims:
        if run.claims:
            st.dataframe(
                pd.DataFrame({"Claim": [strip_numbering(c) for c in run.claims]}),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.caption("No claim list recorded for this row.")
    with tab_docs:
        render_documents(run.retrieved_documents)
    with tab_grader:
        st.code(run.grader_logs or "No grader log recorded.", language="text")


# -- live mode -----------------------------------------------------------


def live_view(settings: Settings):
    missing = settings.missing()

    st.markdown(
        f"Runs the real pipeline: `{settings.provider}` / `{settings.model}`, with "
        "Tavily for both retrieval tiers. Each node streams as it executes."
    )

    query = st.text_input("Question", value=st.session_state.get("query", EXAMPLES[0]))
    cols = st.columns(len(EXAMPLES))
    for col, example in zip(cols, EXAMPLES, strict=True):
        if col.button(example[:34] + ("…" if len(example) > 34 else ""), key=example):
            st.session_state["query"] = example
            st.rerun()

    if missing:
        st.error(
            f"Add {', '.join(missing)} in the sidebar to run live, or switch to "
            "replay mode to explore recorded runs."
        )
        return

    if not st.button("Run pipeline", type="primary", disabled=not query.strip()):
        return

    from dgtr.graph import Pipeline

    try:
        pipeline = Pipeline(settings)
    except Exception as exc:
        st.error(f"Could not start the pipeline: {exc}")
        return

    state: dict = {}
    trace_box = st.container()
    progress = st.progress(0.0, text="Starting…")

    try:
        for i, (node, update) in enumerate(pipeline.stream(query)):
            state.update(update)
            label = NODE_LABELS.get(node, node)
            trace_box.markdown(f"**`{node}`** — {label}")
            progress.progress(min(0.95, (i + 1) / 12), text=label)
    except Exception as exc:
        progress.empty()
        st.error(f"Run failed: {exc}")
        return
    progress.progress(1.0, text="Done")

    st.markdown("#### Answers")
    baseline_col, proposed_col = st.columns(2)
    baseline = state.get("baseline_answer", "")
    answer = state.get("answer", "")
    with baseline_col:
        st.markdown("**Baseline** (zero-shot, no retrieval)")
        st.info(baseline or "_empty_")
    with proposed_col:
        st.markdown("**Pipeline** (tiered retrieval + claim verification)")
        st.success(answer or "_empty_")

    if answer == baseline:
        st.info(
            "Retrieval was bypassed: closed-book claim scoring cleared the "
            f"{settings.accept_threshold:.0f}-point threshold, so the zero-shot answer stands."
        )
    if state.get("is_fallback"):
        st.warning(
            "The trusted tier had nothing the grader accepted; this answer came from "
            "the open-web fallback and its sources are not on the curated allowlist."
        )
    if state.get("domain"):
        st.caption(
            f"Domain: **{state['domain']}** → allowlist: "
            f"{', '.join(TRUSTED_SOURCES.get(state['domain'], []))}"
        )

    metrics_row(
        state.get("score", 0.0),
        state.get("groundedness_score", 0.0),
        state.get("hallucination_rate", 0.0),
        state.get("unsupported_rate", 0.0),
    )

    tab_claims, tab_docs, tab_grader = st.tabs(
        ["Claim-level verdicts", "Retrieved documents", "Grader log"]
    )
    with tab_claims:
        render_claims(
            state.get("claims", []),
            state.get("labels", []),
            state.get("claim_reasons", []),
        )
        constraints = state.get("constraint_labels") or []
        if "VIOLATION" in constraints:
            st.warning(
                f"{constraints.count('VIOLATION')} claim(s) violated a constraint in the "
                "query, which is what forced retrieval."
            )
    with tab_docs:
        render_documents(state.get("retrieved_docs", ""))
    with tab_grader:
        st.code(state.get("grader_logs") or "Grader did not run.", language="text")


# -- static views --------------------------------------------------------


def results_view():
    st.markdown(
        "Pairwise LLM-as-judge comparison over 650 queries. Both systems run the same "
        "`llama3.1:8b` generator, so the comparison isolates the retrieval-and-verification "
        "scaffolding rather than raw model capability. Judge: `gemma3:27b`."
    )
    st.dataframe(RESULTS, use_container_width=True, hide_index=True)
    st.bar_chart(
        RESULTS.sort_values("Win rate %").set_index("Benchmark")["Win rate %"],
        horizontal=True,
        height=280,
    )
    st.caption(
        "Gains are largest on temporal-biographical queries (TimeQA) and precise numerical "
        "facts (MMLU) — where an 8B model's parametric knowledge is thinnest."
    )
    st.info(
        "**Reproducibility note.** Four of five win rates regenerate exactly from the "
        "notebooks in `evaluation/`. TimeQA does not: the run prints 48.0% over 150 queries "
        "while the paper reports 83.7% over 86, because the paper's denominator excludes 64 "
        "tie rows. The paper's groundedness column uses factual-answer subsets that are not "
        "computed anywhere in this codebase. Both gaps are documented in the README."
    )


def about_view():
    st.markdown(
        """
The pipeline treats retrieval as a **ladder**, not a reflex.

1. **Answer first.** A zero-shot answer is generated with no retrieval at all.
2. **Decompose.** The answer is split into atomic, decontextualised claims — pronouns
   resolved, temporal and spatial qualifiers preserved, negations kept intact.
3. **Check constraints.** Each claim is validated against the query's own constraints
   (dates, locations, exclusions, false premises). Any violation skips closed-book
   scoring entirely: if the answer already breaks the question's rules, asking the model
   whether it believes itself is pointless.
4. **Score closed-book.** A critic grades each claim from parametric knowledge alone, and
   is deliberately ruthless — anything needing a search engine to confirm scores zero.
   Clear the threshold and the pipeline stops here, having spent nothing on retrieval.
5. **Escalate.** Otherwise the query is routed to one of seven domains and searched
   against that domain's curated allowlist. Retrieved documents pass a binary relevance
   grader before reaching the generator.
6. **Fall back, then decline.** Only if the trusted tier yields nothing usable does the
   open web get consulted. If that also fails, the pipeline declines to answer rather
   than emitting an ungrounded one.
"""
    )

    st.markdown("#### Domain allowlists")
    st.dataframe(
        pd.DataFrame(
            [{"Domain": d, "Trusted sources": ", ".join(s)} for d, s in TRUSTED_SOURCES.items()]
        ),
        use_container_width=True,
        hide_index=True,
    )

    diagram = Path(__file__).parent / "report" / "pipeline.png"
    if diagram.exists():
        st.markdown("#### Architecture")
        st.image(str(diagram), use_container_width=True)


# -- entry point ---------------------------------------------------------


def main():
    st.title("Mitigating LLM Hallucinations through Domain-Grounded Tiered Retrieval")
    st.caption(
        "Md. Asraful Haque, Aasar Mehdi, Maaz Mahboob, Tamkeen Fatima · "
        f"[arXiv:2603.17872]({PAPER_URL}) · [source]({REPO_URL})"
    )

    mode, settings = sidebar()
    tab_run, tab_results, tab_about = st.tabs(["Pipeline", "Results", "How it works"])

    with tab_run:
        if mode == "Run live":
            live_view(settings)
        else:
            replay_view()
    with tab_results:
        results_view()
    with tab_about:
        about_view()


main()
