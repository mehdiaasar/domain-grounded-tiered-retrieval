# Mitigating LLM Hallucinations through Domain-Grounded Tiered Retrieval

A self-verifying LangGraph agent that decomposes its own output into atomic claims, grades each
claim for factual support, and escalates through a tiered retrieval ladder — domain-curated
trusted sources first, open web only as a fallback — before committing to an answer.

The core idea is that not every query needs retrieval, and not every retrieval should draw from
the open web. The pipeline first asks whether the model's parametric knowledge is already
sufficient; only when claim-level scoring says otherwise does it reach outward, and when it does,
it reaches for a domain-appropriate allowlist before anything else.

## Pipeline

```
                          generate  (zero-shot answer)
                              │
                    extract_claims_initial  (decompose into atomic claims)
                              │
                      check_constraints  (PASS / VIOLATION per claim)
                              │
              ┌───────────────┴───────────────┐
         all PASS                        any VIOLATION
              │                                │
       score_intrinsic                         │   (skip closed-book scoring)
      (closed-book grading)                    │
              │                                │
     score ≥ 70 ──► accept                     │
              │                                │
              └──────────► detect_domain ◄─────┘
                              │
                       trusted_search        (domain allowlist, Tavily)
                              │
                       grade_documents       (binary relevance filter)
                              │
              ┌───────────────┴───────────────┐
        relevant                          not relevant
              │                                │
         regenerate                    general_web_search  (fallback tier)
              │                                │
     extract_claims_regen                      └──► grade_documents
              │
     check_constraints_regen
              │
        score_retrieved  ──► score ≥ 70 ? accept : escalate or generate_apology
```

`generate_apology` is a circuit breaker: when every retrieval tier is exhausted, the pipeline
declines to answer rather than emitting an ungrounded one.

### Domain routing

Queries are classified into one of seven domains — `law`, `medical`, `finance`, `science`,
`engineering`, `history`, `general` — each mapped to a curated source allowlist (for example
`law` → indiankanoon.org, law.cornell.edu, legislation.gov.uk, eur-lex.europa.eu). Trusted-tier
retrieval is restricted to that allowlist; the open web is only consulted after the trusted tier
returns nothing the relevance grader accepts.

### Scored signals

The graph state carries `hallucination_rate`, `unsupported_rate`, and `groundedness_score`
alongside the composite `score`, so a run can be audited claim-by-claim rather than only at the
answer level.

## Results

LLM-as-judge pairwise comparison against a `gpt-4o-mini` zero-shot baseline, 150 questions per
benchmark (50 for MMLU global facts):

| Benchmark | Proposed | Tie | Baseline | Win rate vs. non-tie |
|---|---:|---:|---:|---:|
| FreshQA | 97 | 37 | 16 | 85.8% |
| TruthfulQA | 82 | 56 | 12 | 87.2% |
| HaluEval (general) | 75 | 45 | 30 | 71.4% |
| MMLU (global facts) | 39 | 8 | 3 | 92.9% |
| TimeQA | 72 | 74 | 4 | 94.7% |

TimeQA produces by far the highest tie rate (74/150). Per-row inspection shows `rag_triggered`
and `general_search_used` are `True` for all 150 rows, so the ties are not a routing failure —
they concentrate in questions where both systems land on the same answer and the judge sees no
factual daylight between them.

Raw judge outputs, including per-row reasoning, retrieved documents, grader logs, and extracted
claim lists, are in `evaluation/`.

> **Note on the results files:** the `_rm_constraint` / `_rem_ceck_const` suffixes mark the
> ablation runs with the constraint-checking node disabled. The `ground_truth` column in the
> TimeQA file is `N/A` throughout — that run was judged pairwise, not against gold labels.

## Repository layout

```
final_graph.ipynb                        main pipeline (LangGraph, GPT-4o-mini)
claim_based_critic batch_v2.ipynb        batched claim critic, current version
claim_based_critic batch_v2_ollama.ipynb same critic against local Ollama models
claim_based_v2_domain_retriever.ipynb    domain classifier + trusted-source retriever
generating_queries.ipynb                 synthetic eval query generation
evaluation/final_graph.ipynb             evaluation harness (LLM judge)
evaluation/*.csv                         per-benchmark judge outputs, 150 questions each
datasets/                                benchmark inputs (small subsets; see below)
report/                                  algorithm write-up, evidence notes, pipeline figure
diagram/                                 architecture diagrams (PNG + SVG)
```

## Setup

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env    # then fill in your keys
```

Required keys in `.env`:

| Variable | Used for |
|---|---|
| `OPENAI_API_KEY` | generation, claim extraction, scoring, LLM judge |
| `TAVILY_API_KEY` | trusted-tier and general web search |
| `GOOGLE_API_KEY` | optional — Gemini comparison runs |

Every notebook loads these via `load_dotenv()` in its first cell. **No keys are hardcoded; `.env`
is gitignored and must never be committed.**

## Datasets

Small benchmark files are included under `datasets/` for reproducibility. The TimeQA splits are
excluded — `timeqa_train.csv` alone is 280 MB, past GitHub's 100 MB per-file limit. Fetch them
from the [TimeQA release](https://github.com/wenhuchen/Time-Sensitive-QA) and drop them into
`datasets/` to reproduce the TimeQA runs.

## Reference work

Not redistributed here; consult the original sources:

- Song et al., *Measuring and Enhancing Trustworthiness of LLMs in RAG* (arXiv:2502.14302)
- arXiv:2512.01852, arXiv:2601.09929
- *Hallucination Mitigation in LLMs* (AI, MDPI, 6(11):260)
