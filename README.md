# Mitigating LLM Hallucinations through Domain-Grounded Tiered Retrieval

Reference implementation for the paper:

> **Mitigating LLM Hallucinations through Domain-Grounded Tiered Retrieval**
> Md. Asraful Haque, Aasar Mehdi, Maaz Mahboob, Tamkeen Fatima
> arXiv:[2603.17872](https://arxiv.org/abs/2603.17872) [cs.CL] — v1 18 Mar 2026, v2 25 Mar 2026

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

## Evaluation setup

Pairwise LLM-as-judge comparison over 650 queries from five benchmarks:

| Role | Model |
|---|---|
| Baseline | `llama3.1:8b`, zero-shot |
| Proposed | `llama3.1:8b` driving the full tiered-retrieval pipeline |
| Judge | `gemma3:27b` |

Both systems run the same 8B generator, so the comparison isolates the contribution of the
retrieval-and-verification scaffolding rather than raw model capability. All three run locally
via Ollama — the harness in `evaluation/final_graph.ipynb` has the OpenAI configuration
commented out in favour of the local models. Win rate is the percentage of queries where the
judge preferred the pipeline's answer over the zero-shot baseline.

## Results

Reproducing Table 3 of the paper:

| Benchmark | Queries | Wins | Ties | Losses | Win rate | Groundedness |
|---|---:|---:|---:|---:|---:|---:|
| TimeQA v2 | 86 | 72 | 10 | 4 | **83.7%** | 86.4% |
| MMLU Global Facts | 50 | 39 | 8 | 3 | **78.0%** | 66.9% |
| FreshQA v2 | 150 | 97 | 37 | 16 | **64.7%** | 19.2% |
| TruthfulQA | 150 | 82 | 56 | 12 | **54.7%** | 84.9% |
| HaluEval General | 150 | 75 | 45 | 30 | **50.0%** | 78.8% |

Gains are largest on temporal-biographical queries (TimeQA) and precise numerical facts (MMLU) —
exactly the cases where an 8B model's parametric knowledge is thinnest and grounded retrieval
contributes most. The paper identifies **False-Premise Overclaiming** as the main residual
failure mode and proposes pre-retrieval answerability estimation as future work.

Raw judge outputs — per-row reasoning, retrieved documents, grader logs, and extracted claim
lists — are in `evaluation/`.

### Provenance

Every reported CSV was produced by the two run notebooks in `evaluation/`, which are the
artifacts submitted with the paper. Their saved cell outputs contain the complete per-query
execution trace (node-by-node state updates for all 650 queries) plus the summary block printed
at the end of each run:

| Notebook | Benchmarks run |
|---|---|
| `final_graph copy.ipynb` | FreshQA (150), TruthfulQA (150) |
| `final_graph copy 2.ipynb` | TimeQA (150), MMLU (50), HaluEval (150) |

### What the code prints vs. what the paper reports

`run_evaluation_study` computes win rate as `wins / total_valid`, where `total_valid` counts
every row judged `Proposed`, `Tie`, or `Baseline` — no subsetting. Its printed output:

| Benchmark | Win rate (printed) | Win rate (paper) | Groundedness (printed) | Groundedness (paper) |
|---|---:|---:|---:|---:|
| FreshQA v2 | 64.7% | 64.7% ✓ | 7.2% | 19.2% |
| TruthfulQA | 54.7% | 54.7% ✓ | 8.9% | 84.9% |
| MMLU Global Facts | 78.0% | 78.0% ✓ | 35.0% | 66.9% |
| HaluEval General | 50.0% | 50.0% ✓ | 53.0% | 78.8% |
| **TimeQA v2** | **48.0%** | **83.7%** | 23.1% | 86.4% |

Four of five win rates reproduce exactly. **TimeQA does not:** the run prints 48.0% over 150
queries (72 W / 74 T / 4 L), while the paper reports 83.7% over 86. Wins (72) and losses (4)
agree, so it is the same run — the paper's denominator excludes 64 of the 74 tie rows.

Groundedness in the paper is uniformly higher than the raw means the code prints, consistent
with the paper's stated use of "factual-answer subsets". That subsetting step is **not present
in this codebase** — it was applied downstream of these notebooks, so neither the TimeQA
denominator nor the paper's groundedness column can be regenerated from this repository as it
stands. Anyone re-running these notebooks will get the "printed" column.

> **Note on filenames:** the `_rm_constraint` / `_rem_ceck_const` suffixes mark runs with the
> constraint-checking node disabled. The `ground_truth` column in the TimeQA file is `N/A`
> throughout — that run was judged pairwise, not against gold labels.
>
> **Tie semantics:** `run_evaluation_study` assigns `Tie` mechanically, without consulting the
> judge, whenever `baseline_ans == proposed_ans` — i.e. whenever the pipeline accepted its own
> zero-shot answer. Ties are therefore "RAG bypassed or produced identical text", not "judge
> found the answers equally good". This is why TimeQA's tie count is high (74/150) despite
> `rag_triggered` being `True` on every row.

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

To reproduce the reported evaluation you also need [Ollama](https://ollama.com) with both models
pulled:

```bash
ollama pull llama3.1:8b && ollama pull gemma3:27b
```

Required keys in `.env`:

| Variable | Used for |
|---|---|
| `TAVILY_API_KEY` | trusted-tier and general web search — needed for every run |
| `OPENAI_API_KEY` | the GPT-4o-mini pipeline variant in `final_graph.ipynb`; not used by the reported eval |
| `GOOGLE_API_KEY` | optional — Gemini comparison runs |

Note that the root `final_graph.ipynb` is configured for `gpt-4o-mini`, while the harness in
`evaluation/final_graph.ipynb` is configured for the local Ollama models used in the paper.

Every notebook loads these via `load_dotenv()` in its first cell. **No keys are hardcoded; `.env`
is gitignored and must never be committed.**

## Datasets

Small benchmark files are included under `datasets/` for reproducibility. The TimeQA splits are
excluded — `timeqa_train.csv` alone is 280 MB, past GitHub's 100 MB per-file limit. Fetch them
from the [TimeQA release](https://github.com/wenhuchen/Time-Sensitive-QA) and drop them into
`datasets/` to reproduce the TimeQA runs.

## Citation

```bibtex
@misc{haque2026mitigating,
  title         = {Mitigating LLM Hallucinations through Domain-Grounded Tiered Retrieval},
  author        = {Haque, Md. Asraful and Mehdi, Aasar and Mahboob, Maaz and Fatima, Tamkeen},
  year          = {2026},
  eprint        = {2603.17872},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  url           = {https://arxiv.org/abs/2603.17872}
}
```

## Reference work

Not redistributed here; consult the original sources:

- Song et al., *Measuring and Enhancing Trustworthiness of LLMs in RAG* (arXiv:2502.14302)
- arXiv:2512.01852, arXiv:2601.09929
- *Hallucination Mitigation in LLMs* (AI, MDPI, 6(11):260)
