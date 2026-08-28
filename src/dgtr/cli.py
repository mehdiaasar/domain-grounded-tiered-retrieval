"""Command-line entry point: ``python -m dgtr "your question"``."""

from __future__ import annotations

import argparse
import json
import sys

from dgtr import demo
from dgtr.claims import strip_numbering
from dgtr.config import DEFAULT_MODELS, Settings
from dgtr.graph import NODE_LABELS


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dgtr",
        description="Domain-grounded tiered retrieval: ask a question and watch "
        "the pipeline verify its own answer.",
    )
    parser.add_argument("query", nargs="?", help="The question to answer.")
    parser.add_argument(
        "--provider",
        choices=sorted(DEFAULT_MODELS),
        help="LLM provider (default: auto-detected from the keys present).",
    )
    parser.add_argument("--model", help="Override the provider's default model.")
    parser.add_argument(
        "--threshold",
        type=float,
        help="Claim score at or above which an answer is accepted (default 70).",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Replay a recorded evaluation run instead of calling any API.",
    )
    parser.add_argument("--json", action="store_true", help="Emit the final state as JSON.")
    parser.add_argument(
        "--list-demos", action="store_true", help="List a sample of replayable queries."
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.list_demos:
        for run in list(demo.search("", limit=20)):
            print(f"[{run.benchmark}] {run.query}")
        return 0

    if not args.query:
        build_parser().print_help()
        return 2

    if args.offline:
        return _run_offline(args.query, as_json=args.json)

    overrides = {}
    if args.provider:
        overrides["provider"] = args.provider
    if args.model:
        overrides["model"] = args.model
    if args.threshold is not None:
        overrides["accept_threshold"] = args.threshold

    settings = Settings.from_env(**overrides)
    missing = settings.missing()
    if missing:
        print(
            f"Missing credentials: {', '.join(missing)}\n"
            f"Set them in .env, or run with --offline to replay a recorded run.",
            file=sys.stderr,
        )
        return 1

    from dgtr.graph import Pipeline

    pipeline = Pipeline(settings)
    print(f"provider={settings.provider} model={settings.model}\n")

    state: dict = {}
    for node, update in pipeline.stream(args.query):
        print(f"-> {node}: {NODE_LABELS.get(node, '')}")
        state.update(update)

    if args.json:
        print(json.dumps(_serialisable(state), indent=2, default=str))
        return 0

    print(f"\nAnswer:\n{state.get('answer', '(none)')}")
    if state.get("claims"):
        print(f"\nClaims ({len(state['claims'])}):")
        for claim in state["claims"]:
            print(f"  - {strip_numbering(claim)}")
    print(
        f"\nScore {state.get('score', 0):.0f}/100 | "
        f"grounded {state.get('groundedness_score', 0):.0f}% | "
        f"hallucinated {state.get('hallucination_rate', 0):.0f}%"
    )
    if state.get("is_fallback"):
        print(
            "\nNote: the trusted tier had nothing usable; this answer came from "
            "an open-web fallback and its sources are not on the curated allowlist."
        )
    return 0


def _run_offline(query: str, *, as_json: bool) -> int:
    run = demo.find(query)
    if run is None:
        print(
            "No recorded run matches that query. Try --list-demos for replayable ones.",
            file=sys.stderr,
        )
        return 1

    if as_json:
        print(json.dumps(run.__dict__, indent=2, default=str))
        return 0

    print(f"Replaying a recorded {run.benchmark} run (no API calls).\n")
    for node, detail in run.trace():
        print(f"-> {node}: {detail}")
    print(f"\nBaseline (zero-shot):\n{run.baseline_answer}")
    print(f"\nPipeline answer:\n{run.proposed_answer}")
    print(f"\nJudge verdict: {run.winner} - {run.judge_reasoning}")
    return 0


def _serialisable(state: dict) -> dict:
    return {k: v for k, v in state.items() if not callable(v)}


if __name__ == "__main__":
    raise SystemExit(main())
