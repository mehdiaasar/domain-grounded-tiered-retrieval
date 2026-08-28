"""Parsing of the claim-extractor's output.

The extractor is asked for a bare numbered list, but smaller instruction-tuned
models routinely wrap it in chatter -- "Here is the decomposed output:" before,
a bulleted restatement of the rules after. The notebooks kept every non-empty
line, so that chatter became pseudo-claims that were then constraint-checked
and fact-scored. Since an unverifiable line scores 0, it dragged the composite
score down and pushed queries into retrieval that did not need it.

Keeping only the numbered lines fixes that. When the model emits no numbered
line at all the raw lines are kept, so nothing is silently dropped.
"""

from __future__ import annotations

import re

_NUMBERED = re.compile(r"^\s*\d+[.)]\s+")


def parse_claims(raw: str | list[str]) -> list[str]:
    """Return the atomic claims in ``raw``, discarding model preamble."""
    lines = raw.split("\n") if isinstance(raw, str) else [str(item) for item in raw]
    lines = [line.strip() for line in lines if line.strip()]
    numbered = [line for line in lines if _NUMBERED.match(line)]
    return numbered or lines


def strip_numbering(claim: str) -> str:
    """Drop the leading ``1. `` from a claim, for display."""
    return _NUMBERED.sub("", claim).strip()
