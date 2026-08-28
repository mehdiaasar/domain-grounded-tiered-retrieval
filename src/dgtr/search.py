"""Tavily-backed retrieval for both tiers.

Every failure path returns an empty context string rather than prose. An
apologetic sentence in ``retrieved_docs`` would be graded, scored, and cited
as if it were a document -- the "context bleed" bug the notebooks call out.
"""

from __future__ import annotations

from typing import Any

from dgtr.config import Settings


class SearchUnavailable(RuntimeError):
    """Raised when no Tavily key is configured."""


def build_search_tool(settings: Settings) -> Any:
    if not settings.tavily_api_key:
        raise SearchUnavailable(
            "Tavily key missing. Set TAVILY_API_KEY or paste one in the sidebar."
        )
    from langchain_tavily import TavilySearch

    return TavilySearch(
        tavily_api_key=settings.tavily_api_key,
        search_depth="advanced",
        max_results=settings.max_search_results,
    )


def format_results(results: Any) -> tuple[str, list[str]]:
    """Flatten a Tavily response into a context string and its source URLs."""
    if isinstance(results, dict) and "results" in results:
        docs = results["results"]
    elif isinstance(results, list):
        docs = results
    else:
        docs = []

    blocks, urls = [], []
    for item in docs:
        if isinstance(item, dict) and "content" in item:
            url = item.get("url", "Unknown")
            blocks.append(f"Source: {url}\n{item['content']}")
            urls.append(url)

    return ("\n\n".join(blocks) if blocks else "", urls)


def split_documents(context: str) -> list[str]:
    """Split a formatted context string back into individual documents."""
    if not context:
        return []
    parts = [chunk.strip() for chunk in context.split("Source: ") if chunk.strip()]
    return [f"Source: {chunk}" for chunk in parts]
