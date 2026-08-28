"""Domain-Grounded Tiered Retrieval.

A self-verifying LangGraph agent that decomposes its own output into atomic
claims, grades each claim for factual support, and escalates through a tiered
retrieval ladder before committing to an answer.

Reference implementation for arXiv:2603.17872.
"""

from dgtr.config import Settings
from dgtr.graph import Pipeline, build_graph

__all__ = ["Settings", "Pipeline", "build_graph", "__version__"]
__version__ = "1.0.0"
