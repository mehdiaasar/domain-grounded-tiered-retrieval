"""Curated per-domain source allowlists for the trusted retrieval tier.

Trusted-tier search is restricted to these hosts. The open web is only
consulted after the trusted tier returns nothing the relevance grader accepts.
"""

from __future__ import annotations

TRUSTED_SOURCES: dict[str, list[str]] = {
    "law": [
        "indiankanoon.org",         # India case law & statutes
        "prsindia.org",             # Indian bills & parliamentary tracking
        "law.cornell.edu",          # US federal & state law / Supreme Court
        "casemine.com",             # US/UK/India case law
        "legislation.gov.uk",       # United Kingdom official law
        "eur-lex.europa.eu",        # European Union law
    ],
    "engineering": [
        "arxiv.org",                # Open-access CS/Math/Physics/EE
        "engineering.com",          # General engineering concepts
        "techxplore.com",           # Engineering & tech breakthroughs
        "sciencedaily.com",         # Broad scientific and engineering research
    ],
    "history": [
        "britannica.com",           # Global encyclopedic history
        "worldhistory.org",         # Open-access global history
        "archives.gov",             # US history
        "nationalarchives.gov.uk",  # UK/European history
        "gutenberg.org",            # Primary historical texts and literature
    ],
    "medical": [
        "ncbi.nlm.nih.gov/pmc",     # PubMed Central (free full text)
        "who.int",                  # World Health Organization
        "mayoclinic.org",           # Consumer-friendly diagnostics
        "clevelandclinic.org",      # Structured medical information
        "nhs.uk",                   # UK National Health Service
    ],
    "finance": [
        "investopedia.com",         # Financial definitions and concepts
        "imf.org",                  # International Monetary Fund
        "worldbank.org",            # Global economic data
        "rbi.org.in",               # Reserve Bank of India
        "bloomberg.com",            # Global market news
    ],
    "science": [
        "nature.com",               # Top-tier scientific journal
        "science.org",              # AAAS Science journal
        "nasa.gov",                 # Space and astrophysics
        "phys.org",                 # Physics, earth science, and biology
    ],
    "general": [
        "britannica.com",
        "wikipedia.org",
        "reuters.com",
        "apnews.com",
        "bbc.com",
    ],
}

DOMAINS: tuple[str, ...] = tuple(TRUSTED_SOURCES)


def allowlist_for(domain: str | None) -> list[str]:
    """Return the allowlist for ``domain``, falling back to ``general``."""
    return TRUSTED_SOURCES.get(domain or "general", TRUSTED_SOURCES["general"])
