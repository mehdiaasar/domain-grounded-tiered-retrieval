---
title: Domain-Grounded Tiered Retrieval
emoji: 🔎
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: 1.40.0
app_file: app.py
pinned: false
license: mit
short_description: Self-verifying LLM agent that fact-checks its own claims
---

# Mitigating LLM Hallucinations through Domain-Grounded Tiered Retrieval

A self-verifying LangGraph agent that decomposes its own output into atomic
claims, grades each claim for factual support, and escalates through a tiered
retrieval ladder — domain-curated trusted sources first, open web only as a
fallback — before committing to an answer.

**This Space opens in replay mode and needs no API key.** It renders any of the
650 runs recorded during the paper's evaluation: real answers, real retrieved
documents, real grader decisions. Switch to live mode in the sidebar and paste a
[Groq](https://console.groq.com/keys) and [Tavily](https://app.tavily.com) key —
both free — to run the pipeline yourself.

Paper: [arXiv:2603.17872](https://arxiv.org/abs/2603.17872) ·
Code: [GitHub](https://github.com/mehdiaasar/domain-grounded-tiered-retrieval)
