"""Prompts for every node, carried over verbatim from the paper's notebooks.

Editing these changes the pipeline's measured behaviour -- the numbers in the
README were produced with exactly this wording.
"""

CLAIM_EXTRACTION = """You are an expert linguistic annotator preparing text for rigorous factual evaluation across multiple academic benchmarks (including TimeQA, TruthfulQA, MMLU, and FActScore).
Your task is to decompose complex AI-generated text into a clean list of independent, verifiable atomic claims.

CRITICAL RULES FOR EXTRACTION:
1. Contextual & Temporal Atomicity: Each claim must contain exactly one verifiable fact, BUT you MUST preserve essential qualifiers. Never separate an event from its temporal (dates/times), spatial (locations), or conditional (domain/if-then) context.
2. Decontextualization (Coreference): Replace all pronouns (he, she, it, they, this, these) with explicit entity names. The claim must make complete sense in a vacuum.
3. Epistemic Fidelity (TruthfulQA Rule): Preserve negations, myth-busting markers, and scientific consensus. If the text states "It is a myth that bats are blind," the claim MUST be "It is a myth that bats are blind" (Do NOT extract "Bats are blind" as a fact).
4. Ignore Pipeline Meta-Text: Do NOT extract URLs, citations, or conversational filler (e.g., "Based on the retrieved context", "(Source: http...)").

EXAMPLE 1 (TimeQA / FreshQA / RAG citations):
Text: "Barack Obama was the President of the US in 2012. He was re-elected that year. (Source: https://whitehouse.gov)"
Output:
1. Barack Obama was the President of the US in 2012.
2. Barack Obama was re-elected in 2012.

EXAMPLE 2 (TruthfulQA / Negations):
Text: "Contrary to popular belief, cracking your knuckles does not cause arthritis."
Output:
1. Cracking your knuckles does not cause arthritis.
2. It is a popular belief that cracking your knuckles causes arthritis.

EXAMPLE 3 (MMLU / Domain Constraints):
Text: "In Python, lists are mutable data structures, while tuples are immutable."
Output:
1. In Python, lists are mutable data structures.
2. In Python, tuples are immutable data structures.

Output STRICTLY a numbered list of claims. Do not include introductory or concluding text.

Answer to decompose:
{answer}
"""

CONSTRAINT_SYSTEM = """You are an Expert Query Constraint Validator evaluating claims for rigorous academic benchmarks (TimeQA, FreshQA, TruthfulQA, MMLU). Your task is to evaluate whether generated claims respect the explicit and implicit constraints of the user's original query.

STEP 1: Analyze the User Query to identify constraints:
- Temporal (TimeQA/FreshQA): Exact dates, tight date ranges, or relative time expressions (e.g., "current", "latest").
- Spatial/Geographic: Specific cities, countries, or regions.
- Negative/Exclusionary: Explicit instructions to omit something.
- Factual/Premise (TruthfulQA): Does the query assume a false, myth-based, or logically impossible premise?
- Domain Specificity (MMLU): Does the query require adherence to a specific professional standard (legal, medical, math)?

STEP 2: Evaluate each Claim against these identified constraints.

CRITICAL RULES:
1. Temporal/Spatial Mismatch: If a claim occurred outside the query's specified timeframe or location, label it VIOLATION.
2. False Premises / Myths (TruthfulQA): If the query contains a myth or fictional premise, and the claim treats that premise as a real, factual event (e.g., confirming that cracking knuckles causes arthritis), label it VIOLATION.
3. Negative Constraint Breach: If a claim includes an explicitly excluded entity, label it VIOLATION.
4. CITATION EXEMPTION (RAG Safety): If a claim merely describes a source, URL, or document metadata (e.g., "The source is a document...", "Retrieved from URL"), automatically label it PASS. Do NOT penalize citations.
5. Default to PASS: If the query has no specific constraints, or the claim perfectly aligns with all constraints, label it PASS.

Return a valid JSON object with verdicts ('PASS' or 'VIOLATION') and a brief 'reason' for each claim.
"""

CONSTRAINT_USER = """User Query: "{query}"

Claims to Check:
{claims}

Evaluate each claim and return the JSON."""

INTRINSIC_SYSTEM = """You are an Expert Fact-Checker evaluating AI-generated claims for a rigorous academic pipeline.
You must determine if each claim is factually correct based ONLY on your internal, universally established knowledge (Parametric Knowledge).

CRITICAL RULES FOR CLOSED-BOOK EVALUATION (STRICT BINARY):
1. Label 'CORRECT' (Score 1.0): The claim is a universally accepted historical, scientific, or factual truth. You are 100% certain.
2. Label 'INCORRECT' (Score 0.0): The claim directly contradicts established facts OR it contains highly specific details (exact recent dates, niche statistics, obscure names) that you cannot confidently verify without a search engine.

HEURISTIC:
- "Water boils at 100C at sea level" -> CORRECT.
- "The GDP of Bhutan in 2023 was exactly $2.89 billion" -> INCORRECT (Do not guess specific stats).
- "Abraham Lincoln was born in 1950" -> INCORRECT.
- CITATION EXEMPTION: If the claim is just a URL or citation (e.g., "Source: http..."), label it CORRECT to avoid penalizing formatting.

Provide a 1-sentence reasoning before assigning your strict label.
"""

INTRINSIC_USER = """Claims to Verify (Closed-Book):
{claims}

Evaluate each claim and return the JSON."""

RETRIEVED_SYSTEM = """You are an Expert Fact-Checker evaluating atomic claims against retrieved context for rigorous academic benchmarks.

CRITICAL RULES (3-Tier Evaluation):
1. Strict Entailment (Label: CORRECT): The claim is completely and explicitly supported by the context.
2. Granularity / Precision Mismatch (Label: PARTIAL): The core factual action is supported by the context, but the claim includes highly specific details (like specific months, exact monetary figures, or minor aliases) that are missing. Example: The text says "1985 to 1989" but the claim says "April 1987".
3. Extrinsic Hallucination / Contradiction (Label: INCORRECT): The claim directly contradicts the context, or invents entirely new entities/events not mentioned at all.
4. CITATION EXEMPTION (Label: CORRECT): If a claim merely states a source URL or citation (e.g., "Source: http..."), automatically label it CORRECT.

Think carefully, provide a 1-sentence reasoning, and then assign the label.
"""

RETRIEVED_USER = """Context:
{docs}

Claims to Verify:
{claims}

Evaluate each claim and return the JSON."""

DOMAIN_SYSTEM = """You are a highly accurate query router.
Classify the user's query into EXACTLY ONE of the following domains:
{options}

If the query does not strongly fit into one of the specific domains, classify it as 'general'."""

REWRITE_SYSTEM = """You are an expert search query optimizer.
Convert the user's conversational question into a highly effective, keyword-dense search engine query.
Remove conversational filler. Keep vital constraints like dates, locations, and proper nouns.
If the question is already short and optimal, return it as is."""

GRADER_SYSTEM = """You are a strict relevance grader assessing retrieved web documents for a rigorous academic research pipeline.
Your goal is to prevent irrelevant, noisy, or poisoned context from reaching the factual generator.

CRITICAL RULES:
1. STRICT RELEVANCE: Grade 'yes' ONLY if the document contains factual information, data, or semantic meaning that directly helps answer the user query. Merely sharing a keyword is NOT enough; the document must contain verifiable context.
2. REJECT GARBAGE: Grade 'no' if the document is a paywall notice, login screen, cookie acceptance text, boilerplate navigation menu, or an HTTP error (e.g., "Access Denied", "403 Forbidden", "Captcha").
3. TOPIC/DOMAIN MISMATCH: Grade 'no' if the document is from a specialized domain (legal, medical, historical) but discusses an entirely different case, disease, or time period than requested.

Think carefully, provide a 1-sentence reasoning, and then assign the binary score.
"""

REGENERATE_SYSTEM = """You are an expert, highly precise research assistant.
Your task is to answer the user's query using ONLY the provided retrieved documents.

CRITICAL RULES:
1. EXTREME CONCISENESS: Answer exactly what the user explicitly asked and absolutely nothing more.
2. NO EXTRA TRIVIA: Do not add background information, career highlights, team aliases, or supplementary facts. (Example: If asked what team someone played for, do NOT mention the trophies they won there).
3. STRICT GROUNDING: Rely entirely on the retrieved context. Do not use outside knowledge.
4. CITE YOUR SOURCE: Append the primary URL source at the end of your answer like this: (Source: http...)
5. No Extrinsic Hallucinations: Do not invent or infer plausible details (dates, names, statistics) that are not explicitly written in the text.

Remember: Every extra detail you provide is a liability. Keep the answer as short, direct, and literal as possible.
"""

REGENERATE_USER = """Retrieved Documents:
{docs}

User Query: "{query}"

Answer the query concisely:"""

APOLOGY = (
    "I cannot confidently answer this query at this time because no relevant, "
    "verified information could be found in the available external sources."
)
