# Deploying the demo

The app has no server-side state and no database. It is one Streamlit process
that reads the CSVs in `evaluation/` and, when keys are present, calls a chat
model and Tavily. Any host that can run a Python web process will serve it.

All four options below are free.

---

## Keys, and what happens without them

| Key | What it is for | Free tier |
|---|---|---|
| `TAVILY_API_KEY` | both retrieval tiers | [tavily.com](https://app.tavily.com) — 1,000 credits/month |
| `GROQ_API_KEY` | the chat model | [console.groq.com](https://console.groq.com/keys) — free, rate-limited |

Groq is the default because its free tier serves `llama-3.1-8b-instant`, the
same 8B generator the paper evaluates — so the hosted demo runs the model the
results were measured on, not a substitute. `OPENAI_API_KEY` and
`GOOGLE_API_KEY` work too; set `LLM_PROVIDER` to pick.

**With no keys at all the app still works.** It opens in replay mode and
renders any of the 650 recorded evaluation runs — real answers, real retrieved
documents, real grader decisions. A visitor without an API key sees the
pipeline's actual behaviour, not an error page.

Visitors can also paste their own keys into the sidebar. Those live in the
browser session only; they are never written to disk or logged.

---

## Option 1 — Hugging Face Spaces

Free, always-on, and secrets are managed in the UI.

1. Create a Space at [huggingface.co/new-space](https://huggingface.co/new-space):
   SDK **Streamlit**, hardware **CPU basic (free)**.
2. Add `README-space.md` as the Space's `README.md` — Spaces reads its YAML
   header for configuration.
3. Push:

   ```bash
   git remote add space https://huggingface.co/spaces/<your-username>/domain-grounded-tiered-retrieval
   git push space HEAD:main
   ```

4. In **Settings → Variables and secrets**, add `GROQ_API_KEY` and
   `TAVILY_API_KEY` as *secrets*. Skip this and the Space runs in replay mode.

The Space's `evaluation/*.csv` files must be present for replay mode; they are
a few MB and well under the LFS threshold.

## Option 2 — Streamlit Community Cloud

Deploys straight from GitHub, with no separate remote to push to.

1. Sign in at [share.streamlit.io](https://share.streamlit.io) with GitHub.
2. **New app** → this repository → main file `app.py`.
3. **Advanced settings → Secrets**, in TOML:

   ```toml
   GROQ_API_KEY = "gsk_..."
   TAVILY_API_KEY = "tvly-..."
   ```

Streamlit Cloud installs from `requirements.txt` automatically.

## Option 3 — Docker, anywhere

```bash
docker build -t dgtr .
docker run -p 8501:8501 dgtr                      # replay mode
docker run -p 8501:8501 \
  -e GROQ_API_KEY=... -e TAVILY_API_KEY=... dgtr   # live mode
```

Or `docker compose up`, which reads `.env` if you have one. The image runs as
an unprivileged user and exposes a health check on `/_stcore/health`, so it
drops into Cloud Run, Fly.io, Render, or a plain VM without changes.

## Option 4 — Locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

---

## Cost control on a public link

The demo defaults to replay mode, so an idle page costs nothing. If you put
your own key on a public Space, the exposure is bounded by the free tier's own
rate limits — Groq and Tavily both stop serving rather than bill you. To remove
the risk entirely, deploy without secrets: visitors then either explore
recorded runs or paste their own key.

## Putting it on a CV

Link the deployed URL and the repository together, for example:

> **Domain-Grounded Tiered Retrieval** — self-verifying LangGraph agent that
> decomposes its own answers into atomic claims and escalates through a tiered
> retrieval ladder. 83.7% win rate over a zero-shot 8B baseline on TimeQA.
> *Live demo · Code · arXiv:2603.17872*

The demo opens in replay mode, so the link works on first click with nothing to
install and no key to enter.
