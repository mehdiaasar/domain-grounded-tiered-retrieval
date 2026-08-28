# Runs the demo app. Build and run with no local Python setup:
#   docker build -t dgtr . && docker run -p 8501:8501 dgtr
# It starts in replay mode with no keys; pass them to enable live runs:
#   docker run -p 8501:8501 -e GROQ_API_KEY=... -e TAVILY_API_KEY=... dgtr

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/src

WORKDIR /app

# Dependencies first, so code edits do not invalidate the layer.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY app.py .
COPY .streamlit/ ./.streamlit/
# Recorded evaluation runs power replay mode, so the image works without keys.
COPY evaluation/*.csv ./evaluation/
COPY report/pipeline.png ./report/

# Streamlit writes to its config dir at runtime; run unprivileged.
RUN useradd --create-home --uid 1000 app && chown -R app:app /app
USER app

EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD python -c "import urllib.request;urllib.request.urlopen('http://localhost:8501/_stcore/health')"

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
