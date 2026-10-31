FROM python:3.12-slim
RUN pip install --no-cache-dir uv
WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN uv sync --frozen --no-dev
ENV APLENS_ROOT=/app PATH="/app/.venv/bin:$PATH"
# Mount ./data (built Parquet) and ./outputs at runtime. Needs ANTHROPIC_API_KEY and APLENS_API_KEY.
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request;urllib.request.urlopen('http://localhost:8000/healthz')"
CMD ["uvicorn", "ap_lens.api:app", "--host", "0.0.0.0", "--port", "8000"]
