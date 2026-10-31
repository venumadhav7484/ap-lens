FROM python:3.12-slim
RUN pip install --no-cache-dir uv
WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN uv sync --frozen --no-dev
# Mount ./data (the BPIC19 XES or built Parquet) and ./outputs at runtime.
ENV APLENS_ROOT=/app
ENTRYPOINT ["uv", "run", "--no-dev"]
CMD ["ap-lens-mcp"]
