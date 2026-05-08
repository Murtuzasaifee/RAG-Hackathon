FROM python:3.13-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev

COPY src/ src/

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "app.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
