FROM python:3.11-slim

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends curl git \
    && rm -rf /var/lib/apt/lists/*

# Copy everything
COPY . /app/env
WORKDIR /app/env

# Install dependencies directly
RUN pip install --no-cache-dir \
    "openenv-core>=0.2.1" \
    "fastapi>=0.110.0" \
    "uvicorn[standard]>=0.27.0" \
    "pydantic>=2.0.0" \
    "python-multipart>=0.0.9" \
    "openai>=1.0.0"

ENV PYTHONPATH="/app/env:$PYTHONPATH"

HEALTHCHECK --interval=10s --timeout=5s --start-period=30s --retries=5 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
