# joing-ai Dockerfile
FROM python:3.12-slim AS build

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir boto3

COPY src /app/src

ENV PYTHONPATH=/app/src \
    AWS_REGION=ap-northeast-2 \
    PORT=8000

EXPOSE ${PORT}

# HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
#   CMD curl -v http://localhost:${PORT}/ready | jq -e '.status == "ok"' || exit 1

CMD ["bash", "-c", "uvicorn src.main:app --host 0.0.0.0 --port ${PORT}"]

