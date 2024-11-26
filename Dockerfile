FROM python:3.11-slim AS build

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src /app/src

FROM python:3.11-slim

RUN mkdir -p /root/.aws && \
    chmod 700 /root/.aws

WORKDIR /app

COPY --from=build /usr/local/lib/python3.11/site-packages/ /usr/local/lib/python3.11/site-packages/
COPY --from=build /usr/local/bin/ /usr/local/bin/
COPY --from=build /app/src /app/src

ENV PYTHONPATH=/app/src \
    AWS_REGION=ap-northeast-2 \
    PORT=8000

EXPOSE ${PORT}

CMD ["bash", "-c", "uvicorn src.main:app --host 0.0.0.0 --port ${PORT}"]
