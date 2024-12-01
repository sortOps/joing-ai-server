FROM python:3.11-slim AS build

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src /app/src

FROM python:3.11-slim

WORKDIR /app

# Create docker config directory first
RUN mkdir -p /kaniko/.docker && \
    echo '{"auths":{}}' > /kaniko/.docker/config.json && \
    chmod 600 /kaniko/.docker/config.json

COPY --from=build /usr/local/lib/python3.11/site-packages/ /usr/local/lib/python3.11/site-packages/
COPY --from=build /usr/local/bin/ /usr/local/bin/
COPY --from=build /app/src /app/src

ENV NODE_ENV=production
ENV DOCKER_CONFIG=/kaniko/.docker
ENV PYTHONPATH=/app/src \
    AWS_REGION=ap-northeast-2 \
    PORT=8000

EXPOSE ${PORT}

CMD ["bash", "-c", "uvicorn src.main:app --host 0.0.0.0 --port ${PORT}"]
