# joing-fastapi/Dockerfile

FROM python:3.11-slim AS build

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade -r requirements.txt


FROM python:3.11-slim

WORKDIR /app

COPY --from=build /usr/local/lib/python3.12/site-packages/ /usr/local/lib/python3.12/site-packages/
COPY --from=build /usr/local/bin/ /usr/local/bin/

ARG OPENAI_API_KEY
ENV OPENAI_API_KEY=$OPENAI_API_KEY
ENV PYTHONPATH=/app/src

COPY ./src ./src

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
