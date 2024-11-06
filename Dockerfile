# joing-fastapi/Dockerfile

FROM python:3.12-slim

WORKDIR /app

ARG OPENAI_API_KEY
ENV OPENAI_API_KEY=$OPENAI_API_KEY
ENV PYTHONPATH=/app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY ./src ./src

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
