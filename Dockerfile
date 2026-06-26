FROM node:20-slim AS frontend-builder

WORKDIR /src/frontend

COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci

COPY frontend/ ./
ENV NEXT_PUBLIC_BACKEND_URL=""
RUN npm run build

FROM python:3.11-slim

WORKDIR /app

ENV APP_ENV=local \
    NEUROSIGHT_RUNTIME_MODE=demo \
    NEUROSIGHT_FRONTEND_DIR=/app/frontend/out \
    PORT=7860 \
    PYTHONPATH=/app \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-space.txt /app/
RUN pip install --upgrade pip \
    && pip install -r requirements-space.txt

COPY . /app
COPY --from=frontend-builder /src/frontend/out /app/frontend/out

RUN addgroup --system neurosight \
    && adduser --system --ingroup neurosight --home /app neurosight \
    && chown -R neurosight:neurosight /app

USER neurosight

EXPOSE 7860

CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-7860}"]
