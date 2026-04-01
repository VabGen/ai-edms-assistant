FROM python:3.13-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    libpq-dev \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p /app/prompts /app/logs /app/db_init_sql
COPY --chown=root:root entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

EXPOSE 8002

HEALTHCHECK --interval=15s --timeout=10s --retries=5 \
    CMD curl -sf http://localhost:8002/health || exit 1

ENTRYPOINT ["/app/entrypoint.sh"]
