FROM python:3.11-slim

WORKDIR /app

# System deps for psycopg2, nginx, curl
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq-dev gcc nginx curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY src/  ./src/

COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY start.sh ./start.sh
RUN chmod +x ./start.sh

EXPOSE 80

HEALTHCHECK CMD curl --fail http://localhost:80/ || exit 1

ENTRYPOINT ["./start.sh"]
