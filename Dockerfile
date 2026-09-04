# ─────────────────────────────────────────────────────────────
# UPI RADAR — Docker image (API / Dashboard / Agent use this)
# Note: heavy ML deps (tensorflow/prophet) are intentionally kept
# in requirements-ml.txt so containers stay light. Training can be
# run on a dev machine with the full requirements.txt.
# ─────────────────────────────────────────────────────────────
FROM python:3.11-slim

WORKDIR /app

# System deps used by some data/science libs
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential curl && \
    rm -rf /var/lib/apt/lists/*

# Core runtime requirements (fast boot, demo-mode capable)
COPY requirements-runtime.txt ./requirements-runtime.txt
RUN pip install --no-cache-dir -r requirements-runtime.txt

# Optional heavy layers — uncomment when training inside Docker
# COPY requirements.txt ./requirements.txt
# RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p audit_logs reports data/raw data/processed data/synthetic

EXPOSE 8000 8501

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
