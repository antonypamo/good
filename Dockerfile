FROM python:3.10-slim

ENV APP_HOME=/app \
    PORT=8000

WORKDIR ${APP_HOME}

# Install dependencies first for better layer caching
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and artifacts
COPY app.py config.json logreg_rrf_savant.joblib ./

# Expose the application port
EXPOSE ${PORT}

# Start the FastAPI service
CMD ["/bin/sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT}"]
