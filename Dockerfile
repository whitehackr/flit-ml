# BNPL ML API Production Dockerfile for Railway Deployment
FROM python:3.11-slim

# Set environment variables for production
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Set working directory
WORKDIR /app

# Install system dependencies for ML libraries
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Poetry configuration (commented out due to Railway build issues)
# RUN pip install poetry==1.6.1
# ENV POETRY_NO_INTERACTION=1 \
#     POETRY_VENV_IN_PROJECT=1 \
#     POETRY_CACHE_DIR=/tmp/poetry_cache
# COPY pyproject.toml poetry.lock ./
# RUN poetry install --only=main && rm -rf $POETRY_CACHE_DIR

# Use requirements.txt for Railway deployment (workaround for Poetry connection issues)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash app \
    && chown -R app:app /app
USER app

# Expose port (Railway will set PORT environment variable)
EXPOSE 8000

# Health check for Railway
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:$PORT/v1/bnpl/health || exit 1

# Start command for Railway deployment (updated for pip-based install)
# CMD poetry run uvicorn flit_ml.api.main:app --host 0.0.0.0 --port $PORT
CMD python -m uvicorn flit_ml.api.main:app --host 0.0.0.0 --port ${PORT:-8000}