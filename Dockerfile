# ─────────────────────────────────────────────
# Dockerfile — Anomaly Detection API
# ─────────────────────────────────────────────

# Start from an official Python image
# "slim" = stripped down version, smaller file size (~150MB vs ~900MB)
FROM python:3.11-slim

# Set working directory inside the container
# All subsequent commands run from here
WORKDIR /app

# Copy requirements first (separate layer = faster rebuilds)
# Docker caches layers — if requirements.txt hasn't changed,
# it won't re-install packages on the next build
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the code
COPY . .

# Tell Docker this container listens on port 8000
EXPOSE 8000

# Command to run when container starts
# 0.0.0.0 = accept connections from outside the container
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
