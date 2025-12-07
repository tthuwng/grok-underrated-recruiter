# Grok Underrated Recruiter - FastAPI Backend + Worker + UI
# Stage 1: Build the UI
FROM node:20-alpine AS ui-builder

# Install pnpm
RUN corepack enable && corepack prepare pnpm@latest --activate

WORKDIR /ui

# Copy UI package files
COPY ui/package.json ui/pnpm-lock.yaml* ./

# Install dependencies
RUN pnpm install --frozen-lockfile || pnpm install

# Copy UI source code
COPY ui/ ./

# Build UI with production API URL
ENV VITE_API_URL=/api
RUN pnpm run build

# Stage 2: Python backend
FROM python:3.13-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies (all in requirements.txt)
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY api/ ./api/
COPY src/ ./src/
COPY seeds.yaml ./
COPY criteria.yaml ./

# Copy data directory (for initial migration - can be removed after migration)
COPY data/processed/ ./data/processed/
COPY data/enriched/ ./data/enriched/

# Copy built UI from builder stage
COPY --from=ui-builder /ui/dist ./static

# Create writable data directories for worker outputs
RUN mkdir -p ./data/evaluations/fast_screen

# Create non-root user for security
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8080

# Health check (only applies to web process)
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')" || exit 1

# Default command (can be overridden by fly.toml processes)
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8080"]
