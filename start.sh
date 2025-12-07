#!/bin/bash
# Start both API and worker processes on the same machine

# Start the arq worker in the background
echo "[startup] Starting arq worker..."
arq api.worker.WorkerSettings &
WORKER_PID=$!

# Give worker a moment to start
sleep 2

# Start the API server (foreground)
echo "[startup] Starting uvicorn API server..."
exec uvicorn api.main:app --host 0.0.0.0 --port 8080
