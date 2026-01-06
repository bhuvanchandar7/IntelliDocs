#!/bin/bash

# Function to kill processes on exit
cleanup() {
    echo "Stopping IntelliDocs..."
    kill $(jobs -p)
    exit
}

trap cleanup EXIT

echo "Starting IntelliDocs RAG System..."

# 1. Start Backend
echo "-> Starting Backend (FastAPI)..."
source .venv/bin/activate
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload &
BACKEND_PID=$!

# Wait for backend to be ready
sleep 3

# 2. Start Frontend
echo "-> Starting Frontend (React/Vite)..."
cd frontend
npm run dev &
FRONTEND_PID=$!

echo "=================================================="
echo "   IntelliDocs is running!"
echo "   Backend: http://localhost:8000/docs"
echo "   Frontend: http://localhost:5173"
echo "=================================================="

wait
