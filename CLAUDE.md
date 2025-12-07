# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Grok Underrated Recruiter** - A talent discovery system that analyzes X (Twitter) social graphs to find exceptional candidates for AI companies. It uses xAI's Grok API for candidate evaluation and builds a knowledge graph from X interactions.

## Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   React UI      │────▶│  FastAPI        │────▶│  Redis Worker   │
│   (Vite)        │     │  Backend        │     │  (arq)          │
└─────────────────┘     └─────────────────┘     └─────────────────┘
        │                       │                       │
        └───────────────────────┼───────────────────────┘
                                ▼
                    ┌───────────────────────┐
                    │  Data Layer           │
                    │  - SQLite (local)     │
                    │  - JSON files (evals) │
                    │  - CSV (graph data)   │
                    └───────────────────────┘
```

### Key Components

- **`src/`** - Core Python modules for data pipeline
  - `graph_builder.py` - Builds knowledge graph from X interactions
  - `deep_evaluator.py` - Deep candidate evaluation using xAI SDK with web_search/x_search tools
  - `grok_client.py` - Grok API client for fast screening
  - `x_client.py` - X (Twitter) API client
  - `ranking.py` - PageRank and underratedness scoring

- **`api/`** - FastAPI backend
  - `main.py` - All API endpoints (candidates, search, saved, admin)
  - `auth.py` - X OAuth1 authentication flow
  - `worker.py` - Background job processor using arq (Redis queue)
  - `database.py` - SQLite operations for saved candidates and DM history

- **`ui/`** - React frontend (Vite + TypeScript)
  - `src/api/client.ts` - API client with SSE streaming support
  - Components for search, candidate cards, DM composer, graph visualization

## Development Commands

### Backend (Python)
```bash
# Install dependencies
uv sync

# Run API server
uvicorn api.main:app --reload --port 8000

# Run background worker (requires Redis)
arq api.worker.WorkerSettings
```

### Frontend (React)
```bash
cd ui
pnpm install
pnpm dev        # Development server on :5173
pnpm build      # Production build
```

### Data Pipeline (Jupyter)
Run `discovery_pipeline.ipynb` for full pipeline execution: graph building → fast screening → deep evaluation

## Configuration

Environment variables (`.env`):
- `X_API_KEY`, `X_API_KEY_SECRET` - X OAuth credentials
- `XAI_API_KEY` - xAI API key for Grok
- `FRONTEND_URL` - Frontend URL for CORS/OAuth callbacks
- `REDIS_URL` - Redis for job queue (optional for local dev)
- `ADMIN_HANDLES` - Comma-separated X handles for admin access

Seed accounts configured in `seeds.yaml`, evaluation criteria in `criteria.yaml`.

## Data Flow

1. **Graph Discovery**: Start from seed accounts (xAI employees) → expand via following relationships
2. **Fast Screen**: Grok evaluates candidate relevance (stored in `data/evaluations/fast_screen/`)
3. **Deep Evaluation**: Full analysis with search tools (stored in `data/enriched/deep_*.json`)
4. **API Serving**: Backend serves evaluation results, supports NL search with Grok re-ranking

## API Patterns

- Streaming endpoints use SSE (Server-Sent Events) for real-time Grok reasoning
- Search endpoints at `/search/stream` stream thinking traces before returning results
- Admin endpoints require JWT auth via X OAuth (`/auth/x/login`)

## Deployment

Deployed to Fly.io with multi-process setup:
- `app` process: uvicorn serving FastAPI + static UI
- `worker` process: arq background job processor
