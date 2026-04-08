# Sid Sloth — AI Stock Advisor

A private AI-powered stock advisor chatbot built on a specific analyst's investment methodology. Ask about covered stocks, portfolio allocation, PE valuation, and the analyst's framework — all backed by structured data and RAG over analyst commentary.

---

## Features

- **Multi-turn chat** with ticker-aware context — switching tickers mid-conversation works correctly
- **Real-time prices** via Finnhub (30s in-memory cache)
- **Structured stock data** — buy ranges, sell zones, PE targets, EGF, fundamentals scores
- **RAG** — semantic search over analyst commentary using pgvector
- **Ingestion pipelines** — Excel workbook, Patreon posts, PDF/text documents
- **Three-layer anonymization** — scrub, prompt rules, post-generation check
- **Admin UI** — file upload with live progress tracking for all ingestion types
- **Entity resolution** — handles typos, nicknames, and shorthands via DB alias cache + LLM fallback

## Stack

| Layer | Technology |
|-------|-----------|
| Backend | FastAPI + Python 3.11 |
| Database | PostgreSQL 16 + pgvector |
| Frontend | Streamlit |
| Primary LLM | Qwen3-235B-A22B via OpenRouter (extended thinking) |
| Fallback LLM | DeepSeek V3 |
| Embeddings | Gemini text-embedding-004 |
| Prices | Finnhub (real-time) + yfinance (PE ratio) |
| Storage | Cloudflare R2 |

## Project Structure

```
app/
  api/          # FastAPI routers (chat, admin, auth)
  core/         # Decision engine, security, rate limiter
  ingestion/    # Excel, document, Patreon parsers + anonymizer
  jobs/         # Background jobs (principle distillation)
  llm/          # LLM orchestrator, prompt templates
  models/       # SQLAlchemy ORM models
  services/     # Price service, embeddings, earnings
frontend/
  app.py        # Streamlit UI (login, chat, admin)
scripts/        # CLI ingestion tools
tests/          # Unit tests (79 passing)
```

## Setup

### 1. Clone & install

```bash
git clone https://github.com/Clairehou111/stock-advisor.git
cd stock-advisor
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

### 2. Configure environment

```bash
cp .env.example .env
# Fill in your API keys and sensitive values
```

Required env vars:

| Variable | Description |
|----------|-------------|
| `SECRET_KEY` | JWT signing key |
| `SHARED_PASSWORD` | Shared login password for all users |
| `DATABASE_URL` | PostgreSQL connection string |
| `OPENROUTER_API_KEY` | For Qwen3 (primary chat model) |
| `DEEPSEEK_API_KEY` | For fallback + ingestion rephrasing |
| `GEMINI_API_KEY` | For embeddings + chart analysis |
| `FINNHUB_API_KEY` | Real-time stock prices |
| `ANON_EXTRA_RULES` | JSON array of sensitive anonymization rules |
| `PATREON_SESSION_ID` | Browser session cookie for Patreon ingestion |
| `PATREON_CAMPAIGN_ID` | Numeric campaign ID |
| `IDENTITY_STRIP_PATTERNS` | JSON array of regex patterns to strip from posts |
| `POLITICAL_SIGNALS` | JSON array of off-topic keyword signals |

### 3. Start services

```bash
# Start PostgreSQL (or use docker-compose)
docker-compose up -d db

# Run backend
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload --log-config log_config.json

# Run frontend (separate terminal)
streamlit run frontend/app.py
```

### 4. Run tests

```bash
pytest tests/ -q
```

## Deployment (Railway)

1. Create a Railway project and add a PostgreSQL plugin
2. Set all env vars in Railway → Variables
3. Deploy — Railway uses the `Dockerfile` automatically

The app creates all DB tables on startup (no migration needed for fresh deploys).

## Anonymization

Sensitive information (real names, personal URLs, political content) is kept out of source code entirely:

- **`ANON_EXTRA_RULES`** — regex replacement rules loaded from env var into DB at startup
- **`IDENTITY_STRIP_PATTERNS`** — paragraph-level strip patterns for Patreon ingestion
- **`POLITICAL_SIGNALS`** — off-topic keyword filter for Patreon posts

See `.env.example` for the JSON format.

## License

Private — not for redistribution.
