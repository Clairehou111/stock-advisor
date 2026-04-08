# Stock Advisor — Architecture & Design Document

> An AI-powered private stock assistant that translates a complex analyst's investment
> philosophy into accessible, actionable advice.

---

## 1. Product Vision

### What This Is

Stock Advisor translates the investment philosophy of a private analyst (anonymized as
"Sid Sloth") into conversational, actionable advice. Sid Sloth publishes dense spreadsheets
and monthly Patreon posts covering PE-based valuation analysis of AI and tech stocks. Most
users — including the product owner — find this material hard to follow.

**The primary value is translation**: taking opaque analyst reasoning and making it
conversational, contextual, and grounded in the user's actual portfolio.

### What This Is Not

- Not a stock screener or generic financial chatbot
- Not a system that generates its own market opinions
- In V1, it strictly represents what the analyst has stated — it never derives
  predictions for stocks the analyst hasn't covered
- This constraint is deliberate: rushing to derive predictions before deeply
  understanding the philosophy would reproduce the kind of shallow mainstream
  thinking the analyst's philosophy rejects

### The Persona

Responses are delivered through a **direct, clear, professional advisor persona** — cutting
through noise and giving practical, data-driven guidance without jargon. The system prompt
enforces no celebrity impersonation; the persona is grounded in Sid Sloth's methodology.

### Core Value Propositions

1. **Translation** — Dense analyst reasoning becomes conversational explanations
2. **Dual-timeframe reconciliation** — Long-term spreadsheet data and short-term
   monthly documents often appear to conflict; the system explains the timeframe
   difference and reconciles them
3. **Portfolio context** — Every stock is analyzed relative to the user's actual
   holdings (concentration, cost basis, category balance), never in isolation
4. **Deterministic math + LLM reasoning** — Zone detection, trim guidance, and PE
   positioning are computed deterministically; the LLM explains the "why"
5. **Learning over time** — The philosophy corpus grows with each document; principles
   are distilled, refined, and confidence-scored

---

## 2. System Architecture

### Component Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                  FRONTEND (Streamlit)                        │
│  Chat UI  │  Admin / Ingestion Interface                     │
│  frontend/app.py — Login, Chat, Admin pages                  │
└─────┬─────┴────────────────────────┬────────────────────────┘
      │                              │
      ▼                              ▼
  POST /api/chat            POST /api/admin/ingest/*
  GET /api/auth/login       GET /api/admin/ingest/status/{id}
      │                              │
      └──────────────┬───────────────┘
                     │
          ┌──────────▼───────────┐
          │     FastAPI App      │
          │    (app/main.py)     │
          │  lifespan: pgvector  │
          │  + table create_all  │
          │  + startup seeding   │
          └──────────┬───────────┘
                     │
    ┌────────────────┼──────────────────┐
    │                │                  │
┌───▼────┐  ┌────────▼───────┐  ┌───────▼────────┐
│ Chat   │  │   Ingestion    │  │   Core /       │
│ API    │  │   Pipelines    │  │   Services     │
│chat.py │  │ Excel / Doc /  │  │ price_service  │
│        │  │ Patreon        │  │ rate_limiter   │
└───┬────┘  └────────┬───────┘  │ embedding_svc  │
    │                │          │ earnings_svc   │
    └────────────────┴──────────┘
                     │
         ┌───────────▼───────────┐
         │   PostgreSQL 16        │
         │   + pgvector           │
         │  (all tables, RAG)     │
         └───────────────────────┘

External APIs:
  • OpenRouter (Qwen3 235B A22B — primary chat)
  • DeepSeek V3 (fallback chat, rephrase, signal extraction, summarization)
  • Gemini 2.5 Flash (metadata extraction, chart analysis)
  • text-embedding-004 / Gemini embedding (768-dim vectors)
  • Finnhub (real-time stock prices, 60 req/min free tier)
  • yfinance (PE ratio, index prices, fallback price)
  • Cloudflare R2 (file storage for Excel, PDFs, Patreon images)
```

### Key Architectural Decisions

| Decision | Rationale |
|---|---|
| pgvector over dedicated vector DB | Single database simplifies ops; volume is modest (thousands of chunks) |
| Deterministic math before LLM | Prevents hallucinated numbers; zones and trim % are computed, not generated |
| Three-layer anonymization | Pre-storage scrub + system prompt rules + output post-check |
| Async SQLAlchemy + asyncpg | Non-blocking DB during LLM wait times (which dominate latency) |
| Qwen3 primary, DeepSeek fallback | OpenRouter for primary; DeepSeek as economy/fallback |
| Background task for summarization | `asyncio.create_task()` inside FastAPI endpoint — safe within the running event loop |
| Zero-price caching suppression | Prevents stale zero prices from blocking real fetches on subsequent requests |

---

## 3. Data Model

### Tables (defined in `app/models/tables.py`, created via `Base.metadata.create_all` at startup)

| Table | Purpose | Key Columns |
|---|---|---|
| `users` | Auth, preferences | id, username, password_hash, daily_token_limit, is_admin, preferred_lang |
| `conversations` | Chat sessions | id, user_id, title, summary, summarized_through, context_map (JSONB) |
| `messages` | Chat turns | id, conversation_id, role, content, model_used, tokens_used, source_ids, tickers_mentioned, metadata_json |
| `portfolio_holdings` | User positions | id, user_id, ticker, shares, avg_cost_basis |
| `stock_predictions` | Analyst data (Excel) | ticker, buy_high/low, sell_start, pe_range_high/low, fair_value, egf*, fundamentals, trend_status, is_current, superseded_by |
| `analyst_chunks` | RAG chunks | id, ticker, chunk_type, content_text, embedding (Vector 768), temporal_scope, metadata_json, outlook_horizon, publish_date, tickers_mentioned, thesis_direction, retrieval_count, avg_relevance, last_retrieved, is_stale |
| `principle_corpus` | Investing principles (static) | id, principle_text, category, source_ids, version |
| `derived_principles` | Principles extracted from posts | id, principle_text, category, confidence_score, times_stated, source_chunk_ids, first_seen, last_reinforced, is_active, superseded_by |
| `upload_sources` | File provenance | id, file_type, r2_key, sheet_name, extracted_json, conflict_report, change_summary, raw_content, upload_timestamp |
| `trade_signals` | Buy/trim/sell signals from posts | id, ticker, action, price_level, confidence, post_id, publish_date, context_text, upload_source_id |
| `rate_limit_usage` | Token tracking | user_id, usage_date, tokens_used, queries_count |
| `anonymization_rules` | Dynamic scrub rules | original_term (unique), replacement, category |
| `price_cache` | Price/PE cache (5-min TTL) | ticker (PK), price, pe_ratio, fetched_at |
| `entity_aliases` | Alias → symbol cache | alias (PK, lowercase), resolved_type, resolved_value, created_at |

### Entity Relationships

```
users 1──* conversations 1──* messages
users 1──* portfolio_holdings
users 1──* rate_limit_usage

upload_sources 1──* stock_predictions
upload_sources 1──* analyst_chunks
upload_sources 1──* trade_signals

analyst_chunks *──* derived_principles (via source_chunk_ids)

stock_predictions ── price_cache (joined on ticker at query time)
entity_aliases (standalone lookup table, grown at runtime by LLM entity extraction)
```

### Schema Management

There are **no Alembic migrations**. All tables are created with `Base.metadata.create_all`
at app startup inside the lifespan handler. This is simple and safe for the current scale.
New columns added to models will not be auto-migrated — the table must be dropped/recreated
or a manual `ALTER TABLE` run.

---

## 4. Ingestion Pipelines

### Excel Spreadsheet Ingestion (`scripts/ingest_excel.py` + `app/ingestion/excel_parser.py`)

Triggered via `POST /api/admin/ingest/excel` (file upload) or CLI: `python -m scripts.ingest_excel <path.xlsx>`

1. `parse_workbook()` (friend's code, `app/ingestion/excel_parser.py`) extracts stocks, principles, diffs
2. Create `UploadSource` record, mark all existing `StockPrediction.is_current = False`
3. Per stock: anonymize text fields (`strategy_text`, `trend_status`), rephrase via DeepSeek to neutral voice
4. Insert new `StockPrediction` rows with `is_current=True`
5. Upsert `PrincipleCorpus` entries (match by title prefix, increment version if existing)
6. Return `{stock_count, principle_count, upload_source_id}`

### Patreon Post Ingestion (`app/ingestion/patreon_parser.py`)

Triggered via `POST /api/admin/ingest/patreon` (post URL or numeric ID). Supports `force=True` to wipe and re-ingest.

1. Fetch post JSON from Patreon API using `session_id` cookie
2. Save raw `content_json_string` to `UploadSource.raw_content`
3. Extract ProseMirror nodes (text + image) from content JSON
4. Filter: remove identity markers (URLs, copyright, Patreon refs) and political-only paragraphs
5. Split into sections using CONTENTS heading structure
6. Extract trade signals per section via DeepSeek (structured JSON: ticker, action, price_level, confidence)
7. Per section: accumulate text until image node, then flush as chunk:
   - Rephrase text to neutral voice (DeepSeek)
   - Download chart image → upload to R2
   - Analyze chart via Gemini 2.5 Flash vision
   - Extract metadata (tickers, temporal_scope, horizon, thesis_direction, chunk_type) via Gemini 2.5 Flash
   - Create `AnalystChunk` record
8. Embed all new chunks in batch (text-embedding-004 / 768-dim)
9. Store chunks with embeddings to DB
10. Distill principles from new chunk IDs (`app/jobs/distill_principles.py`)
11. Resume support: tracks already-processed sections, skips them on retry; always re-extracts signals

### Document Ingestion (`app/ingestion/doc_parser.py`)

Triggered via `POST /api/admin/ingest/doc` (PDF, .txt, or .md upload).

1. Read file (PDF via pypdf, or plain text)
2. Chunk into 300–500 token segments at paragraph boundaries; falls back to sentence boundaries for oversized paragraphs
3. Anonymize all chunks via `Anonymizer.scrub()`
4. Rephrase each chunk to neutral voice (DeepSeek, 0.5s delay between calls)
5. Extract metadata per chunk via Gemini 2.5 Flash (1 req/sec rate limit)
6. Embed all chunks in batch
7. Create `UploadSource` + `AnalystChunk` records

### Admin Background Task Pattern

All three ingestion pipelines run as async background tasks dispatched with `asyncio.create_task()`.
The endpoint immediately returns a `task_id`; the frontend polls `GET /api/admin/ingest/status/{task_id}`.
Task state is held in the in-memory `_tasks` dict (not persisted — lost on restart).

---

## 5. Chat Pipeline

### Request Lifecycle (`app/api/chat.py`)

```
User sends POST /api/chat {message, conversation_id?}
  │
  ▼
[1] AUTH — validate JWT, load User from DB
  │
  ▼
[2] RATE LIMIT — check daily tokens
      <80%:  Qwen3 (primary)
      80–100%: DeepSeek (degraded)
      >100%: 429 reject
  │
  ▼
[3] ENTITY EXTRACTION
    a. DB lookup: query n-grams against entity_aliases (instant)
    b. LLM fallback (DeepSeek): only when unresolved financial terms remain,
       short unresolved tokens exist, or prev_tickers present but no match
       — LLM also returns needs_history + is_social flags
       — New aliases persisted back to entity_aliases for future queries
  │
  ▼
[4] SHORT-CIRCUIT: is_social=true
    → tiny LLM call (60 max_tokens), save messages, return
  │
  ▼
[5] COVERAGE CHECK — split tickers into covered / uncovered
    If needs_history and no tickers found: carry prev_tickers from history
  │
  ▼
[6] CONVERSATION HISTORY — ticker-aware filtering
    - Most recent turn: always included verbatim
    - Older turns: include verbatim if tickers overlap; else compress to marker
    - Per-ticker context_map injected for currently discussed tickers
    - Flat summary used if no context_map available
  │
  ▼
[7] INTENT CLASSIFICATION → max_tokens
    portfolio_review: 2000 | ticker_analysis: 1500 | philosophy: 1000 | casual: 400
  │
  ▼
[8] CONTEXT BUILDING (covered tickers)
    - Load StockPrediction from DB
    - Fetch live price (Finnhub) + PE (yfinance) concurrently
    - Run decision engine (deterministic zone/trim math)
    - Format stock context + metrics strings
  │
  ▼
[9] CONTEXT BUILDING (uncovered tickers)
    - Fetch live price + PE
    - Inject as general framework note (no analyst-specific zones)
  │
  ▼
[10] RAG — embed query → cosine search analyst_chunks
     Re-rank with ticker boost (+0.1 per overlapping ticker)
     Update retrieval_count, avg_relevance, last_retrieved
  │
  ▼
[11] LOAD PRINCIPLES
     PrincipleCorpus (all) + DerivedPrinciple (active, top 20 by confidence)
  │
  ▼
[12] SYSTEM PROMPT ASSEMBLY
     Principles + stock context + decision metrics + RAG chunks +
     portfolio holdings + earnings calendar + index prices +
     focus directive (if conversation history present)
  │
  ▼
[13] LLM CALL (Qwen3 via OpenRouter, fallback DeepSeek)
     max_tokens = intent-based budget
     Qwen3: sends max_tokens + THINKING_BUDGET (10,000) to API
  │
  ▼
[14] POST-PROCESSING
     - Anonymizer.post_check() (Layer 3: flag URLs/emails in output)
     - Fact check: ticker drift detection + price sanity (>50% off = flag)
     - Ticker drift: prepend redirect header if response discusses wrong tickers
  │
  ▼
[15] SAVE
     - Record token usage (rate_limiter)
     - Create/update Conversation record
     - Save user + assistant Messages with tickers_mentioned + metadata_json annotations
  │
  ▼
[16] BACKGROUND (non-blocking, asyncio.create_task)
     If conversation has ≥ _SUMMARIZE_AFTER_TURNS * 2 messages:
       Summarize older turns → Conversation.summary + context_map
```

---

## 6. Anonymization System

Three layers protect the analyst's identity at different points in the data flow.

### Layer 1 — Pre-Storage Scrub (`app/ingestion/anonymizer.py`)

Applied during ingestion before any data is written to DB. The `Anonymizer` class compiles
regex rules from:

1. `DEFAULT_SCRUB_RULES` (hardcoded, generic — Patreon URLs, YouTube links, email patterns, platform refs)
2. `_RUNTIME_RULES` (sensitive — loaded at startup via `set_runtime_rules()`)

**Important**: `Anonymizer()` compiles patterns at `__init__` time by merging
`DEFAULT_SCRUB_RULES + _RUNTIME_RULES`. If you instantiate `Anonymizer()` before
`load_anon_rules_into_memory()` has run, the instance will have no sensitive rules.
All ingestion code (doc_parser, ingest_excel, patreon_parser) instantiates `Anonymizer()`
inside the pipeline function (not at module level), so startup ordering is safe.
The module-level `anonymizer = Anonymizer()` in `chat.py` IS instantiated at import time —
but this is only for the output post-check (Layer 3), which uses generic `FLAG_PATTERNS`
patterns, not sensitive replacement rules. The scrub rules on the post-check instance do
include the runtime rules… which means the chat.py module-level instance will only have
rules from when the module was first imported (at startup). Since startup seeding happens
in the lifespan `before` any requests are served, this is safe in practice.

### Layer 2 — System Prompt Rules (`app/llm/prompts.py`)

Hard rules in the LLM system prompt:
- Never reveal the analyst's real identity; deflect if asked
- Never reference Patreon, YouTube, or personal identifiers
- Never quote strategy text or analyst notes verbatim — always paraphrase
- Never invent person names or attribute funds/ETFs to persons not in the data

### Layer 3 — Output Post-Check (`Anonymizer.post_check()`)

Applied to every LLM response before returning to the user. Flags (but does not always
replace) residual URLs and email addresses. Flags are logged as warnings; the `scrub()`
pass also applies all replacement rules.

### Runtime Rules Loading Sequence (startup)

```
lifespan() → ensure_anon_rules_seeded(db)  — seeds DB from ANON_EXTRA_RULES env var
           → load_anon_rules_into_memory(db) — calls set_runtime_rules()
```

The `ANON_EXTRA_RULES` env var is a JSON array of `[pattern, replacement, category]` tuples.
Example: `[["\\bJohnDoe\\b", "TheAnalyst", "name"]]`. Set in Railway environment variables.
Also: `IDENTITY_STRIP_PATTERNS` (extra URL/domain patterns for Patreon paragraph filtering)
and `POLITICAL_SIGNALS` (keywords for off-topic paragraph removal during Patreon ingestion).

---

## 7. LLM Routing

### Model Assignments

| Task | Model | Mechanism |
|---|---|---|
| Chat (primary) | Qwen3 235B A22B | OpenRouter (`qwen/qwen3-235b-a22b`) |
| Chat (fallback/economy) | DeepSeek V3 | Direct DeepSeek API (`deepseek-chat`) |
| Rephrase (ingestion) | DeepSeek V3 | Direct DeepSeek API |
| Signal extraction (ingestion) | DeepSeek V3 | Direct DeepSeek API |
| Entity extraction (chat) | DeepSeek V3 | Direct DeepSeek API (8s timeout) |
| Conversation summarization | DeepSeek V3 | Direct DeepSeek API |
| Metadata extraction | Gemini 2.5 Flash | google-genai SDK (sync, via asyncio.to_thread) |
| Chart analysis | Gemini 2.5 Flash | google-genai SDK with vision |
| Embeddings | text-embedding-004 | `app/services/embedding_service.py` |

### Qwen3 Thinking Budget

`THINKING_BUDGET = 10000` tokens is reserved for Qwen3's chain-of-thought. The API call
sends `max_tokens = output_budget + 10000`. For a casual query (400 output tokens), this
sends 10,400 to the API. This is intentional — Qwen3's thinking tokens are separate from
output tokens and the budget caps the reasoning cost. The thinking output is filtered from
the response (the `call_openrouter` return path uses `message.content`, which excludes
thinking blocks).

### Token Budgets (rate limiting)

- Default daily limit: 100,000 tokens/user (configurable per-user)
- At 80%: degrade to DeepSeek V3 (`is_degraded=True`)
- At 100%: reject with 429
- Reset: daily (new date = new `rate_limit_usage` row; no cron needed)

### Max Output Tokens (by intent)

| Query Type | Max Output Tokens |
|---|---|
| Portfolio review | 2,000 |
| Ticker analysis | 1,500 |
| Philosophy question | 1,000 |
| Casual / social | 400 (social short-circuit: 60) |

---

## 8. Entity Resolution

The entity resolution system eliminates manual alias maintenance.

### Two-Stage Approach

1. **DB lookup (fast path)**: All n-grams (1-, 2-, 3-word) from the query are batch-fetched
   from `entity_aliases`. Common financial words are pre-seeded at startup (`_SEED_ALIASES`
   in `app/core/security.py`).

2. **LLM fallback**: DeepSeek is called when the DB has unresolved short tokens that look
   like tickers, or when conversation context exists but no tickers were found (potential
   follow-up). The LLM also returns `needs_history` and `is_social` flags.

3. **Alias persistence**: New aliases found by the LLM are written back to `entity_aliases`
   so future queries skip the LLM call.

---

## 9. Deployment

### Platform: Railway

The backend (FastAPI + PostgreSQL) and frontend (Streamlit) are deployed to Railway.

### Required Environment Variables

| Variable | Description |
|---|---|
| `DATABASE_URL` | PostgreSQL connection string (`postgresql+asyncpg://...`) |
| `SECRET_KEY` | JWT signing secret |
| `SHARED_PASSWORD` | V1 shared login password |
| `ADMIN_USERNAME` | Username auto-promoted to admin on startup |
| `ANALYST_PERSONA` | Public pseudonym (default: "Sid Sloth") |
| `OPENROUTER_API_KEY` | For Qwen3 235B via OpenRouter |
| `DEEPSEEK_API_KEY` | For fallback chat, rephrase, entity extraction, summarization |
| `GEMINI_API_KEY` | For metadata extraction, chart analysis, embeddings |
| `FINNHUB_API_KEY` | Real-time stock prices |
| `PATREON_SESSION_ID` | Browser session cookie (refresh ~every 2 weeks) |
| `PATREON_CAMPAIGN_ID` | Analyst's Patreon campaign ID |
| `R2_ACCOUNT_ID` | Cloudflare R2 account |
| `R2_ACCESS_KEY_ID` | R2 credentials |
| `R2_SECRET_ACCESS_KEY` | R2 credentials |
| `R2_BUCKET_NAME` | R2 bucket (default: `stock-advisor-uploads`) |
| `R2_ENDPOINT_URL` | `https://<account_id>.r2.cloudflarestorage.com` |
| `ANON_EXTRA_RULES` | JSON array of `[pattern, replacement, category]` — sensitive identity rules |
| `IDENTITY_STRIP_PATTERNS` | JSON array of regex strings — extra domains to strip in Patreon ingestion |
| `POLITICAL_SIGNALS` | JSON array of keywords for off-topic paragraph filtering |
| `DEFAULT_DAILY_TOKEN_LIMIT` | Per-user daily token cap (default: 100,000) |
| `PRICE_CACHE_TTL_SECONDS` | Price cache TTL (default: 300 = 5 min) |

### Frontend

Streamlit app (`frontend/app.py`). The `API_BASE` is hardcoded to `http://localhost:8000/api`
and should be set to the Railway backend URL for production. The JWT is persisted in the URL
`?t=<jwt>` query param so page refreshes don't require re-login.

---

## 10. Project Structure

```
stock-advisor/
├── app/
│   ├── config.py                     # Settings, env vars (pydantic-settings)
│   ├── main.py                       # FastAPI app, lifespan, CORS, startup seeding
│   ├── api/
│   │   ├── auth.py                   # POST /api/auth/login → JWT
│   │   ├── chat.py                   # POST /api/chat — full chat pipeline
│   │   └── admin.py                  # POST /api/admin/ingest/* — ingestion endpoints
│   ├── core/
│   │   ├── decision_engine.py        # Deterministic math: zones, trim, PE position
│   │   └── security.py              # JWT, bcrypt, startup seeding functions
│   ├── db/
│   │   └── session.py               # Async engine + session factory
│   ├── ingestion/
│   │   ├── anonymizer.py            # 3-layer anonymization (Layer 1 + 3)
│   │   ├── doc_parser.py            # PDF/text ingestion pipeline
│   │   ├── excel_parser.py          # Spreadsheet parser (friend's code)
│   │   └── patreon_parser.py        # Patreon post ingestion pipeline
│   ├── jobs/
│   │   └── distill_principles.py    # Principle distillation from chunks
│   ├── llm/
│   │   ├── orchestrator.py          # Model routing: Qwen3, DeepSeek, Gemini Flash
│   │   └── prompts.py               # System prompt templates
│   ├── models/
│   │   ├── base.py                  # DeclarativeBase + TimestampMixin
│   │   └── tables.py               # All ORM models
│   └── services/
│       ├── earnings_service.py      # Upcoming earnings calendar
│       ├── embedding_service.py     # text-embedding-004 via Gemini
│       ├── price_service.py         # Finnhub + yfinance price/PE fetching
│       └── rate_limiter.py          # Daily token budget enforcement
├── scripts/
│   └── ingest_excel.py              # CLI: load Excel workbook → DB
├── frontend/
│   └── app.py                       # Streamlit UI (Login + Chat + Admin pages)
├── docker-compose.yml
├── Dockerfile
├── pyproject.toml
└── .env.example
```

---

## 11. Known Issues and TODOs (from code review, April 2026)

### Bugs

**B1. `scripts/ingest_excel.py` CLI path never commits (data silently discarded)**
File: `scripts/ingest_excel.py`, lines 220–225.
When called from the CLI (`db=None`), the code opens `async with async_session() as session:`
and calls `await _run(session)` but never calls `await session.commit()`. SQLAlchemy's
`async_sessionmaker` does not auto-commit on context manager exit. All inserts and updates
are rolled back when the `async with` block exits. The API path is correct (caller commits).
Fix: add `await session.commit()` before the `async with` block closes.

**B2. `_summarize_older_messages` receives an un-entered session context manager**
File: `app/api/chat.py`, line 1078.
```python
asyncio.create_task(
    _summarize_older_messages(uuid.UUID(conv_id), async_session())
)
```
`async_session` is an `async_sessionmaker`. Calling `async_session()` returns an
`AsyncSession` object that has NOT been entered (i.e., `__aenter__` not called). The
function `_summarize_older_messages` then calls `await db.execute(...)` on this un-entered
session, which will raise `sqlalchemy.exc.InvalidRequestError` or similar.
Fix: wrap it in a helper coroutine that properly enters the session:
```python
async def _run_summarize():
    async with async_session() as db:
        await _summarize_older_messages(uuid.UUID(conv_id), db)
asyncio.create_task(_run_summarize())
```

**B3. `logger` used but never defined in `app/core/security.py`**
File: `app/core/security.py`, lines 201, 219, 235.
`logging` is imported but `logger = logging.getLogger(__name__)` is missing. The three
`logger.warning/info` calls in `ensure_anon_rules_seeded` and `load_anon_rules_into_memory`
will raise `NameError: name 'logger' is not defined` at runtime if those code paths execute.
Fix: add `logger = logging.getLogger(__name__)` after the imports.

### Security

**S1. CORS wildcard in production**
File: `app/main.py`, line 47.
`allow_origins=["*"]` is flagged with a comment "tighten in production" but has not been
tightened. Since this is a private app on Railway, restrict to the Streamlit frontend URL.

**S2. JWT stored in URL query param**
File: `frontend/app.py`, `_save_token_to_url()`.
The JWT is appended as `?t=<jwt>` in the URL for bookmark/refresh persistence. This means
the token appears in browser history, server access logs, and referrer headers. Acceptable
for a private internal tool, but worth noting. A `localStorage`-backed approach would be
more secure if the frontend is ever made public.

**S3. `secret_key` and `shared_password` have insecure defaults**
File: `app/config.py`, lines 7–8.
`secret_key = "change-me-in-production"` and `shared_password = "change-me"` are the
defaults. If Railway env vars are not set, the app runs with known secrets. The app has
no startup check or warning for this. Consider adding a startup assertion:
`assert settings.secret_key != "change-me-in-production", "Set SECRET_KEY env var"`.

### Correctness

**C1. `_get_identity_patterns()` and `_get_political_signals()` reparse JSON on every call**
Files: `app/ingestion/patreon_parser.py`, lines 54–75.
Both functions call `json.loads(settings.identity_strip_patterns)` and
`json.loads(settings.political_signals)` every time they are invoked. They are called
inside `_is_identity()` and `_is_political()`, which are called once per node in
`_filter_nodes()`. For a post with hundreds of nodes, this is O(N) JSON parses of the same
string. `settings` values are immutable after startup so these could be module-level
constants. Low performance risk for current post volumes (hundreds of nodes), but wasteful.
Fix: parse and cache at module load time (or use `functools.lru_cache`).

**C2. Price cache: zero prices not cached → excessive API calls for delisted/invalid tickers**
File: `app/services/price_service.py`, lines 114–129.
Zero prices are deliberately not cached (correct for valid tickers that fail transiently).
However, if a user repeatedly asks about a delisted ticker or an invalid symbol, every
request triggers a full Finnhub + yfinance round-trip. This could exhaust Finnhub's 60
req/min free-tier limit. Consider caching zero-price results with a shorter TTL (e.g. 60s)
to rate-limit repeated failures.

**C3. `_classify_intent` computes `persona_lc` inside the function on every call**
File: `app/api/chat.py`, `_classify_intent()`.
`persona_lc = settings.analyst_persona.lower()` is computed on every call. Trivial
cost, but it could be a module-level constant since `settings` is immutable after startup.
Style issue only.

**C4. `_get_stock_data` uses `ilike` instead of exact match**
File: `app/api/chat.py`, line 337.
`StockPrediction.ticker.ilike(f"%{ticker}%")` will match "NVDA" for a query for "AMD"
if "AMD" appears in a stock name field or if "NVDA" contains the queried substring.
Tickers are exact symbols; this should be an exact match or case-insensitive equality:
`StockPrediction.ticker.ilike(ticker)` (no wildcards).

**C5. Patreon `flush_chunk` closure captures mutable `created_chunks` by reference**
File: `app/ingestion/patreon_parser.py`, lines 577–608.
`flush_chunk` is a nested `async def` that mutates `created_chunks` via `nonlocal`. This
works correctly in Python because `nonlocal` properly references the enclosing variable,
but it's an unusual pattern that could cause subtle bugs if the function is refactored.
Minor style issue.

### Consistency

**CS1. Three copies of the `_REPHRASE_PROMPT`**
Files: `scripts/ingest_excel.py`, `app/ingestion/doc_parser.py`, `app/ingestion/patreon_parser.py`.
All three define nearly identical rephrasing prompts. Should be extracted to a shared module
(e.g., `app/ingestion/prompts.py` or added to `app/llm/prompts.py`).

**CS2. Three near-identical `_rephrase()` async HTTP functions**
Same three files. All call the DeepSeek API with slightly different `max_tokens` (200 vs 300 vs 600)
and `trust_env` values (False vs True). Should be consolidated into a single utility.

**CS3. R2 upload logic duplicated between `patreon_parser.py` and `admin.py`**
Both files define their own boto3 S3 client setup. Should be extracted to `app/storage/`.

**CS4. `ModelTier.GEMINI_FLASH` is defined in the enum but only called directly, never via `select_model`**
File: `app/llm/orchestrator.py`. The `select_model()` function never returns `ModelTier.GEMINI_FLASH`.
Gemini Flash is only used for metadata extraction (outside the main chat routing). The enum
entry is slightly misleading but not harmful.

**CS5. Docstring in `orchestrator.py` references Claude Sonnet but code routes to Qwen3**
File: `app/llm/orchestrator.py`, the `chat()` function docstring says
`"Claude Sonnet 4.6 (OpenRouter) → Qwen3 235B (OpenRouter)"` but `QWEN3_MODEL` is the
primary and there is no Claude routing. The docstring is stale.

### Dead Code

**D1. `url` variable in `fetch_post_json` is constructed but never used**
File: `app/ingestion/patreon_parser.py`, lines 399–407.
`url` (campaign URL) is built but the function only uses `post_url`. Dead code.

**D2. `KNOWN_TICKERS` global in `chat.py` is declared but never populated**
File: `app/api/chat.py`, line 84.
`KNOWN_TICKERS: set[str] = set()  # populated at startup from DB` — there is no code
that populates this set. It is never read either. Dead variable.

**D3. `_COMMON_WORDS` could exclude legitimate ticker-like words**
File: `app/api/chat.py`, `_COMMON_WORDS` set includes `"price"`, `"market"`, `"stock"`,
`"stocks"`, `"fund"`, `"etf"`, `"crypto"`, `"buy"`, `"sell"` — these are in the exclusion
list for n-gram unresolved filtering. This means a user typing just "buy etf" won't trigger
the LLM entity resolver. This is likely intentional (filter noise) but the comment says
these are "common words" — having financial terms like "buy" and "sell" mixed in with
stop words is confusing. Low impact.

---

## 12. V2 Roadmap

**V2.1: Prediction Outcome Tracking**
- Daily job: compare stock_predictions zones vs. actual prices
- Record zone crossings in `prediction_outcomes` table
- Surface accuracy metrics in admin dashboard

**V2.2: Similarity Engine + Coverage Expansion**
- Feature vectors for covered stocks (sector, growth, PE, market cap, EGF)
- K-nearest covered stocks for uncovered tickers
- Derive zones, buy/sell ranges by weighted analogy
- Label clearly as "Framework-derived, not analyst-stated"

**V2.3: Analogy Weights + Validation**
- On spreadsheet upload: batch re-derive, compare, score
- Update analogy weights based on accuracy
- Surface confidence in chat responses

**V2.4: Streaming Responses**
- WebSocket or SSE for progressive LLM output

**V2.5: Alembic Migrations**
- Replace `create_all` with Alembic for proper schema versioning
- Enables zero-downtime column additions

**V2.6: Chunk Staleness Job**
- Weekly cron: flag `analyst_chunks.is_stale` where `last_retrieved > 90d` AND `publish_date > 6m`
- Stale chunks deprioritized in RAG but not deleted
