from pydantic import field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):

    @field_validator('database_url', mode='after')
    def ensure_asyncpg_dialect(cls, v):
        if v and v.startswith('postgresql://'):
            return v.replace('postgresql://', 'postgresql+asyncpg://', 1)
        return v

    # App
    app_name: str = "Sid Sloth"
    debug: bool = False
    secret_key: str = "change-me-in-production"
    shared_password: str = "change-me"  # v1 shared auth
    admin_username: str = "admin"  # username that gets is_admin=True on first seed

    # Analyst persona — public pseudonym used throughout prompts and UI
    analyst_persona: str = "Sid Sloth"

    # Sensitive anonymization rules — JSON array of [pattern, replacement, category] tuples
    # Example: [["\\bRealName\\b", "TheAnalyst", "name"], ...]
    # Set this in .env or Railway env vars. Never commit real names to source code.
    anon_extra_rules: str = ""

    # Extra regex patterns whose matching paragraphs get stripped during Patreon ingestion.
    # JSON array of regex strings. Example: ["marketoracle\\.co\\.uk", "Market Oracle Ltd"]
    identity_strip_patterns: str = ""

    # Political/off-topic signal keywords for Patreon paragraph filtering.
    # JSON array of lowercase strings. Example: ["epstein", "taco", "clown president"]
    political_signals: str = ""

    # Patreon campaign ID for the analyst's campaign
    patreon_campaign_id: str = ""

    # Database
    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/stock_advisor"

    # LLM APIs
    gemini_api_key: str = ""
    deepseek_api_key: str = ""
    openrouter_api_key: str = ""

    # Market data
    finnhub_api_key: str = ""

    # Patreon
    patreon_session_id: str = ""  # browser session cookie, refresh every ~2 weeks

    # Cloudflare R2
    r2_account_id: str = ""
    r2_access_key_id: str = ""
    r2_secret_access_key: str = ""
    r2_bucket_name: str = "stock-advisor-uploads"
    r2_endpoint_url: str = ""  # https://<account_id>.r2.cloudflarestorage.com

    # Rate limiting
    default_daily_token_limit: int = 100_000

    # Price data
    price_cache_ttl_seconds: int = 300  # 5 minutes

    model_config = {
        "env_file": [".env", "../.env"],  # works from backend/ or repo root
        "env_file_encoding": "utf-8",
    }


settings = Settings()
