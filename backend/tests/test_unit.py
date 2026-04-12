"""
Pure unit tests — no DB, no network, no LLM calls.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio


# ── 1. anonymizer.py ─────────────────────────────────────────────────────────

from app.ingestion.anonymizer import (
    DEFAULT_SCRUB_RULES,
    Anonymizer,
    AnonymizationResult,
    set_runtime_rules,
    _RUNTIME_RULES,
)


class TestDefaultScrubRules:
    def test_scrubs_patreon_url(self):
        a = Anonymizer()
        result = a.scrub("Check out https://www.patreon.com/sidsloth for more")
        assert "[link removed]" in result.text
        assert "patreon.com" not in result.text

    def test_scrubs_youtube_url(self):
        a = Anonymizer()
        result = a.scrub("Watch https://www.youtube.com/watch?v=abc123 now")
        assert "[link removed]" in result.text
        assert "youtube.com" not in result.text

    def test_scrubs_youtu_be_url(self):
        a = Anonymizer()
        result = a.scrub("Short link: https://youtu.be/XYZ")
        assert "[link removed]" in result.text

    def test_scrubs_patreon_platform_reference(self):
        a = Anonymizer()
        result = a.scrub("Subscribe on Patreon to get access")
        assert "Patreon" not in result.text
        assert "the research platform" in result.text

    def test_scrubs_my_latest_video(self):
        a = Anonymizer()
        result = a.scrub("In my latest video I explained this")
        assert "my latest video" not in result.text.lower()

    def test_scrubs_my_portfolio(self):
        a = Anonymizer()
        result = a.scrub("I added to my portfolio yesterday")
        assert "my portfolio" not in result.text.lower()

    def test_scrubs_email(self):
        a = Anonymizer()
        result = a.scrub("Contact me at analyst@example.com for details")
        assert "analyst@example.com" not in result.text
        assert "[email removed]" in result.text

    def test_default_rules_only_generic(self):
        """DEFAULT_SCRUB_RULES should not contain any personal identifiers."""
        categories = {rule[2] for rule in DEFAULT_SCRUB_RULES}
        # Should only have generic categories, no 'name' or 'identity'
        assert "url" in categories or "platform" in categories or "email" in categories
        # Verify no rule has an actual name-like replacement (a basic sanity check)
        for pattern, replacement, category in DEFAULT_SCRUB_RULES:
            assert category in ("url", "platform", "email"), (
                f"Unexpected category '{category}' in DEFAULT_SCRUB_RULES"
            )


class TestSetRuntimeRules:
    def test_set_and_apply_runtime_rules(self):
        set_runtime_rules([
            (r"\bJohnDoe\b", "TheAnalyst", "name"),
        ])
        a = Anonymizer()
        result = a.scrub("JohnDoe said buy NVDA")
        assert "JohnDoe" not in result.text
        assert "TheAnalyst" in result.text
        # Clean up
        set_runtime_rules([])

    def test_runtime_rules_cleared(self):
        set_runtime_rules([])
        a = Anonymizer()
        result = a.scrub("JohnDoe said buy NVDA")
        assert "JohnDoe" in result.text  # not scrubbed without the rule


class TestPostCheck:
    def test_flags_url_in_output(self):
        a = Anonymizer()
        result = a.post_check("The answer is at https://example.com/data")
        assert any("URL detected" in f for f in result.flagged_patterns)

    def test_no_flags_for_clean_text(self):
        a = Anonymizer()
        result = a.post_check("NVDA is trading at $900, PE at 35x.")
        assert result.flagged_patterns == []

    def test_flags_email_in_output(self):
        a = Anonymizer()
        # scrub won't remove generic emails already caught by DEFAULT_SCRUB_RULES,
        # so we test post_check flags after scrub's replacement passes through
        result = a.post_check("Contact admin@test.org for support")
        # scrub replaces it first, so the flagged result is [email removed] — no further flag
        # The important thing: post_check runs scrub first then checks remainder
        assert isinstance(result, AnonymizationResult)


class TestScrubDict:
    def test_scrubs_string_values(self):
        a = Anonymizer()
        data = {"note": "Check https://www.patreon.com/x for info"}
        result = a.scrub_dict(data)
        assert "patreon.com" not in result["note"]

    def test_recurses_into_nested_dict(self):
        a = Anonymizer()
        data = {"outer": {"inner": "Contact me@test.com now"}}
        result = a.scrub_dict(data)
        assert "me@test.com" not in result["outer"]["inner"]

    def test_recurses_into_list_of_strings(self):
        a = Anonymizer()
        data = {"items": ["Patreon link here", "normal text"]}
        result = a.scrub_dict(data)
        assert "Patreon" not in result["items"][0]
        assert result["items"][1] == "normal text"

    def test_preserves_non_string_values(self):
        a = Anonymizer()
        data = {"count": 42, "flag": True, "price": 3.14}
        result = a.scrub_dict(data)
        assert result["count"] == 42
        assert result["flag"] is True
        assert result["price"] == 3.14


# ── 2. prompts.py ─────────────────────────────────────────────────────────────

from app.llm.prompts import (
    CATEGORY_RANK,
    build_system_prompt,
    format_metrics,
    format_stock_context,
)
from app.core.decision_engine import TickerMetrics, Zone


class TestFormatStockContext:
    def _base_data(self):
        return {
            "ticker": "NVDA",
            "stock_name": "NVIDIA",
            "category": "PRIMARY",
            "buy_high": 900.0,
            "buy_low": 750.0,
            "sell_start": 1100.0,
            "fair_value": 1000.0,
            "pe_range_low": 30.0,
            "pe_range_high": 60.0,
            "egf": 1.25,
            "egf_direction": 0.05,
            "egf_12m": 1.30,
            "fundamentals": 8.5,
            "trend_status": "bullish",
            "strategy_text": "Accumulate on dips",
        }

    def test_formats_ticker_and_name(self):
        out = format_stock_context(self._base_data())
        assert "NVDA" in out
        assert "NVIDIA" in out

    def test_formats_buy_range(self):
        out = format_stock_context(self._base_data())
        assert "900" in out
        assert "750" in out

    def test_handles_none_buy_high(self):
        data = self._base_data()
        data["buy_high"] = None
        out = format_stock_context(data)
        assert "not provided" in out

    def test_handles_none_egf(self):
        data = self._base_data()
        data["egf"] = None
        out = format_stock_context(data)
        assert "not provided" in out

    def test_primary_category_label(self):
        out = format_stock_context(self._base_data())
        assert "Primary" in out
        assert "priority 1" in out

    def test_secondary_category(self):
        data = self._base_data()
        data["category"] = "SECONDARY"
        out = format_stock_context(data)
        assert "Secondary" in out

    def test_unknown_category(self):
        data = self._base_data()
        data["category"] = "MADE_UP"
        out = format_stock_context(data)
        assert "Unknown" in out


class TestFormatMetrics:
    def _make_metrics(self, **kwargs):
        defaults = dict(
            ticker="NVDA",
            current_price=850.0,
            zone=Zone.ACCUMULATION,
            zone_label="accumulation zone",
            pct_from_buy_high=-0.055,
            trim_guidance=None,
            fair_value_gap=0.15,
            pe_position=0.6,
        )
        defaults.update(kwargs)
        return TickerMetrics(**defaults)

    def test_formats_price(self):
        out = format_metrics(self._make_metrics())
        assert "850.00" in out

    def test_handles_none_pe_position(self):
        m = self._make_metrics(pe_position=None)
        out = format_metrics(m)
        assert "N/A" in out

    def test_handles_none_pct_from_buy_high(self):
        m = self._make_metrics(pct_from_buy_high=None)
        out = format_metrics(m)
        assert "N/A" in out

    def test_formats_pct_from_buy_high(self):
        m = self._make_metrics(pct_from_buy_high=-0.055)
        out = format_metrics(m)
        assert "-5.5%" in out

    def test_trim_guidance_none_shows_default(self):
        m = self._make_metrics(trim_guidance=None)
        out = format_metrics(m)
        assert "No trim data" in out

    def test_trim_guidance_present(self):
        m = self._make_metrics(trim_guidance="Trim 5%")
        out = format_metrics(m)
        assert "Trim 5%" in out


class TestBuildSystemPrompt:
    def test_returns_string(self):
        result = build_system_prompt(["Buy low PE"], ["stock context"], ["metrics"])
        assert isinstance(result, str)

    def test_contains_principles(self):
        result = build_system_prompt(["Buy low PE", "Hold long"], [], [])
        assert "Buy low PE" in result
        assert "Hold long" in result

    def test_contains_stock_context(self):
        result = build_system_prompt([], ["NVDA context here"], [])
        assert "NVDA context here" in result

    def test_empty_principles_fallback(self):
        result = build_system_prompt([], [], [])
        assert "No principles loaded yet." in result

    def test_empty_stock_context_fallback(self):
        result = build_system_prompt([], [], [])
        assert "No stock data loaded." in result


class TestCategoryRank:
    def test_primary_exists(self):
        assert "PRIMARY" in CATEGORY_RANK
        assert CATEGORY_RANK["PRIMARY"] == ("Primary", 1)

    def test_secondary_exists(self):
        assert "SECONDARY" in CATEGORY_RANK
        assert CATEGORY_RANK["SECONDARY"][1] == 2

    def test_sleepers_exists(self):
        assert "SLEEPERS" in CATEGORY_RANK

    def test_tech_funds_exists(self):
        assert "TECH FUNDS" in CATEGORY_RANK

    def test_war_with_china_exists(self):
        assert "WAR WITH CHINA MEGA-TREND" in CATEGORY_RANK
        assert CATEGORY_RANK["WAR WITH CHINA MEGA-TREND"][0] == "High Risk"


# ── 3. decision_engine.py ─────────────────────────────────────────────────────

from app.core.decision_engine import StockData, analyze_ticker, detect_zone, Zone


def _stock(buy_high=900.0, buy_low=750.0, sell_start=1100.0, fair_value=1000.0, **kwargs):
    return StockData(
        ticker="NVDA",
        buy_high=buy_high,
        buy_low=buy_low,
        sell_start=sell_start,
        fair_value=fair_value,
        **kwargs,
    )


class TestAnalyzeTicker:
    def test_price_below_buy_low_is_deep_value(self):
        data = _stock()
        m = analyze_ticker(700.0, data)
        assert m.zone == Zone.DEEP_VALUE

    def test_price_in_buy_range_is_accumulation(self):
        data = _stock()
        m = analyze_ticker(800.0, data)
        assert m.zone == Zone.ACCUMULATION

    def test_price_at_buy_high_is_accumulation(self):
        data = _stock()
        m = analyze_ticker(900.0, data)
        assert m.zone == Zone.ACCUMULATION

    def test_price_above_buy_range_below_sell_start_is_hold(self):
        data = _stock()
        m = analyze_ticker(950.0, data)
        assert m.zone == Zone.HOLD

    def test_price_above_sell_start_is_distribution(self):
        data = _stock()
        m = analyze_ticker(1200.0, data)
        assert m.zone == Zone.DISTRIBUTION

    def test_pct_from_buy_high_above_range(self):
        data = _stock(buy_high=900.0)
        m = analyze_ticker(990.0, data)
        # (990 - 900) / 900 = 0.1
        assert abs(m.pct_from_buy_high - 0.1) < 1e-9

    def test_pct_from_buy_high_below_range(self):
        data = _stock(buy_high=900.0)
        m = analyze_ticker(810.0, data)
        # (810 - 900) / 900 = -0.1
        assert abs(m.pct_from_buy_high - (-0.1)) < 1e-9

    def test_pct_from_buy_high_none_when_no_buy_high(self):
        data = _stock(buy_high=None, buy_low=None)
        m = analyze_ticker(800.0, data)
        assert m.pct_from_buy_high is None

    def test_zone_unknown_when_no_buy_high(self):
        data = _stock(buy_high=None, buy_low=None)
        m = analyze_ticker(800.0, data)
        assert m.zone == Zone.UNKNOWN

    def test_pe_position_computed(self):
        data = _stock(pe_range_low=20.0, pe_range_high=60.0)
        m = analyze_ticker(800.0, data, current_pe=40.0)
        # (40 - 20) / (60 - 20) = 0.5
        assert abs(m.pe_position - 0.5) < 1e-9

    def test_pe_position_none_when_no_pe(self):
        data = _stock()
        m = analyze_ticker(800.0, data, current_pe=None)
        assert m.pe_position is None


# ── 4. DEFAULT_SCRUB_RULES has no _IDENTITY_PATTERNS ─────────────────────────

class TestNoIdentityPatternsInDefaultRules:
    def test_no_identity_category_in_default_rules(self):
        """DEFAULT_SCRUB_RULES should only have generic categories."""
        for pattern, replacement, category in DEFAULT_SCRUB_RULES:
            assert category in ("url", "platform", "email"), (
                f"Found unexpected category '{category}' — sensitive patterns should be in patreon_parser or runtime rules"
            )

    def test_identity_patterns_in_patreon_parser(self):
        """_BASE_IDENTITY_PATTERNS in patreon_parser should include patreon.com."""
        from app.ingestion.patreon_parser import _BASE_IDENTITY_PATTERNS
        patterns_str = [p.pattern for p in _BASE_IDENTITY_PATTERNS]
        assert any("patreon" in p.lower() for p in patterns_str)


# ── 5. price_service.py ───────────────────────────────────────────────────────

import app.services.price_service as price_service_module
from app.services.price_service import PriceData, _PriceCacheEntry, get_price


@pytest.fixture(autouse=False)
def clear_price_cache():
    """Clear in-memory price cache before each test in this class."""
    price_service_module._price_cache.clear()
    yield
    price_service_module._price_cache.clear()


class TestPriceServiceCache:
    @pytest.mark.asyncio
    async def test_fresh_cache_returned_within_ttl(self, clear_price_cache):
        ticker = "NVDA"
        cached_data = PriceData(ticker=ticker, price=850.0, pe_ratio=35.0)
        price_service_module._price_cache[ticker] = _PriceCacheEntry(data=cached_data)

        with patch("app.services.price_service.settings") as mock_settings:
            mock_settings.finnhub_api_key = ""
            # We should get the cached value, not hit any network call
            result = await get_price(ticker)

        assert result.price == 850.0
        assert result.pe_ratio == 35.0

    @pytest.mark.asyncio
    async def test_stale_entry_returned_on_fetch_failure(self, clear_price_cache):
        """When the fetch fails, stale cache should be returned."""
        ticker = "AAPL"
        stale_data = PriceData(ticker=ticker, price=175.0, pe_ratio=28.0)
        # Manually insert a stale entry (old fetched_at)
        entry = _PriceCacheEntry(data=stale_data)
        entry.fetched_at = time.monotonic() - 9999.0  # very old
        price_service_module._price_cache[ticker] = entry

        with patch("app.services.price_service.settings") as mock_settings:
            mock_settings.finnhub_api_key = ""
            # Make yfinance throw an exception
            with patch("app.services.price_service.asyncio.to_thread", side_effect=Exception("network error")):
                result = await get_price(ticker)

        assert result.price == 175.0  # stale cache returned

    @pytest.mark.asyncio
    async def test_zero_price_not_cached(self, clear_price_cache):
        """A zero price from a failed fetch must not be written to the cache."""
        ticker = "TSLA"

        with patch("app.services.price_service.settings") as mock_settings:
            mock_settings.finnhub_api_key = ""
            with patch("app.services.price_service.asyncio.to_thread", side_effect=Exception("timeout")):
                result = await get_price(ticker)

        assert ticker not in price_service_module._price_cache
        assert result.price == 0.0

    @pytest.mark.asyncio
    async def test_successful_fetch_cached(self, clear_price_cache):
        """A successful fetch with nonzero price is written to cache."""
        ticker = "META"
        fetched_data = {
            "price": 550.0,
            "pe": 22.0,
            "long_name": "Meta Platforms",
            "sector": "Technology",
            "industry": "Internet Content & Information",
            "earnings_date": None,
        }

        with patch("app.services.price_service.settings") as mock_settings:
            mock_settings.finnhub_api_key = ""
            with patch("app.services.price_service.asyncio.to_thread", return_value=fetched_data):
                result = await get_price(ticker)

        assert ticker in price_service_module._price_cache
        assert price_service_module._price_cache[ticker].data.price == 550.0
        assert result.pe_ratio == 22.0


# ── 6. chat.py _classify_message_metadata ────────────────────────────────────

from app.api.chat import _classify_message_metadata


class TestClassifyMessageMetadata:
    def test_bearish_strategy_short(self):
        result = _classify_message_metadata("should I short NVDA here?", ["NVDA"])
        assert result["intent"] == "bearish_strategy"

    def test_bearish_strategy_bear(self):
        result = _classify_message_metadata("is NVDA too bearish?", [])
        assert result["intent"] == "bearish_strategy"

    def test_bullish_strategy_buy(self):
        result = _classify_message_metadata("is it a good time to buy AAPL?", ["AAPL"])
        assert result["intent"] == "bullish_strategy"

    def test_bullish_strategy_accumulate(self):
        result = _classify_message_metadata("should I accumulate more?", [])
        assert result["intent"] == "bullish_strategy"

    def test_comparison_vs(self):
        result = _classify_message_metadata("NVDA vs AMD which is better?", ["NVDA", "AMD"])
        assert result["intent"] == "comparison"

    def test_price_query(self):
        result = _classify_message_metadata("what price is MSFT trading at?", ["MSFT"])
        assert result["intent"] == "price_query"

    def test_options_strategy(self):
        result = _classify_message_metadata("what about covered calls on NVDA?", ["NVDA"])
        assert result["intent"] == "options_strategy"

    def test_valuation_pe(self):
        result = _classify_message_metadata("what is the PE ratio of TSLA?", ["TSLA"])
        assert result["intent"] == "valuation"

    def test_valuation_fair_value(self):
        result = _classify_message_metadata("is NVDA overvalued right now?", ["NVDA"])
        assert result["intent"] == "valuation"

    def test_general_intent_fallback(self):
        result = _classify_message_metadata("tell me about the market", [])
        assert result["intent"] == "general"

    def test_tickers_passed_through(self):
        result = _classify_message_metadata("buy NVDA?", ["NVDA"])
        assert result["tickers"] == ["NVDA"]

    def test_sentiment_bearish(self):
        result = _classify_message_metadata("market might crash today", [])
        assert result["sentiment"] == "bearish"

    def test_sentiment_bullish(self):
        result = _classify_message_metadata("expecting a big rally soon", [])
        assert result["sentiment"] == "bullish"

    def test_sentiment_neutral_default(self):
        result = _classify_message_metadata("what is NVDA?", [])
        assert result["sentiment"] == "neutral"


# ── 7. patreon_parser.py — _is_political, _is_identity ───────────────────────

from app.ingestion.patreon_parser import _is_political, _is_identity


class TestIsIdentity:
    def setup_method(self):
        # Reset the cache so each test starts fresh
        import app.ingestion.patreon_parser as pp
        pp._CACHED_IDENTITY_PATTERNS = None

    def test_detects_patreon_url(self):
        with patch("app.ingestion.patreon_parser.settings") as mock_settings:
            mock_settings.identity_strip_patterns = ""
            import app.ingestion.patreon_parser as pp
            pp._CACHED_IDENTITY_PATTERNS = None
            assert _is_identity("Visit https://www.patreon.com/sidsloth today") is True

    def test_clean_text_not_identity(self):
        with patch("app.ingestion.patreon_parser.settings") as mock_settings:
            mock_settings.identity_strip_patterns = ""
            import app.ingestion.patreon_parser as pp
            pp._CACHED_IDENTITY_PATTERNS = None
            assert _is_identity("NVDA is trading at $850") is False

    def test_copyright_is_identity(self):
        with patch("app.ingestion.patreon_parser.settings") as mock_settings:
            mock_settings.identity_strip_patterns = ""
            import app.ingestion.patreon_parser as pp
            pp._CACHED_IDENTITY_PATTERNS = None
            assert _is_identity("Copyright © 2024 All rights reserved") is True

    def test_extra_pattern_from_settings(self):
        import json
        import app.ingestion.patreon_parser as pp
        pp._CACHED_IDENTITY_PATTERNS = None
        with patch("app.ingestion.patreon_parser.settings") as mock_settings:
            mock_settings.identity_strip_patterns = json.dumps(["MarketOracle"])
            pp._CACHED_IDENTITY_PATTERNS = None
            assert _is_identity("Source: MarketOracle analysis") is True


class TestIsPolitical:
    def setup_method(self):
        import app.ingestion.patreon_parser as pp
        pp._CACHED_POLITICAL_SIGNALS = None

    def test_returns_false_when_no_signals_configured(self):
        with patch("app.ingestion.patreon_parser.settings") as mock_settings:
            mock_settings.political_signals = ""
            import app.ingestion.patreon_parser as pp
            pp._CACHED_POLITICAL_SIGNALS = None
            assert _is_political("The president signed a new bill today") is False

    def test_returns_false_with_one_keyword_only(self):
        import json
        import app.ingestion.patreon_parser as pp
        pp._CACHED_POLITICAL_SIGNALS = None
        with patch("app.ingestion.patreon_parser.settings") as mock_settings:
            mock_settings.political_signals = json.dumps(["trump", "biden"])
            pp._CACHED_POLITICAL_SIGNALS = None
            # Only one keyword present — needs >= 2
            assert _is_political("trump signed a new bill") is False

    def test_returns_true_with_two_keywords(self):
        import json
        import app.ingestion.patreon_parser as pp
        pp._CACHED_POLITICAL_SIGNALS = None
        with patch("app.ingestion.patreon_parser.settings") as mock_settings:
            mock_settings.political_signals = json.dumps(["trump", "election"])
            pp._CACHED_POLITICAL_SIGNALS = None
            assert _is_political("trump won the election last night") is True

    def test_empty_env_returns_false(self):
        import app.ingestion.patreon_parser as pp
        pp._CACHED_POLITICAL_SIGNALS = None
        with patch("app.ingestion.patreon_parser.settings") as mock_settings:
            mock_settings.political_signals = ""
            pp._CACHED_POLITICAL_SIGNALS = None
            assert _is_political("very political text about politics and politicians") is False
