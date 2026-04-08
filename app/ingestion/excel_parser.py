"""
Excel parser for Sid Sloth's AI Tech Stocks Portfolio spreadsheet.

Known column layout (main sheet "AI Tech Stocks Port - Current"):
    B  → stock name          H  → ticker
    I  → buy_high (From)     J  → buy_low (To)
    L  → sell_start (1st trim)
    Z  → pe_current          AD → pe_range_high     AE → pe_range_low
    AI → fair_value           V → egf
    W  → egf_direction        X → egf_12m
    Y  → fundamentals        AH → trend_status
    T  → prob_new_ath        AF → strategy_text
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from openpyxl import load_workbook
from openpyxl.worksheet.worksheet import Worksheet

logger = logging.getLogger(__name__)

# Row layout constants
HEADER_ROW = 2  # Column headers
DATA_START_ROW = 6  # First stock row
# Section labels that mark category boundaries (not stock data)
SECTION_LABELS = {"PRIMARY", "SECONDARY", "SLEEPERS", "TECH FUNDS", "WAR WITH CHINA MEGA-TREND"}
# Rows with summary text like "TARGET 25%", "Av Exposure", "S&P 500"
SKIP_PREFIXES = ("TARGET", "Av Exposure", "S&P")


@dataclass
class StockRow:
    """One ticker's data extracted from the main sheet."""

    stock_name: str
    ticker: str
    buy_high: float | None = None
    buy_low: float | None = None
    sell_start: float | None = None
    pe_range_high: float | None = None
    pe_range_low: float | None = None
    fair_value: float | None = None
    egf: float | None = None
    egf_direction: float | None = None
    egf_12m: float | None = None
    fundamentals: float | None = None
    trend_status: str | None = None
    prob_new_ath: float | None = None
    strategy_text: str | None = None
    category: str = "PRIMARY"  # PRIMARY / SECONDARY / SLEEPERS / TECH FUNDS / WAR WITH CHINA


@dataclass
class SheetSnapshot:
    """All tickers from one sheet (current or dated)."""

    sheet_name: str
    stocks: list[StockRow] = field(default_factory=list)


@dataclass
class PrincipleEntry:
    """One section from the Investing Guide or Real Secret sheet."""

    section_number: str
    title: str
    content: str
    category: str  # valuation / accumulation / distribution / risk / sentiment


def _safe_float(value) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def _safe_str(value) -> str | None:
    if value is None:
        return None
    s = str(value).strip()
    return s if s else None


def _classify_principle(title: str) -> str:
    """Map Investing Guide section titles to principle categories."""
    title_lower = title.lower()
    if any(kw in title_lower for kw in ["valuation", "p/e", "pe", "buy and sell", "basis of"]):
        return "valuation"
    if any(kw in title_lower for kw in ["accumulate", "started", "dollar cost", "best time to buy"]):
        return "accumulation"
    if any(kw in title_lower for kw in ["trim", "distribute", "sell"]):
        return "distribution"
    if any(kw in title_lower for kw in ["leverage", "stop loss", "short", "drawdown", "never sell at"]):
        return "risk"
    if any(kw in title_lower for kw in ["emotion", "fear", "greed", "sentiment", "news", "forget"]):
        return "sentiment"
    return "general"


def parse_stock_sheet(ws: Worksheet, sheet_name: str) -> SheetSnapshot:
    """Parse a single sheet (current or dated) into a SheetSnapshot."""
    snapshot = SheetSnapshot(sheet_name=sheet_name)
    current_category = "PRIMARY"

    for row in ws.iter_rows(min_row=DATA_START_ROW, max_row=ws.max_row):
        # Build a dict keyed by column letter
        cells = {}
        for cell in row:
            if cell.value is not None:
                cells[cell.column_letter] = cell.value

        # Check for section label in column A
        a_val = _safe_str(cells.get("A"))
        if a_val and a_val in SECTION_LABELS:
            current_category = a_val
            continue

        # Get stock name and ticker — both required
        stock_name = _safe_str(cells.get("B"))
        ticker = _safe_str(cells.get("H"))
        if not stock_name or not ticker:
            continue

        # Skip summary rows
        if any(stock_name.startswith(p) for p in SKIP_PREFIXES):
            continue

        # Skip junk rows: ticker must contain at least one letter (filters out crypto/numeric rows)
        if not any(c.isalpha() for c in ticker):
            continue

        stock = StockRow(
            stock_name=stock_name,
            ticker=ticker,
            buy_high=_safe_float(cells.get("I")),
            buy_low=_safe_float(cells.get("J")),
            sell_start=_safe_float(cells.get("L")),
            pe_range_high=_safe_float(cells.get("AD")),
            pe_range_low=_safe_float(cells.get("AE")),
            fair_value=_safe_float(cells.get("AI")),
            egf=_safe_float(cells.get("V")),
            egf_direction=_safe_float(cells.get("W")),
            egf_12m=_safe_float(cells.get("X")),
            fundamentals=_safe_float(cells.get("Y")),
            trend_status=_safe_str(cells.get("AH")),
            prob_new_ath=_safe_float(cells.get("T")),
            strategy_text=_safe_str(cells.get("AF")),
            category=current_category,
        )
        snapshot.stocks.append(stock)

    logger.info("Parsed %d stocks from sheet '%s'", len(snapshot.stocks), sheet_name)
    return snapshot


def parse_investing_guide(ws: Worksheet) -> list[PrincipleEntry]:
    """Parse the 'Investing Guide' sheet into principle entries."""
    principles: list[PrincipleEntry] = []
    current_number = ""
    current_title = ""
    current_content_parts: list[str] = []

    def _flush():
        if current_title and current_content_parts:
            principles.append(PrincipleEntry(
                section_number=current_number,
                title=current_title,
                content="\n".join(current_content_parts).strip(),
                category=_classify_principle(current_title),
            ))

    for row in ws.iter_rows(min_row=1, max_row=ws.max_row):
        cells = {}
        for cell in row:
            if cell.value is not None:
                cells[cell.column_letter] = cell.value

        a_val = cells.get("A")
        b_val = _safe_str(cells.get("B"))
        c_val = _safe_str(cells.get("C"))

        # Detect section header: column A has a number, column B has a title
        # Detailed content rows have number + text merged or text in later columns
        if a_val is not None and b_val:
            try:
                num = str(float(a_val))
            except (ValueError, TypeError):
                num = ""

            if num:
                # Check if this row is a header (short title) vs content (long text with " - ")
                full_text = b_val
                if " - " in full_text:
                    # This is a content row like "1.0 | UNDERSTAND WHAT... - detailed text..."
                    _flush()
                    parts = full_text.split(" - ", 1)
                    current_number = num
                    current_title = parts[0].strip()
                    current_content_parts = [parts[1].strip()] if len(parts) > 1 else []
                else:
                    # Just a title row — content follows in subsequent rows
                    _flush()
                    current_number = num
                    current_title = full_text
                    current_content_parts = []
                continue

        # Continuation text in various columns
        for col in ["B", "C", "D", "X"]:
            text = _safe_str(cells.get(col))
            if text and current_title:
                current_content_parts.append(text)

    _flush()
    logger.info("Extracted %d principles from Investing Guide", len(principles))
    return principles


def parse_real_secret(ws: Worksheet) -> list[PrincipleEntry]:
    """Parse the 'Real Secret' sheet into principle entries."""
    content_parts: list[str] = []

    for row in ws.iter_rows(min_row=1, max_row=ws.max_row):
        for cell in row:
            if cell.value is not None:
                text = str(cell.value).strip()
                if text:
                    content_parts.append(text)

    full_text = "\n".join(content_parts)
    if not full_text:
        return []

    return [PrincipleEntry(
        section_number="0",
        title="The Real Secret for Successful Trading",
        content=full_text,
        category="sentiment",
    )]


def diff_snapshots(old: SheetSnapshot, new: SheetSnapshot) -> list[dict]:
    """Compare two dated snapshots and return a list of changes."""
    old_map = {s.ticker: s for s in old.stocks}
    new_map = {s.ticker: s for s in new.stocks}
    changes = []

    all_tickers = set(old_map.keys()) | set(new_map.keys())
    tracked_fields = [
        "buy_high", "buy_low", "sell_start", "pe_range_high", "pe_range_low",
        "fair_value", "egf", "egf_12m", "fundamentals", "trend_status",
        "prob_new_ath", "strategy_text",
    ]

    for ticker in sorted(all_tickers):
        if ticker not in old_map:
            changes.append({"ticker": ticker, "change": "added", "details": {}})
            continue
        if ticker not in new_map:
            changes.append({"ticker": ticker, "change": "removed", "details": {}})
            continue

        old_stock = old_map[ticker]
        new_stock = new_map[ticker]
        field_changes = {}
        for f in tracked_fields:
            old_val = getattr(old_stock, f)
            new_val = getattr(new_stock, f)
            if old_val != new_val:
                field_changes[f] = {"old": old_val, "new": new_val}

        if field_changes:
            changes.append({"ticker": ticker, "change": "updated", "details": field_changes})

    return changes


def parse_workbook(file_path: str | Path) -> dict:
    """
    Parse the full Sid Sloth workbook.

    Returns:
        {
            "current": SheetSnapshot,
            "dated_snapshots": [SheetSnapshot, ...],
            "principles": [PrincipleEntry, ...],
            "diffs": [{"from": str, "to": str, "changes": [...]}, ...],
        }
    """
    wb = load_workbook(file_path, read_only=True, data_only=True)

    result = {
        "current": None,
        "dated_snapshots": [],
        "principles": [],
        "diffs": [],
    }

    # Parse main current sheet
    current_sheet_name = "AI Tech Stocks Port - Current"
    if current_sheet_name in wb.sheetnames:
        result["current"] = parse_stock_sheet(wb[current_sheet_name], current_sheet_name)

    # Parse Investing Guide
    if "Investing Guide" in wb.sheetnames:
        result["principles"].extend(parse_investing_guide(wb["Investing Guide"]))

    # Parse Real Secret
    if "Real Secret" in wb.sheetnames:
        result["principles"].extend(parse_real_secret(wb["Real Secret"]))

    # Parse dated sheets (sorted chronologically by sheet order)
    dated_sheets = []
    skip_names = {current_sheet_name, "Port Charts", "Investing Guide", "Real Secret",
                  "CI18", "Cryptos", "EGFs", "Big Picture", "New Buyers"}
    for name in wb.sheetnames:
        if name not in skip_names:
            try:
                snapshot = parse_stock_sheet(wb[name], name)
                if snapshot.stocks:
                    dated_sheets.append(snapshot)
            except Exception as e:
                logger.warning("Failed to parse dated sheet '%s': %s", name, e)

    result["dated_snapshots"] = dated_sheets

    # Generate diffs between consecutive dated snapshots
    for i in range(1, len(dated_sheets)):
        old_snap = dated_sheets[i - 1]
        new_snap = dated_sheets[i]
        changes = diff_snapshots(old_snap, new_snap)
        if changes:
            result["diffs"].append({
                "from": old_snap.sheet_name,
                "to": new_snap.sheet_name,
                "changes": changes,
            })

    wb.close()
    logger.info(
        "Workbook parsed: %d current stocks, %d dated snapshots, %d principles, %d diffs",
        len(result["current"].stocks) if result["current"] else 0,
        len(result["dated_snapshots"]),
        len(result["principles"]),
        len(result["diffs"]),
    )
    return result
