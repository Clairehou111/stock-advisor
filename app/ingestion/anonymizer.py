"""
Three-layer anonymization for analyst identity protection.

Layer 1 (this module): Pre-storage scrub — dictionary replacement of known identifiers
Layer 2: System prompt rules — never attribute, deflect identity questions
Layer 3: Output post-check — regex filter for URL/email/name leakage

Sensitive rules (real names, nicknames) are NOT stored in source code.
They are loaded at startup from:
  1. ANON_EXTRA_RULES env var (JSON array of [pattern, replacement, category])
  2. AnonymizationRule table in DB (seeded from env var, managed via admin)
Call set_runtime_rules() during app startup to activate them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class AnonymizationResult:
    text: str
    replacements_made: int = 0
    flagged_patterns: list[str] = field(default_factory=list)


# Generic rules safe to keep in source — no identifying information here.
DEFAULT_SCRUB_RULES: list[tuple[str, str, str]] = [
    # URLs
    (r"https?://(?:www\.)?patreon\.com/[^\s]+", "[link removed]", "url"),
    (r"https?://(?:www\.)?youtube\.com/[^\s]+", "[link removed]", "url"),
    (r"https?://youtu\.be/[^\s]+", "[link removed]", "url"),
    # Platform references
    (r"\bPatreon\b", "the research platform", "platform"),
    (r"\bmy latest videos?\b", "the latest analysis", "platform"),
    (r"\bmy portfolio\b", "the portfolio", "platform"),
    # Email patterns (generic catch-all)
    (r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", "[email removed]", "email"),
]

# Patterns to flag but not replace (potential leakage in LLM output)
FLAG_PATTERNS: list[tuple[str, str]] = [
    (r"https?://[^\s]+", "URL detected in output"),
    (r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", "email detected in output"),
    (r"\bpatron(?:s)?\b", "possible Patreon reference"),
]

# Runtime rules loaded at startup from DB + env var.
# Populated by set_runtime_rules() — never hardcoded here.
_RUNTIME_RULES: list[tuple[str, str, str]] = []


def set_runtime_rules(rules: list[tuple[str, str, str]]) -> None:
    """Load sensitive anonymization rules into memory. Called once at app startup."""
    global _RUNTIME_RULES
    _RUNTIME_RULES = list(rules)


class Anonymizer:
    """Scrubs identifying information from analyst content.

    Rules are merged from DEFAULT_SCRUB_RULES (generic, in-code) and
    _RUNTIME_RULES (sensitive, loaded from DB/env at startup).
    Always instantiate after startup so runtime rules are available.
    """

    def __init__(self, extra_rules: list[tuple[str, str, str]] | None = None):
        all_rules = DEFAULT_SCRUB_RULES + _RUNTIME_RULES + (extra_rules or [])
        self._compiled_rules = [
            (re.compile(pattern, re.IGNORECASE), replacement, category)
            for pattern, replacement, category in all_rules
        ]
        self._flag_patterns = [
            (re.compile(pattern, re.IGNORECASE), description)
            for pattern, description in FLAG_PATTERNS
        ]

    def scrub(self, text: str) -> AnonymizationResult:
        """Layer 1: Replace all known identifying terms in text."""
        result_text = text
        total_replacements = 0

        for compiled_pattern, replacement, _category in self._compiled_rules:
            result_text, count = compiled_pattern.subn(replacement, result_text)
            total_replacements += count

        return AnonymizationResult(
            text=result_text,
            replacements_made=total_replacements,
        )

    def post_check(self, text: str) -> AnonymizationResult:
        """Layer 3: Check LLM output for any leaked identifying information."""
        result = self.scrub(text)
        for compiled_pattern, description in self._flag_patterns:
            matches = compiled_pattern.findall(result.text)
            for match in matches:
                result.flagged_patterns.append(f"{description}: {match}")
        return result

    def scrub_dict(self, data: dict) -> dict:
        """Recursively scrub all string values in a dictionary."""
        result = {}
        for key, value in data.items():
            if isinstance(value, str):
                result[key] = self.scrub(value).text
            elif isinstance(value, dict):
                result[key] = self.scrub_dict(value)
            elif isinstance(value, list):
                result[key] = [
                    self.scrub(item).text if isinstance(item, str)
                    else self.scrub_dict(item) if isinstance(item, dict)
                    else item
                    for item in value
                ]
            else:
                result[key] = value
        return result
