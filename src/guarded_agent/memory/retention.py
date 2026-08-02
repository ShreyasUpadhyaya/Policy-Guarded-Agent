from __future__ import annotations

import re

_CARD_PATTERN = re.compile(r"(?<!\d)(?:\d[ -]?){13,19}(?!\d)")
_EMAIL_PATTERN = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_PHONE_PATTERN = re.compile(r"(?<!\d)(\+?\d{1,3}[-.\s])?\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}(?!\d)")


def redact_pii(text: str | None) -> str | None:
    """Scrub card-like digit sequences, emails, and phone numbers from free
    text before it can leave this module.

    Card numbers are matched first: a card-like run of 13-19 digits (retail
    payment fields) would otherwise get partially consumed by the shorter
    phone pattern. Pure regex, zero I/O -- CLAUDE.md's testing convention
    lists memory retention filters alongside guardrail/escalation logic as
    needing to be deterministic and testable without mocks.

    A heuristic, not a guarantee: it will not catch PII that doesn't match
    these shapes (e.g. a plain name). docs/MEMORY.md documents this as the
    stated limitation of what the case store is allowed to retain.
    """
    if text is None:
        return None
    redacted = _CARD_PATTERN.sub("[REDACTED_CARD]", text)
    redacted = _EMAIL_PATTERN.sub("[REDACTED_EMAIL]", redacted)
    redacted = _PHONE_PATTERN.sub("[REDACTED_PHONE]", redacted)
    return redacted
