from __future__ import annotations

from pathlib import Path

import pytest

from guarded_agent.guardrails.policy_retrieval import (
    PolicyRetriever,
    get_policy_context,
    load_policy_clauses,
)

SAMPLE_POLICY = """\
# Test policy

## Return delivered order

A refund is issued to the original payment method within 5-7 business days after a
return is confirmed.

## Cancel pending order

An order can be cancelled before it ships, with a full refund issued immediately.

## Shipping information

Standard shipping takes 3-5 business days depending on the destination.
"""


@pytest.fixture(scope="module")
def policy_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("policy") / "policy.md"
    path.write_text(SAMPLE_POLICY)
    return path


@pytest.fixture(scope="module")
def retriever(policy_path: Path) -> PolicyRetriever:
    # Loads the real local embedding model once for the whole module -- no
    # live API calls (CLAUDE.md's testing convention), but real, deterministic
    # embeddings, not a mock.
    return PolicyRetriever.from_policy_file(policy_path)


def test_load_policy_clauses_splits_by_header_with_stable_ids(policy_path: Path) -> None:
    docs = load_policy_clauses(policy_path)
    clause_ids = [d.metadata["clause_id"] for d in docs]

    assert "return-delivered-order" in clause_ids
    assert "cancel-pending-order" in clause_ids
    assert "shipping-information" in clause_ids
    assert len(clause_ids) == len(set(clause_ids))


def test_duplicate_headers_get_distinct_clause_ids(tmp_path: Path) -> None:
    text = "# P\n\n## Notes\n\nFirst clause.\n\n## Notes\n\nSecond clause.\n"
    path = tmp_path / "dup.md"
    path.write_text(text)

    docs = load_policy_clauses(path)
    clause_ids = [d.metadata["clause_id"] for d in docs]

    assert len(clause_ids) == len(set(clause_ids))


def test_refund_query_retrieves_refund_clauses(retriever: PolicyRetriever) -> None:
    """PLAN.md commit 15's literal acceptance check."""
    results = retriever.retrieve("process a refund for the customer", k=2)
    result_ids = {r.clause_id for r in results}

    assert result_ids == {"return-delivered-order", "cancel-pending-order"}
    assert "shipping-information" not in result_ids


def test_get_policy_context_returns_clauses_when_confident(retriever: PolicyRetriever) -> None:
    context = get_policy_context(
        retriever,
        "process a refund",
        full_policy_text="FULL POLICY TEXT",
        top_k=2,
        min_confidence=0.1,
    )

    assert context.used_fallback is False
    assert context.full_policy_text is None
    assert len(context.clauses) == 2


def test_get_policy_context_falls_back_on_low_confidence(retriever: PolicyRetriever) -> None:
    context = get_policy_context(
        retriever,
        "process a refund",
        full_policy_text="FULL POLICY TEXT",
        top_k=2,
        min_confidence=0.999,
    )

    assert context.used_fallback is True
    assert context.fallback_reason == "low confidence"
    assert context.full_policy_text == "FULL POLICY TEXT"


class _FailingRetriever:
    """Stand-in for a retrieval backend that's down, e.g. index missing."""

    def retrieve(self, query: str, k: int) -> list[object]:
        raise RuntimeError("index unavailable")


def test_get_policy_context_falls_back_on_retrieval_error_without_raising() -> None:
    context = get_policy_context(
        _FailingRetriever(),  # type: ignore[arg-type]
        "process a refund",
        full_policy_text="FULL POLICY TEXT",
        top_k=2,
        min_confidence=0.1,
    )

    assert context.used_fallback is True
    assert "retrieval error" in (context.fallback_reason or "")
    assert context.full_policy_text == "FULL POLICY TEXT"
