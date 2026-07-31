from __future__ import annotations

import re
import warnings
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import MarkdownHeaderTextSplitter
from pydantic import BaseModel

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

_HEADERS_TO_SPLIT_ON = [("##", "section"), ("###", "subsection")]

_SLUG_PATTERN = re.compile(r"[^a-z0-9]+")


def _slugify(*parts: str) -> str:
    text = " ".join(p for p in parts if p).lower()
    slug = _SLUG_PATTERN.sub("-", text).strip("-")
    return slug


def load_policy_clauses(policy_path: Path) -> list[Document]:
    """Split a domain policy.md into clause-level Documents by markdown header.

    Each Document's metadata gets a stable `clause_id` derived from its
    section/subsection headers (e.g. "return-delivered-order"), used later
    to cite the exact clause a policy verdict is based on (commit 16).
    """
    splitter = MarkdownHeaderTextSplitter(headers_to_split_on=_HEADERS_TO_SPLIT_ON)
    documents = splitter.split_text(policy_path.read_text(encoding="utf-8"))

    seen_ids: dict[str, int] = {}
    for index, document in enumerate(documents):
        section = document.metadata.get("section", "")
        subsection = document.metadata.get("subsection", "")
        clause_id = _slugify(section, subsection) or f"clause-{index}"
        if clause_id in seen_ids:
            seen_ids[clause_id] += 1
            clause_id = f"{clause_id}-{seen_ids[clause_id]}"
        else:
            seen_ids[clause_id] = 0
        document.metadata["clause_id"] = clause_id
    return documents


class RetrievedClause(BaseModel):
    clause_id: str
    text: str
    score: float


class PolicyContext(BaseModel):
    """What a policy_checker node (commit 16) should reason over.

    Either a short list of confidently-retrieved clauses, or -- on low
    confidence or a retrieval error -- the full policy text as a fail-safe
    fallback. Never both partially: a consumer checks `used_fallback` once.
    """

    clauses: list[RetrievedClause]
    used_fallback: bool
    fallback_reason: str | None = None
    full_policy_text: str | None = None


class PolicyRetriever:
    """Retrieval over a domain's policy.md, chunked by clause."""

    def __init__(
        self, documents: list[Document], embedding_model: str = DEFAULT_EMBEDDING_MODEL
    ) -> None:
        embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        self._store = FAISS.from_documents(documents, embeddings)

    @classmethod
    def from_policy_file(
        cls, policy_path: Path, embedding_model: str = DEFAULT_EMBEDDING_MODEL
    ) -> PolicyRetriever:
        return cls(load_policy_clauses(policy_path), embedding_model=embedding_model)

    def retrieve(self, query: str, k: int) -> list[RetrievedClause]:
        # FAISS's default relevance-score normalization assumes a distance/
        # embedding combination that doesn't hold exactly for this model, so
        # it warns that scores aren't bounded to [0, 1]. Verified empirically
        # (guardrails/policy_retrieval.py, PLAN.md commit 15 notes) that
        # ordering and relevant/irrelevant separation are still sound --
        # min_confidence is calibrated against the real observed range, not
        # the theoretical one.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Relevance scores must be between 0 and 1")
            results = self._store.similarity_search_with_relevance_scores(query, k=k)
        return [
            RetrievedClause(clause_id=doc.metadata["clause_id"], text=doc.page_content, score=score)
            for doc, score in results
        ]


def get_policy_context(
    retriever: PolicyRetriever,
    query: str,
    full_policy_text: str,
    top_k: int,
    min_confidence: float,
) -> PolicyContext:
    """Retrieve clauses relevant to `query`, falling back to the full policy
    text on low confidence or a retrieval failure.

    Fail-safe forcing, not a silent pass-through: a retrieval backend error
    (embedding call fails, index missing) is treated the same as a
    low-confidence result, never as "no policy applies."
    """
    try:
        clauses = retriever.retrieve(query, k=top_k)
    except Exception as exc:  # boundary around a third-party retrieval backend
        return PolicyContext(
            clauses=[],
            used_fallback=True,
            fallback_reason=f"retrieval error: {exc}",
            full_policy_text=full_policy_text,
        )

    if not clauses or clauses[0].score < min_confidence:
        reason = "no clauses retrieved" if not clauses else "low confidence"
        return PolicyContext(
            clauses=clauses,
            used_fallback=True,
            fallback_reason=reason,
            full_policy_text=full_policy_text,
        )

    return PolicyContext(clauses=clauses, used_fallback=False)
