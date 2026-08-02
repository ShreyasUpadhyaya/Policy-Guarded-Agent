from __future__ import annotations

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from guarded_agent.guardrails.policy_retrieval import DEFAULT_EMBEDDING_MODEL
from guarded_agent.memory.session import CaseRecord


def _case_to_document(case: CaseRecord) -> Document:
    text = " ".join(part for part in (case.issue_summary, case.resolution_summary) if part)
    return Document(page_content=text or case.session_id, metadata={"case": case.model_dump_json()})


def _document_to_case(document: Document) -> CaseRecord:
    return CaseRecord.model_validate_json(document.metadata["case"])


class CaseStore:
    """Retrieval-backed store of resolved-case summaries.

    Reuses the exact LangChain vectorstore/embeddings pattern
    PolicyRetriever established in commit 15 (PLAN.md commit 19: "not a
    separate bespoke implementation") rather than a bespoke index.

    Starts empty and builds its FAISS index lazily on the first add_case
    call -- unlike PolicyRetriever, which is always seeded from a
    non-empty policy document, a fresh store has no prior cases yet, and
    FAISS.from_documents requires at least one document to exist.
    """

    def __init__(self, embedding_model: str = DEFAULT_EMBEDDING_MODEL) -> None:
        self._embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        self._store: FAISS | None = None

    def add_case(self, case: CaseRecord) -> None:
        document = _case_to_document(case)
        if self._store is None:
            self._store = FAISS.from_documents([document], self._embeddings)
        else:
            self._store.add_documents([document])

    def retrieve(self, query: str, k: int) -> list[CaseRecord]:
        if self._store is None:
            return []
        results = self._store.similarity_search(query, k=k)
        return [_document_to_case(doc) for doc in results]
