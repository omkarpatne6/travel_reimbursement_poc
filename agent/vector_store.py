"""FAISS-backed policy retrieval.

Chunks the travel policy markdown by section, embeds each chunk, and exposes
a small retrieval helper used by the `policy_retrieval` node.

Embeddings prefer OpenAI (`OPENAI_API_KEY`) when available. If no OpenAI key
is configured, a deterministic hashing-based embedding is used instead so the
index still builds and retrieval still works offline / without any provider
key -- only the final `decision` node actually requires a configured LLM.
"""

import hashlib
import os
import re

from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings

EMBEDDING_DIM = 384


class HashingEmbeddings(Embeddings):
    """Deterministic, dependency-free fallback embedding.

    Not semantically meaningful, but stable and good enough for a small,
    section-headed policy document where keyword overlap already does most
    of the retrieval work.
    """

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * EMBEDDING_DIM
        for token in re.findall(r"[a-z0-9]+", text.lower()):
            h = int(hashlib.md5(token.encode()).hexdigest(), 16)
            idx = h % EMBEDDING_DIM
            vec[idx] += 1.0
        norm = sum(v * v for v in vec) ** 0.5
        if norm > 0:
            vec = [v / norm for v in vec]
        return vec

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


def _get_embeddings():
    if os.environ.get("OPENAI_API_KEY"):
        try:
            from langchain_openai import OpenAIEmbeddings

            return OpenAIEmbeddings(model="text-embedding-3-small")
        except Exception:
            pass
    return HashingEmbeddings()


def _split_policy_sections(text: str) -> list[str]:
    """Split the policy markdown into clean, section-level chunks on `##` headers."""
    chunks = re.split(r"\n(?=## )", text.strip())
    return [c.strip() for c in chunks if c.strip()]


def build_vector_store(policy_path: str) -> FAISS:
    with open(policy_path, encoding="utf-8") as f:
        text = f.read()

    sections = _split_policy_sections(text)
    embeddings = _get_embeddings()
    try:
        return FAISS.from_texts(sections, embeddings)
    except Exception as e:
        # The OpenAI key may be present but invalid/expired/rate-limited --
        # don't let a flaky embeddings provider take down policy retrieval.
        if not isinstance(embeddings, HashingEmbeddings):
            print(f"WARNING: embedding provider failed ({e}); falling back to offline hashing embeddings")
            return FAISS.from_texts(sections, HashingEmbeddings())
        raise


def retrieve_policy(store: FAISS, query: str, k: int = 2) -> str:
    docs = store.similarity_search(query, k=k)
    return "\n\n".join(d.page_content for d in docs)


_STORE: FAISS | None = None


def get_store(policy_path: str | None = None) -> FAISS:
    """Return the process-wide policy index, building it on first use."""
    global _STORE
    if _STORE is None:
        _STORE = build_vector_store(policy_path or os.environ.get("POLICY_PATH", "data/policy/travel_policy.md"))
    return _STORE


def reset_store() -> None:
    """Clear the cached index so the next `get_store()` call rebuilds it."""
    global _STORE
    _STORE = None
