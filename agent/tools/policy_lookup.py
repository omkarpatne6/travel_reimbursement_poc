"""Tool: semantic policy search over the FAISS-backed travel policy index."""

from agent.vector_store import get_store, retrieve_policy


def lookup_policy(query: str, k: int = 2) -> str:
    store = get_store()
    return retrieve_policy(store, query, k=k)
