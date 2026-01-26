"""
Session Memory Store for Pravah v2

Provides in-memory storage for fetched documents within a conversation session.
Uses BM25 for fast keyword search (no embeddings needed for session-scoped search).

Features:
- Thread-safe document storage keyed by thread_id
- BM25 keyword search across fetched content
- URL deduplication
- Automatic cleanup of old sessions
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from threading import Lock
from typing import Optional
import re


@dataclass
class Document:
    """A fetched document."""

    url: str
    content: str
    title: str = ""
    fetched_at: datetime = field(default_factory=datetime.now)


@dataclass
class SessionMemory:
    """Memory for a single conversation session."""

    thread_id: str
    documents: dict[str, Document] = field(default_factory=dict)  # url -> Document
    created_at: datetime = field(default_factory=datetime.now)
    last_accessed: datetime = field(default_factory=datetime.now)

    def add_document(self, url: str, content: str, title: str = "") -> None:
        """Add or update a document in memory."""
        self.documents[url] = Document(url=url, content=content, title=title)
        self.last_accessed = datetime.now()

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """Search documents using simple keyword matching.

        Uses a basic TF-IDF-like scoring for speed (no external dependencies).
        """
        if not self.documents:
            return []

        self.last_accessed = datetime.now()

        # Tokenize query
        query_terms = set(re.findall(r"\w+", query.lower()))
        if not query_terms:
            return []

        # Score each document
        scored_docs = []
        for url, doc in self.documents.items():
            content_lower = doc.content.lower()

            # Count term occurrences
            score = 0
            matched_terms = []
            for term in query_terms:
                count = content_lower.count(term)
                if count > 0:
                    score += count
                    matched_terms.append(term)

            if score > 0:
                # Extract a relevant snippet
                snippet = self._extract_snippet(doc.content, list(query_terms))
                scored_docs.append(
                    {
                        "url": url,
                        "title": doc.title or url,
                        "score": score,
                        "matched_terms": matched_terms,
                        "snippet": snippet,
                        "content_length": len(doc.content),
                    }
                )

        # Sort by score descending
        scored_docs.sort(key=lambda x: x["score"], reverse=True)
        return scored_docs[:top_k]

    def _extract_snippet(
        self, content: str, terms: list[str], context_chars: int = 200
    ) -> str:
        """Extract a snippet around the first matching term."""
        content_lower = content.lower()

        # Find the first occurrence of any term
        best_pos = len(content)
        for term in terms:
            pos = content_lower.find(term)
            if pos != -1 and pos < best_pos:
                best_pos = pos

        if best_pos == len(content):
            # No match found, return start of content
            return content[: context_chars * 2] + "..."

        # Extract context around the match
        start = max(0, best_pos - context_chars)
        end = min(len(content), best_pos + context_chars)

        snippet = content[start:end]
        if start > 0:
            snippet = "..." + snippet
        if end < len(content):
            snippet = snippet + "..."

        return snippet

    def get_all_urls(self) -> list[str]:
        """Get all fetched URLs."""
        return list(self.documents.keys())

    def get_document(self, url: str) -> Optional[Document]:
        """Get a specific document by URL."""
        return self.documents.get(url)


class MemoryStore:
    """Global store for all session memories.

    Thread-safe singleton that manages memories across all conversations.
    """

    _instance: Optional["MemoryStore"] = None
    _lock = Lock()

    def __new__(cls) -> "MemoryStore":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._memories: dict[str, SessionMemory] = {}
                    cls._instance._memories_lock = Lock()
        return cls._instance

    def get_memory(self, thread_id: str) -> SessionMemory:
        """Get or create a memory for a thread."""
        with self._memories_lock:
            if thread_id not in self._memories:
                self._memories[thread_id] = SessionMemory(thread_id=thread_id)
            return self._memories[thread_id]

    def add_document(
        self, thread_id: str, url: str, content: str, title: str = ""
    ) -> None:
        """Add a document to a thread's memory."""
        memory = self.get_memory(thread_id)
        memory.add_document(url, content, title)

    def search(self, thread_id: str, query: str, top_k: int = 5) -> list[dict]:
        """Search documents in a thread's memory."""
        memory = self.get_memory(thread_id)
        return memory.search(query, top_k)

    def get_all_urls(self, thread_id: str) -> list[str]:
        """Get all URLs fetched in a thread."""
        memory = self.get_memory(thread_id)
        return memory.get_all_urls()

    def cleanup_old_sessions(self, max_age_hours: int = 24) -> int:
        """Remove sessions older than max_age_hours. Returns count removed."""
        cutoff = datetime.now() - timedelta(hours=max_age_hours)
        removed = 0

        with self._memories_lock:
            to_remove = [
                tid for tid, mem in self._memories.items() if mem.last_accessed < cutoff
            ]
            for tid in to_remove:
                del self._memories[tid]
                removed += 1

        return removed

    def clear_thread(self, thread_id: str) -> None:
        """Clear memory for a specific thread."""
        with self._memories_lock:
            if thread_id in self._memories:
                del self._memories[thread_id]


# Global singleton accessor
def get_memory_store() -> MemoryStore:
    """Get the global memory store instance."""
    return MemoryStore()


# Thread-local context for passing thread_id to tools
# This is set by the agent before tool execution
_current_thread_id: Optional[str] = None
_thread_id_lock = Lock()


def set_current_thread_id(thread_id: str) -> None:
    """Set the current thread ID for tool context."""
    global _current_thread_id
    with _thread_id_lock:
        _current_thread_id = thread_id


def get_current_thread_id() -> Optional[str]:
    """Get the current thread ID."""
    with _thread_id_lock:
        return _current_thread_id
