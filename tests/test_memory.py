"""Tests for pravah.memory module."""

from pravah.memory import MemoryStore, SessionMemory, Document


class TestSessionMemory:
    def test_add_and_retrieve_document(self):
        mem = SessionMemory(thread_id="t1")
        mem.add_document("https://example.com", "Hello world", "Example")

        doc = mem.get_document("https://example.com")
        assert doc is not None
        assert doc.content == "Hello world"
        assert doc.title == "Example"

    def test_get_missing_document_returns_none(self):
        mem = SessionMemory(thread_id="t1")
        assert mem.get_document("https://missing.com") is None

    def test_search_finds_matching_terms(self):
        mem = SessionMemory(thread_id="t1")
        mem.add_document("https://a.com", "Python is a programming language", "Python")
        mem.add_document("https://b.com", "Java is another language", "Java")

        results = mem.search("Python programming")
        assert len(results) > 0
        assert results[0]["url"] == "https://a.com"

    def test_search_empty_memory_returns_empty(self):
        mem = SessionMemory(thread_id="t1")
        assert mem.search("anything") == []

    def test_search_no_match_returns_empty(self):
        mem = SessionMemory(thread_id="t1")
        mem.add_document("https://a.com", "cats and dogs", "Animals")
        assert mem.search("quantum physics") == []

    def test_get_all_urls(self):
        mem = SessionMemory(thread_id="t1")
        mem.add_document("https://a.com", "content a", "A")
        mem.add_document("https://b.com", "content b", "B")

        urls = mem.get_all_urls()
        assert set(urls) == {"https://a.com", "https://b.com"}

    def test_overwrite_document(self):
        mem = SessionMemory(thread_id="t1")
        mem.add_document("https://a.com", "old content", "A")
        mem.add_document("https://a.com", "new content", "A updated")

        doc = mem.get_document("https://a.com")
        assert doc.content == "new content"
        assert doc.title == "A updated"

    def test_extract_snippet(self):
        mem = SessionMemory(thread_id="t1")
        content = "x" * 500 + " Python " + "y" * 500
        snippet = mem._extract_snippet(content, ["python"], context_chars=50)
        assert "python" in snippet.lower()
        assert len(snippet) < len(content)


class TestMemoryStore:
    def _fresh_store(self):
        """Create a fresh MemoryStore (reset singleton)."""
        MemoryStore._instance = None
        return MemoryStore()

    def test_add_and_search(self):
        store = self._fresh_store()
        store.add_document("t1", "https://a.com", "Python programming", "Python")

        results = store.search("t1", "Python")
        assert len(results) > 0

    def test_get_document(self):
        store = self._fresh_store()
        store.add_document("t1", "https://a.com", "Test content", "Test")

        content = store.get_document("t1", "https://a.com")
        assert content == "Test content"

    def test_get_document_missing(self):
        store = self._fresh_store()
        assert store.get_document("t1", "https://missing.com") is None

    def test_get_all_urls(self):
        store = self._fresh_store()
        store.add_document("t1", "https://a.com", "a", "A")
        store.add_document("t1", "https://b.com", "b", "B")

        urls = store.get_all_urls("t1")
        assert len(urls) == 2

    def test_clear_thread(self):
        store = self._fresh_store()
        store.add_document("t1", "https://a.com", "content", "title")
        store.clear_thread("t1")

        # After clearing, searching should return empty (fresh memory)
        results = store.search("t1", "content")
        assert results == []

    def test_cleanup_old_sessions(self):
        store = self._fresh_store()
        store.add_document("t1", "https://a.com", "content", "title")

        # Should not remove recent sessions
        removed = store.cleanup_old_sessions(max_age_hours=24)
        assert removed == 0

    def test_separate_threads(self):
        store = self._fresh_store()
        store.add_document("t1", "https://a.com", "quantum entanglement physics", "T1")
        store.add_document("t2", "https://b.com", "baking chocolate cake recipe", "T2")

        results_t1 = store.search("t1", "quantum physics")
        results_t2 = store.search("t2", "chocolate cake")
        assert len(results_t1) > 0
        assert len(results_t2) > 0

        # Cross-thread search should not find results
        cross = store.search("t1", "chocolate cake")
        assert cross == []
