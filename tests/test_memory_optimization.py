import pytest

from pravah.memory import SessionMemory


def test_extract_snippet_basic():
    memory = SessionMemory(thread_id="test")
    content = "This is a simple test content for snippet extraction."
    terms = ["simple", "test"]
    snippet = memory._extract_snippet(content, terms, context_chars=10)
    assert "simple" in snippet
    assert snippet.startswith("This is a simple")


def test_extract_snippet_no_match():
    memory = SessionMemory(thread_id="test")
    content = "This is a simple test content for snippet extraction."
    terms = ["missing"]
    snippet = memory._extract_snippet(content, terms, context_chars=10)
    assert snippet == content[:20] + "..."


def test_extract_snippet_case_insensitive():
    memory = SessionMemory(thread_id="test")
    content = "This is a SIMPLE test content."
    terms = ["simple"]
    snippet = memory._extract_snippet(content, terms, context_chars=10)
    assert "SIMPLE" in snippet


def test_extract_snippet_multiple_terms():
    memory = SessionMemory(thread_id="test")
    content = "First term is here, second term is there."
    terms = ["second", "first"]
    snippet = memory._extract_snippet(content, terms, context_chars=10)
    assert "First" in snippet
    assert snippet.startswith("First term")


def test_extract_snippet_overlap_or_subset():
    memory = SessionMemory(thread_id="test")
    content = "The word 'apple' is better than 'app'."
    terms = ["app", "apple"]
    snippet = memory._extract_snippet(content, terms, context_chars=10)
    assert "apple" in snippet


def test_extract_snippet_near_end():
    memory = SessionMemory(thread_id="test")
    content = "Short content with match at the end."
    terms = ["end"]
    snippet = memory._extract_snippet(content, terms, context_chars=10)
    assert snippet.endswith("the end.")
