"""Tests for pravah.exceptions module."""

from pravah.exceptions import FetchError, LLMError, PravahError, SearchError


class TestExceptionHierarchy:
    def test_search_error_is_pravah_error(self):
        assert issubclass(SearchError, PravahError)

    def test_fetch_error_is_pravah_error(self):
        assert issubclass(FetchError, PravahError)

    def test_llm_error_is_pravah_error(self):
        assert issubclass(LLMError, PravahError)

    def test_all_are_exceptions(self):
        for exc_class in (PravahError, SearchError, FetchError, LLMError):
            assert issubclass(exc_class, Exception)

    def test_can_catch_pravah_error(self):
        try:
            raise SearchError("search failed")
        except PravahError as e:
            assert "search failed" in str(e)
