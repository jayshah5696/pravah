"""Domain exceptions for the Pravah search engine.

Provides a consistent error hierarchy for the core modules:
- SearchError: Web search failures (Tavily, Gemini)
- FetchError: Page content fetching failures
- LLMError: Language model invocation failures
"""


class PravahError(Exception):
    """Base exception for all Pravah errors."""


class SearchError(PravahError):
    """Raised when a web search operation fails."""


class FetchError(PravahError):
    """Raised when fetching page content fails."""


class LLMError(PravahError):
    """Raised when an LLM invocation fails."""
