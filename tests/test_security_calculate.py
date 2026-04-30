import pytest
import math
import sys
from unittest.mock import MagicMock

# Mocking external dependencies
mock_modules = [
    "langchain_core",
    "langchain_core.tools",
    "langchain_core.runnables",
    "pravah.search",
    "pravah.llm",
    "pravah.memory",
    "litellm",
    "langgraph",
    "langgraph.checkpoint",
    "aiosqlite",
    "tavily",
    "beautifulsoup4",
    "duckduckgo_search",
    "bm25s",
    "rerankers",
    "faiss",
    "lancedb",
    "tantivy",
    "openai",
    "anthropic",
    "cohere",
    "google.genai",
    "pymupdf4llm",
    "tiktoken",
    "streamlit",
    "aiohttp",
    "duckdb",
]

for mod in mock_modules:
    sys.modules[mod] = MagicMock()

# Explicitly make @tool an identity decorator
def identity_decorator(f):
    return f

sys.modules['langchain_core.tools'].tool = identity_decorator

# Now we can import calculate
from pravah.tools import calculate

def test_calculate_basic():
    """Test basic math operations."""
    assert "2 + 2 = 4" in calculate("2 + 2")
    assert "10 / 2 = 5.0" in calculate("10 / 2")
    assert "2 ** 3 = 8" in calculate("2 ** 3")
    assert "2 ^ 3 = 8" in calculate("2 ^ 3") # Test ^ replacement

def test_calculate_math_functions():
    """Test allowed math functions."""
    assert "sqrt(16) = 4.0" in calculate("sqrt(16)")
    assert "sin(0) = 0.0" in calculate("sin(0)")
    assert f"pi = {math.pi}" in calculate("pi")

def test_calculate_invalid_expression():
    """Test handling of invalid expressions."""
    result = calculate("not a math expression")
    assert "Could not calculate" in result

def test_calculate_vulnerability_rce():
    """
    Test for the RCE vulnerability.
    """
    # This exploit attempts to call 'echo VULNERABLE'
    exploit = "[c for c in ().__class__.__mro__[-1].__subclasses__() if c.__name__ == '_wrap_close'][0].__init__.__globals__['system']('echo VULNERABLE')"

    result = calculate(exploit)

    # After fix, this should return an error message
    assert "Could not calculate" in result
    assert "Unsupported AST node type" in result or "Name" in result
