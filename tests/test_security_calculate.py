import math
import sys
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def calculate_module(monkeypatch):
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
        monkeypatch.setitem(sys.modules, mod, MagicMock())

    sys.modules["langchain_core.tools"].tool = lambda f: f

    sys.modules.pop("pravah.tools", None)
    import pravah.tools as tools_module

    return tools_module


def test_calculate_basic(calculate_module):
    calculate = calculate_module.calculate
    assert "2 + 2 = 4" in calculate("2 + 2")
    assert "10 / 2 = 5.0" in calculate("10 / 2")
    assert "2 ** 3 = 8" in calculate("2 ** 3")
    assert "2 ^ 3 = 8" in calculate("2 ^ 3")


def test_calculate_math_functions(calculate_module):
    calculate = calculate_module.calculate
    assert "sqrt(16) = 4.0" in calculate("sqrt(16)")
    assert "sin(0) = 0.0" in calculate("sin(0)")
    assert f"pi = {math.pi}" in calculate("pi")


def test_calculate_invalid_expression(calculate_module):
    result = calculate_module.calculate("not a math expression")
    assert "Could not calculate" in result


def test_calculate_vulnerability_rce(calculate_module):
    exploit = "[c for c in ().__class__.__mro__[-1].__subclasses__() if c.__name__ == '_wrap_close'][0].__init__.__globals__['system']('echo VULNERABLE')"
    result = calculate_module.calculate(exploit)
    assert "Could not calculate" in result
    assert "Unsupported AST node type" in result or "Name" in result
