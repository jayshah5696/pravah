from __future__ import annotations

import sys
from unittest.mock import MagicMock


# Some tests monkeypatch heavy optional dependencies at module-import time.
# Ensure later tests can still import the real packages by clearing those mocks.
def pytest_runtest_setup(item):
    for module_name in [
        "numpy",
        "anthropic",
        "lancedb",
        "lancedb.pydantic",
        "lancedb.embeddings",
        "lancedb.rerankers",
        "bm25s",
        "faiss",
        "fastavro",
        "tiktoken",
        "litellm",
        "rerankers",
        "cachetools",
        "cachetools.keys",
        "langsmith",
        "dotenv",
    ]:
        module = sys.modules.get(module_name)
        if isinstance(module, MagicMock):
            sys.modules.pop(module_name, None)
