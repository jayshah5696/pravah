
import pytest
from unittest.mock import MagicMock, patch
import os
import sys

# Set dummy environment variables before any imports
os.environ.setdefault('TVLY_API_KEY', 'test')
os.environ.setdefault('OPENAI_API_KEY', 'test')
os.environ.setdefault('COHERE_API_KEY', 'test')
os.environ.setdefault('BRAVE_API_KEY', 'test')
os.environ.setdefault('JINA_API_KEY', 'test')
os.environ.setdefault('LANGCHAIN_API_KEY', 'test')
os.environ.setdefault('LANGCHAIN_PROJECT', 'test')
os.environ.setdefault('GROQ_API_KEY', 'test')

# Mock heavy dependencies that are not needed for config tests
_mock_modules = [
    'streamlit', 'rerankers', 'langsmith', 'langsmith.run_trees',
    'pravah.retrieval', 'pravah.search',
    'tavily', 'bs4', 'duckduckgo_search', 'brave_search',
    'fastavro', 'faiss', 'lancedb', 'tantivy',
    'pymupdf4llm', 'tiktoken', 'bm25s',
]

_saved = {}
for mod in _mock_modules:
    if mod not in sys.modules:
        _saved[mod] = None
        sys.modules[mod] = MagicMock()
    else:
        _saved[mod] = sys.modules[mod]

# Now import app -- the heavy deps are mocked
import app


def test_config_has_title_model():
    config = app.Config(search_tvly_api_key="test")
    assert hasattr(config, 'title_model')
    assert hasattr(config, 'title_model_temperature')
    assert config.title_model == 'groq/llama-3.1-8b-instant'


def test_check_api_keys_checks_title_model():
    config = app.Config(search_tvly_api_key="test")
    config.title_model = "openai/gpt-4"

    with patch.dict(os.environ, {}, clear=True):
        with patch.object(app, 'st') as mock_st:
            mock_st.text_input.return_value = None

            app.check_api_keys(config)

            calls = [str(args[0]) for args, _ in mock_st.info.call_args_list]
            assert any("OpenAI API key" in c for c in calls)
