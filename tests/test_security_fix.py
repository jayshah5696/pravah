import asyncio
import sys
import uuid
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def retrieval_module(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    mock_modules = {
        "bm25s": MagicMock(),
        "faiss": MagicMock(),
        "fastavro": MagicMock(),
        "dotenv": MagicMock(),
        "tiktoken": MagicMock(),
        "litellm": MagicMock(),
        "rerankers": MagicMock(),
        "cachetools": MagicMock(),
        "cachetools.keys": MagicMock(),
        "lancedb": MagicMock(),
        "lancedb.pydantic": MagicMock(),
        "lancedb.embeddings": MagicMock(),
        "lancedb.rerankers": MagicMock(),
        "langsmith": MagicMock(),
    }

    mock_modules["dotenv"].load_dotenv = lambda: None
    mock_modules["rerankers"].Reranker = MagicMock()
    mock_modules["cachetools"].LRUCache = MagicMock()
    mock_modules["cachetools"].cached = lambda *args, **kwargs: (lambda f: f)
    mock_modules["cachetools.keys"].hashkey = lambda *args, **kwargs: tuple(args)
    mock_modules["langsmith"].traceable = lambda obj: obj
    mock_modules["langsmith"].Client = MagicMock()
    mock_modules["lancedb.pydantic"].LanceModel = object
    mock_modules["lancedb.pydantic"].Vector = lambda dim: list
    mock_modules["lancedb.embeddings"].OpenAIEmbeddings = MagicMock()
    mock_modules["bm25s"].BM25 = MagicMock()
    mock_modules["bm25s"].tokenize = MagicMock(return_value=[])

    for name, module in mock_modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    sys.modules.pop("pravah.retrieval", None)
    import pravah.retrieval as retrieval_module

    return retrieval_module


@pytest.fixture
def mock_dependencies(retrieval_module):
    with patch.object(retrieval_module.lancedb, "connect"), \
         patch.object(retrieval_module.RetrievalEngine, "create_lancedb_table"), \
         patch.object(retrieval_module.RetrievalEngine, "add_chunks_to_lancedb"), \
         patch.object(retrieval_module.RetrievalEngine, "create_bm25"), \
         patch.object(retrieval_module, "LiteLLMEmbeddingClient"), \
         patch.object(retrieval_module, "Reranker"):
        yield retrieval_module


def test_retrieval_engine_invalid_uuid(mock_dependencies):
    RetrievalEngine = mock_dependencies.RetrievalEngine
    texts = [{"content": "hello", "url": "http://example.com"}]
    with pytest.raises(ValueError, match="Invalid UUID format"):
        RetrievalEngine(texts, uuid_input="not-a-uuid", use_lancedb=True)


def test_retrieval_engine_rejects_sql_injection_payload(mock_dependencies):
    RetrievalEngine = mock_dependencies.RetrievalEngine
    texts = [{"content": "hello", "url": "http://example.com"}]
    payloads = [
        "' OR 1=1 --",
        "'; DROP TABLE pravah_chunks; --",
        "' UNION SELECT * FROM pravah_chunks WHERE '1'='1",
        "abc' OR ''='",
        "1; DELETE FROM pravah_chunks",
    ]
    for payload in payloads:
        with pytest.raises(ValueError, match="Invalid UUID format"):
            RetrievalEngine(texts, uuid_input=payload, use_lancedb=True)


def test_retrieval_engine_rejects_non_string_uuid(mock_dependencies):
    RetrievalEngine = mock_dependencies.RetrievalEngine
    texts = [{"content": "hello", "url": "http://example.com"}]
    with pytest.raises(ValueError, match="Invalid UUID format"):
        RetrievalEngine(texts, uuid_input=12345, use_lancedb=True)
    with pytest.raises(ValueError, match="Invalid UUID format"):
        RetrievalEngine(texts, uuid_input=["not", "a", "uuid"], use_lancedb=True)


def test_retrieval_engine_valid_uuid(mock_dependencies):
    RetrievalEngine = mock_dependencies.RetrievalEngine
    texts = [{"content": "hello", "url": "http://example.com"}]
    valid_uuid = str(uuid.uuid4())

    with patch.object(RetrievalEngine, "chunk_texts", return_value=[]):
        engine = RetrievalEngine(texts, uuid_input=valid_uuid, use_lancedb=False)
        assert engine.uuid_input == valid_uuid


def test_retrieval_engine_none_uuid(mock_dependencies):
    RetrievalEngine = mock_dependencies.RetrievalEngine
    texts = [{"content": "hello", "url": "http://example.com"}]

    with patch.object(RetrievalEngine, "chunk_texts", return_value=[]):
        engine = RetrievalEngine(texts, uuid_input=None, use_lancedb=False)
        assert engine.uuid_input is None

        with patch.object(mock_dependencies.asyncio, "run"):
            engine = RetrievalEngine(texts, uuid_input=None, use_lancedb=True)
            assert engine.uuid_input is not None
            uuid.UUID(engine.uuid_input)


def test_uuid_filter_format(mock_dependencies):
    RetrievalEngine = mock_dependencies.RetrievalEngine
    texts = [{"content": "hello", "url": "http://example.com"}]
    valid_uuid = str(uuid.uuid4())

    with patch.object(RetrievalEngine, "chunk_texts", return_value=[]):
        engine = RetrievalEngine(texts, uuid_input=valid_uuid, use_lancedb=False)
        assert engine._uuid_filter() == f"uuid='{valid_uuid}'"


def test_lancedb_search_quoting(mock_dependencies):
    RetrievalEngine = mock_dependencies.RetrievalEngine
    texts = [{"content": "hello", "url": "http://example.com"}]
    valid_uuid = str(uuid.uuid4())
    expected_filter = f"uuid='{valid_uuid}'"

    async def run_test():
        with patch.object(RetrievalEngine, "chunk_texts", return_value=[]):
            with patch.object(mock_dependencies.asyncio, "run"):
                engine = RetrievalEngine(texts, uuid_input=valid_uuid, use_lancedb=True)
                mock_tbl = MagicMock()
                engine.tbl = mock_tbl

                await engine.lancedb_keyword_search("query")
                mock_tbl.search.assert_called_with("query", query_type="fts")
                mock_tbl.search.return_value.where.assert_called_with(expected_filter)

                mock_tbl.reset_mock()
                await engine.lancedb_semantic_search("query")
                mock_tbl.search.assert_called_with("query", query_type="vector")
                mock_tbl.search.return_value.where.assert_called_with(expected_filter)

                mock_tbl.reset_mock()
                await engine.lancedb_hybrid_search("query")
                mock_tbl.search.assert_called_with("query", query_type="hybrid")
                mock_tbl.search.return_value.where.assert_called_with(expected_filter)

                mock_tbl.reset_mock()
                await engine.lancedb_combined_search("query")
                mock_tbl.search.assert_called_with("query", query_type="hybrid")
                mock_tbl.search.return_value.where.assert_called_with(expected_filter)

    asyncio.run(run_test())
