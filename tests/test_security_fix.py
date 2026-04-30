import sys
from unittest.mock import MagicMock, patch, AsyncMock

# Mock dependencies that are not installed before importing RetrievalEngine
sys.modules['numpy'] = MagicMock()
sys.modules['bm25s'] = MagicMock()
sys.modules['faiss'] = MagicMock()
sys.modules['fastavro'] = MagicMock()
sys.modules['dotenv'] = MagicMock()
sys.modules['tiktoken'] = MagicMock()
sys.modules['litellm'] = MagicMock()
sys.modules['rerankers'] = MagicMock()
sys.modules['cachetools'] = MagicMock()
sys.modules['cachetools.keys'] = MagicMock()
sys.modules['lancedb'] = MagicMock()
sys.modules['lancedb.pydantic'] = MagicMock()
sys.modules['lancedb.embeddings'] = MagicMock()
sys.modules['lancedb.rerankers'] = MagicMock()

# Special mock for langsmith to keep the traceable decorator working
mock_langsmith = MagicMock()
def mock_traceable(obj):
    return obj
mock_langsmith.traceable = mock_traceable
sys.modules['langsmith'] = mock_langsmith

import pytest
import uuid
import asyncio
from pravah.retrieval import RetrievalEngine

@pytest.fixture
def mock_dependencies():
    with patch('pravah.retrieval.lancedb.connect'), \
         patch('pravah.retrieval.RetrievalEngine.create_lancedb_table'), \
         patch('pravah.retrieval.RetrievalEngine.add_chunks_to_lancedb'), \
         patch('pravah.retrieval.RetrievalEngine.create_bm25'), \
         patch('pravah.retrieval.LiteLLMEmbeddingClient'), \
         patch('pravah.retrieval.Reranker'):
        yield

def test_retrieval_engine_invalid_uuid(mock_dependencies):
    texts = [{'content': 'hello', 'url': 'http://example.com'}]
    invalid_uuid = "not-a-uuid"

    with pytest.raises(ValueError, match="Invalid UUID format"):
        RetrievalEngine(texts, uuid_input=invalid_uuid, use_lancedb=True)

def test_retrieval_engine_rejects_sql_injection_payload(mock_dependencies):
    """Verify that SQL injection payloads are rejected by UUID validation."""
    texts = [{'content': 'hello', 'url': 'http://example.com'}]

    injection_payloads = [
        "' OR 1=1 --",
        "'; DROP TABLE pravah_chunks; --",
        "' UNION SELECT * FROM pravah_chunks WHERE '1'='1",
        "abc' OR ''='",
        "1; DELETE FROM pravah_chunks",
    ]

    for payload in injection_payloads:
        with pytest.raises(ValueError, match="Invalid UUID format"):
            RetrievalEngine(texts, uuid_input=payload, use_lancedb=True)

def test_retrieval_engine_rejects_non_string_uuid(mock_dependencies):
    """Verify that non-string types are rejected by UUID validation."""
    texts = [{'content': 'hello', 'url': 'http://example.com'}]

    with pytest.raises(ValueError, match="Invalid UUID format"):
        RetrievalEngine(texts, uuid_input=12345, use_lancedb=True)

    with pytest.raises(ValueError, match="Invalid UUID format"):
        RetrievalEngine(texts, uuid_input=['not', 'a', 'uuid'], use_lancedb=True)

def test_retrieval_engine_valid_uuid(mock_dependencies):
    texts = [{'content': 'hello', 'url': 'http://example.com'}]
    valid_uuid = str(uuid.uuid4())

    with patch('pravah.retrieval.RetrievalEngine.chunk_texts') as mock_chunks:
        mock_chunks.return_value = []
        engine = RetrievalEngine(texts, uuid_input=valid_uuid, use_lancedb=False)
        assert engine.uuid_input == valid_uuid

def test_retrieval_engine_none_uuid(mock_dependencies):
    texts = [{'content': 'hello', 'url': 'http://example.com'}]

    with patch('pravah.retrieval.RetrievalEngine.chunk_texts') as mock_chunks:
        mock_chunks.return_value = []
        # Case 1: use_lancedb=False, uuid_input=None should stay None
        engine = RetrievalEngine(texts, uuid_input=None, use_lancedb=False)
        assert engine.uuid_input is None

        # Case 2: use_lancedb=True, uuid_input=None should generate a new UUID
        with patch('pravah.retrieval.asyncio.run'):
            engine = RetrievalEngine(texts, uuid_input=None, use_lancedb=True)
            assert engine.uuid_input is not None
            # Verify it's a valid UUID
            uuid.UUID(engine.uuid_input)

@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_uuid_filter_format(mock_dependencies):
    """Verify _uuid_filter produces correctly quoted filter strings."""
    texts = [{'content': 'hello', 'url': 'http://example.com'}]
    valid_uuid = str(uuid.uuid4())

    with patch('pravah.retrieval.RetrievalEngine.chunk_texts') as mock_chunks:
        mock_chunks.return_value = []
        engine = RetrievalEngine(texts, uuid_input=valid_uuid, use_lancedb=False)
        expected = f"uuid='{valid_uuid}'"
        assert engine._uuid_filter() == expected

@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_lancedb_search_quoting(mock_dependencies):
    """Verify all LanceDB search methods apply the UUID filter."""
    texts = [{'content': 'hello', 'url': 'http://example.com'}]
    valid_uuid = str(uuid.uuid4())
    expected_filter = f"uuid='{valid_uuid}'"

    async def run_test():
        with patch('pravah.retrieval.RetrievalEngine.chunk_texts') as mock_chunks:
            mock_chunks.return_value = []
            with patch('pravah.retrieval.asyncio.run'):
                engine = RetrievalEngine(texts, uuid_input=valid_uuid, use_lancedb=True)

                # Build a proper mock chain that tracks calls correctly
                mock_tbl = MagicMock()
                engine.tbl = mock_tbl

                # Test lancedb_keyword_search
                await engine.lancedb_keyword_search("query")
                mock_tbl.search.assert_called_with("query", query_type='fts')
                mock_tbl.search.return_value.where.assert_called_with(expected_filter)

                # Test lancedb_semantic_search
                mock_tbl.reset_mock()
                await engine.lancedb_semantic_search("query")
                mock_tbl.search.assert_called_with("query", query_type='vector')
                mock_tbl.search.return_value.where.assert_called_with(expected_filter)

                # Test lancedb_hybrid_search
                mock_tbl.reset_mock()
                await engine.lancedb_hybrid_search("query")
                mock_tbl.search.assert_called_with("query", query_type='hybrid')
                mock_tbl.search.return_value.where.assert_called_with(expected_filter)

                # Test lancedb_combined_search
                mock_tbl.reset_mock()
                await engine.lancedb_combined_search("query")
                mock_tbl.search.assert_called_with("query", query_type='hybrid')
                mock_tbl.search.return_value.where.assert_called_with(expected_filter)

    asyncio.run(run_test())
