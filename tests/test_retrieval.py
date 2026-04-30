import sys
from unittest.mock import MagicMock

import numpy as np


# Isolate this test module from optional heavy dependencies in pravah.retrieval.
mock_bm25s = MagicMock()
mock_bm25s.BM25 = MagicMock()
mock_bm25s.tokenize = MagicMock(return_value=[])
sys.modules.setdefault("bm25s", mock_bm25s)
sys.modules.setdefault("faiss", MagicMock())
sys.modules.setdefault("fastavro", MagicMock())
sys.modules.setdefault("tiktoken", MagicMock())
sys.modules.setdefault("litellm", MagicMock())
sys.modules.setdefault("rerankers", MagicMock())
sys.modules.setdefault("lancedb", MagicMock())
sys.modules.setdefault("lancedb.pydantic", MagicMock())
sys.modules.setdefault("lancedb.embeddings", MagicMock())
sys.modules.setdefault("langsmith", MagicMock(traceable=lambda obj: obj, Client=MagicMock()))

from pravah.retrieval import RetrievalEngine


class TestNormalizeScores:
    def test_normalize_scores_standard(self):
        scores = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        normalized = RetrievalEngine.normalize_scores(scores)
        expected = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        np.testing.assert_array_almost_equal(normalized, expected)

    def test_normalize_scores_identical_values(self):
        scores = np.array([10.0, 10.0, 10.0])
        normalized = RetrievalEngine.normalize_scores(scores)
        expected = np.array([0.0, 0.0, 0.0])
        np.testing.assert_array_almost_equal(normalized, expected)

    def test_normalize_scores_single_element(self):
        scores = np.array([5.0])
        normalized = RetrievalEngine.normalize_scores(scores)
        expected = np.array([0.0])
        np.testing.assert_array_almost_equal(normalized, expected)

    def test_normalize_scores_empty_array(self):
        scores = np.array([])
        normalized = RetrievalEngine.normalize_scores(scores)
        assert normalized.size == 0

    def test_normalize_scores_negative_values(self):
        scores = np.array([-10.0, 0.0, 10.0])
        normalized = RetrievalEngine.normalize_scores(scores)
        expected = np.array([0.0, 0.5, 1.0])
        np.testing.assert_array_almost_equal(normalized, expected)
