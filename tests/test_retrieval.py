import unittest
import numpy as np
from pravah.retrieval import RetrievalEngine

class TestNormalizeScores(unittest.TestCase):
    def test_normalize_scores_standard(self):
        """Test normalization with a standard range of values."""
        scores = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        # Calling as static method
        normalized = RetrievalEngine.normalize_scores(scores)
        expected = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        np.testing.assert_array_almost_equal(normalized, expected)

    def test_normalize_scores_identical_values(self):
        """Test normalization when all values are identical."""
        scores = np.array([10.0, 10.0, 10.0])
        normalized = RetrievalEngine.normalize_scores(scores)
        # Should return zeros and avoid division by zero
        expected = np.array([0.0, 0.0, 0.0])
        np.testing.assert_array_almost_equal(normalized, expected)

    def test_normalize_scores_single_element(self):
        """Test normalization with a single-element array."""
        scores = np.array([5.0])
        normalized = RetrievalEngine.normalize_scores(scores)
        expected = np.array([0.0])
        np.testing.assert_array_almost_equal(normalized, expected)

    def test_normalize_scores_empty_array(self):
        """Test normalization with an empty array."""
        scores = np.array([])
        normalized = RetrievalEngine.normalize_scores(scores)
        self.assertEqual(normalized.size, 0)

    def test_normalize_scores_negative_values(self):
        """Test normalization with negative values."""
        scores = np.array([-10.0, 0.0, 10.0])
        normalized = RetrievalEngine.normalize_scores(scores)
        expected = np.array([0.0, 0.5, 1.0])
        np.testing.assert_array_almost_equal(normalized, expected)

if __name__ == '__main__':
    unittest.main()
