import unittest
from pravah.memory import SessionMemory

class TestSessionMemory(unittest.TestCase):
    def test_search_functionality(self):
        memory = SessionMemory(thread_id="test_thread")
        memory.add_document("url1", "Python is a popular programming language.", "Python")
        memory.add_document("url2", "Java is another programming language.", "Java")

        # Test case 1: Single term search
        results = memory.search("python")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["url"], "url1")
        self.assertIn("python", results[0]["matched_terms"])

        # Test case 2: Multiple terms search
        results = memory.search("programming language")
        self.assertEqual(len(results), 2)
        # Both should match, check scores or existence
        urls = [res["url"] for res in results]
        self.assertIn("url1", urls)
        self.assertIn("url2", urls)

        # Test case 3: No results
        results = memory.search("nonexistent")
        self.assertEqual(len(results), 0)

        # Test case 4: Empty query
        results = memory.search("")
        self.assertEqual(len(results), 0)

if __name__ == "__main__":
    unittest.main()
