import unittest
from pravah.memory import set_current_thread_id, get_current_thread_id

class TestMemoryThreadId(unittest.TestCase):
    def setUp(self):
        # Reset the global state before each test
        set_current_thread_id(None)

    def tearDown(self):
        # Reset the global state after each test
        set_current_thread_id(None)

    def test_initial_state(self):
        """Verify that the initial thread ID is None."""
        self.assertIsNone(get_current_thread_id())

    def test_set_get_thread_id(self):
        """Verify that setting a thread ID and then getting it returns the expected value."""
        test_id = "test-thread-123"
        set_current_thread_id(test_id)
        self.assertEqual(get_current_thread_id(), test_id)

    def test_overwrite_thread_id(self):
        """Verify that setting a new thread ID overwrites the previous one."""
        set_current_thread_id("first-id")
        set_current_thread_id("second-id")
        self.assertEqual(get_current_thread_id(), "second-id")

    def test_set_none_thread_id(self):
        """Verify that setting the thread ID back to None works."""
        set_current_thread_id("some-id")
        set_current_thread_id(None)
        self.assertIsNone(get_current_thread_id())

if __name__ == "__main__":
    unittest.main()
