"""Tests for pravah.history module."""

import tempfile
from pathlib import Path

import pytest
from pravah.history import Conversation, HistoryStore, Message


@pytest.fixture
def store(tmp_path):
    """Create a HistoryStore with a temporary database."""
    db_path = tmp_path / "test_history.db"
    return HistoryStore(db_path)


class TestHistoryStore:
    def test_create_conversation(self, store):
        conv_id = store.create_conversation(title="Test Chat", model="gpt-4o")
        assert conv_id is not None
        assert len(conv_id) > 0

    def test_get_conversation(self, store):
        conv_id = store.create_conversation(title="Test", model="gpt-4o")
        conv = store.get_conversation(conv_id)
        assert conv is not None
        assert conv.title == "Test"
        assert conv.model == "gpt-4o"

    def test_get_missing_conversation(self, store):
        assert store.get_conversation("nonexistent-id") is None

    def test_add_message(self, store):
        conv_id = store.create_conversation(title="Test", model="gpt-4o")
        msg_id = store.add_message(conv_id, "user", "Hello!")
        assert msg_id is not None

        conv = store.get_conversation(conv_id)
        assert conv is not None
        assert len(conv.messages) == 1
        assert conv.messages[0].role == "user"
        assert conv.messages[0].content == "Hello!"

    def test_add_message_auto_creates_conversation(self, store):
        msg_id = store.add_message("auto-conv", "user", "Auto-created")
        assert msg_id is not None

        conv = store.get_conversation("auto-conv")
        assert conv is not None

    def test_first_user_message_sets_title(self, store):
        conv_id = store.create_conversation(title="New Chat", model="test")
        store.add_message(conv_id, "user", "What is Python?")

        conv = store.get_conversation(conv_id)
        assert "What is Python?" in conv.title

    def test_list_conversations(self, store):
        store.create_conversation(title="Chat 1", model="m1")
        store.create_conversation(title="Chat 2", model="m2")

        convs = store.list_conversations()
        assert len(convs) >= 2

    def test_delete_conversation(self, store):
        conv_id = store.create_conversation(title="Delete Me", model="test")
        store.add_message(conv_id, "user", "test message")

        result = store.delete_conversation(conv_id)
        assert result is True
        assert store.get_conversation(conv_id) is None

    def test_delete_nonexistent_returns_false(self, store):
        result = store.delete_conversation("nonexistent")
        assert result is False

    def test_pin_conversation(self, store):
        conv_id = store.create_conversation(title="Pin Me", model="test")
        store.pin_conversation(conv_id, True)

        convs = store.list_conversations()
        pinned = [c for c in convs if c["id"] == conv_id]
        assert len(pinned) == 1
        assert pinned[0]["is_pinned"] is True

    def test_archive_conversation(self, store):
        conv_id = store.create_conversation(title="Archive Me", model="test")
        store.archive_conversation(conv_id, True)

        # Should not appear in default list
        convs = store.list_conversations(include_archived=False)
        archived = [c for c in convs if c["id"] == conv_id]
        assert len(archived) == 0

        # Should appear when including archived
        convs = store.list_conversations(include_archived=True)
        archived = [c for c in convs if c["id"] == conv_id]
        assert len(archived) == 1

    def test_conversation_count(self, store):
        initial = store.get_conversation_count()
        store.create_conversation(title="Count Test", model="test")
        assert store.get_conversation_count() == initial + 1

    def test_conversation_stats(self, store):
        conv_id = store.create_conversation(title="Stats", model="test")
        store.add_message(conv_id, "user", "Hello", tokens_in=10, tokens_out=0)
        store.add_message(
            conv_id, "assistant", "Hi!", tokens_in=0, tokens_out=20, cost_usd=0.001
        )

        stats = store.get_conversation_stats(conv_id)
        assert stats["message_count"] == 2
        assert stats["total_tokens_in"] == 10
        assert stats["total_tokens_out"] == 20

    def test_search_conversations(self, store):
        conv_id = store.create_conversation(title="Quantum Physics", model="test")
        store.add_message(conv_id, "user", "Tell me about quantum entanglement")

        results = store.search_conversations("quantum")
        assert len(results) > 0

    def test_search_empty_query(self, store):
        assert store.search_conversations("") == []
        assert store.search_conversations("   ") == []


class TestMessage:
    def test_to_dict(self):
        msg = Message(role="user", content="Hello")
        d = msg.to_dict()
        assert d["role"] == "user"
        assert d["content"] == "Hello"
        assert "timestamp" in d

    def test_from_dict(self):
        data = {
            "role": "assistant",
            "content": "Hi!",
            "timestamp": "2026-01-01T00:00:00",
        }
        msg = Message.from_dict(data)
        assert msg.role == "assistant"
        assert msg.content == "Hi!"


class TestConversation:
    def test_message_count(self):
        conv = Conversation(
            id="t1",
            title="Test",
            model="test",
            created_at=None,
            updated_at=None,
            messages=[
                Message(role="user", content="Hi"),
                Message(role="assistant", content="Hello"),
            ],
        )
        assert conv.message_count == 2

    def test_preview(self):
        conv = Conversation(
            id="t1",
            title="Test",
            model="test",
            created_at=None,
            updated_at=None,
            messages=[Message(role="user", content="What is Python?")],
        )
        assert "Python" in conv.preview

    def test_empty_preview(self):
        conv = Conversation(
            id="t1",
            title="Test",
            model="test",
            created_at=None,
            updated_at=None,
            messages=[],
        )
        assert conv.preview == "Empty conversation"
