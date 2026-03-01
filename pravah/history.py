"""
Conversation History Storage using DuckDB.

Provides persistent storage for chat conversations with support for:
- Conversation metadata (title, timestamps, model used)
- Message history with tool calls
- Debug metrics (tokens, latency, cost)
"""

import json
import logging
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Generator

import duckdb

logger = logging.getLogger(__name__)


# Default database path
DEFAULT_DB_PATH = Path("pravah_history.db")


@dataclass
class Message:
    """A single message in a conversation."""

    role: str  # "user" or "assistant"
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    tool_calls: list[dict] | None = None
    # Debug metrics
    tokens_in: int | None = None
    tokens_out: int | None = None
    latency_ms: int | None = None
    cost_usd: float | None = None

    def to_dict(self) -> dict:
        """Convert message to dictionary for storage."""
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "tool_calls": self.tool_calls,
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "latency_ms": self.latency_ms,
            "cost_usd": self.cost_usd,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Message":
        """Create message from dictionary."""
        return cls(
            role=data["role"],
            content=data["content"],
            timestamp=datetime.fromisoformat(data["timestamp"])
            if data.get("timestamp")
            else datetime.now(),
            tool_calls=data.get("tool_calls"),
            tokens_in=data.get("tokens_in"),
            tokens_out=data.get("tokens_out"),
            latency_ms=data.get("latency_ms"),
            cost_usd=data.get("cost_usd"),
        )


@dataclass
class Conversation:
    """A conversation with metadata and messages."""

    id: str
    title: str
    model: str
    created_at: datetime
    updated_at: datetime
    messages: list[Message] = field(default_factory=list)

    @property
    def message_count(self) -> int:
        """Number of messages in conversation."""
        return len(self.messages)

    @property
    def preview(self) -> str:
        """First user message as preview, truncated."""
        for msg in self.messages:
            if msg.role == "user":
                preview = msg.content[:100]
                if len(msg.content) > 100:
                    preview += "..."
                return preview
        return "Empty conversation"


class HistoryStore:
    """DuckDB-backed conversation history storage."""

    def __init__(self, db_path: Path | str = DEFAULT_DB_PATH):
        """Initialize the history store.

        Args:
            db_path: Path to the DuckDB database file.
        """
        self.db_path = Path(db_path)
        self._init_schema()

    @contextmanager
    def _connection(self) -> Generator[duckdb.DuckDBPyConnection, None, None]:
        """Context manager for database connections."""
        conn = duckdb.connect(str(self.db_path))
        try:
            yield conn
        finally:
            conn.close()

    def _init_schema(self) -> None:
        """Initialize database schema if not exists."""
        with self._connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id VARCHAR PRIMARY KEY,
                    title VARCHAR NOT NULL,
                    model VARCHAR NOT NULL,
                    created_at TIMESTAMP NOT NULL,
                    updated_at TIMESTAMP NOT NULL,
                    is_pinned BOOLEAN DEFAULT FALSE,
                    is_archived BOOLEAN DEFAULT FALSE
                )
            """)

            # Add new columns if they don't exist (migration for existing DBs)
            try:
                conn.execute(
                    "ALTER TABLE conversations ADD COLUMN is_pinned BOOLEAN DEFAULT FALSE"
                )
            except duckdb.CatalogException:
                logger.debug("Column 'is_pinned' already exists, skipping migration")
            try:
                conn.execute(
                    "ALTER TABLE conversations ADD COLUMN is_archived BOOLEAN DEFAULT FALSE"
                )
            except duckdb.CatalogException:
                logger.debug("Column 'is_archived' already exists, skipping migration")

            conn.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id VARCHAR PRIMARY KEY,
                    conversation_id VARCHAR NOT NULL,
                    role VARCHAR NOT NULL,
                    content TEXT NOT NULL,
                    timestamp TIMESTAMP NOT NULL,
                    tool_calls JSON,
                    tokens_in INTEGER,
                    tokens_out INTEGER,
                    latency_ms INTEGER,
                    cost_usd DOUBLE,
                    FOREIGN KEY (conversation_id) REFERENCES conversations(id)
                )
            """)

            # Index for faster lookups
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_messages_conversation 
                ON messages(conversation_id)
            """)

    def create_conversation(
        self,
        title: str = "New Chat",
        model: str = "unknown",
        conversation_id: str | None = None,
    ) -> str:
        """Create a new conversation.

        Args:
            title: Conversation title.
            model: Model used for the conversation.
            conversation_id: Optional ID (generated if not provided).

        Returns:
            The conversation ID.
        """
        conv_id = conversation_id or str(uuid.uuid4())
        now = datetime.now()

        with self._connection() as conn:
            conn.execute(
                """
                INSERT INTO conversations (id, title, model, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                [conv_id, title, model, now, now],
            )

        return conv_id

    def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        tool_calls: list[dict] | None = None,
        tokens_in: int | None = None,
        tokens_out: int | None = None,
        latency_ms: int | None = None,
        cost_usd: float | None = None,
    ) -> str:
        """Add a message to a conversation.

        Args:
            conversation_id: ID of the conversation.
            role: Message role ("user" or "assistant").
            content: Message content.
            tool_calls: Optional list of tool calls made.
            tokens_in: Input tokens used.
            tokens_out: Output tokens generated.
            latency_ms: Response latency in milliseconds.
            cost_usd: Estimated cost in USD.

        Returns:
            The message ID.
        """
        msg_id = str(uuid.uuid4())
        now = datetime.now()

        with self._connection() as conn:
            # Check if conversation exists, create if not
            result = conn.execute(
                "SELECT id FROM conversations WHERE id = ?",
                [conversation_id],
            ).fetchone()

            if not result:
                self.create_conversation(
                    title="New Chat",
                    model="unknown",
                    conversation_id=conversation_id,
                )

            # Insert message
            conn.execute(
                """
                INSERT INTO messages (
                    id, conversation_id, role, content, timestamp,
                    tool_calls, tokens_in, tokens_out, latency_ms, cost_usd
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    msg_id,
                    conversation_id,
                    role,
                    content,
                    now,
                    json.dumps(tool_calls) if tool_calls else None,
                    tokens_in,
                    tokens_out,
                    latency_ms,
                    cost_usd,
                ],
            )

            # Update conversation timestamp and title if first user message
            if role == "user":
                # Auto-generate title from first user message
                first_msg = conn.execute(
                    """
                    SELECT COUNT(*) FROM messages 
                    WHERE conversation_id = ? AND role = 'user'
                    """,
                    [conversation_id],
                ).fetchone()

                if first_msg and first_msg[0] == 1:
                    # This is the first user message, use it as title
                    title = content[:50] + ("..." if len(content) > 50 else "")
                    conn.execute(
                        "UPDATE conversations SET title = ?, updated_at = ? WHERE id = ?",
                        [title, now, conversation_id],
                    )
                else:
                    conn.execute(
                        "UPDATE conversations SET updated_at = ? WHERE id = ?",
                        [now, conversation_id],
                    )
            else:
                conn.execute(
                    "UPDATE conversations SET updated_at = ? WHERE id = ?",
                    [now, conversation_id],
                )

        return msg_id

    def get_conversation(self, conversation_id: str) -> Conversation | None:
        """Get a conversation by ID.

        Args:
            conversation_id: ID of the conversation.

        Returns:
            The conversation or None if not found.
        """
        with self._connection() as conn:
            conv_row = conn.execute(
                "SELECT id, title, model, created_at, updated_at FROM conversations WHERE id = ?",
                [conversation_id],
            ).fetchone()

            if not conv_row:
                return None

            msg_rows = conn.execute(
                """
                SELECT role, content, timestamp, tool_calls, 
                       tokens_in, tokens_out, latency_ms, cost_usd
                FROM messages
                WHERE conversation_id = ?
                ORDER BY timestamp ASC
                """,
                [conversation_id],
            ).fetchall()

            messages = []
            for row in msg_rows:
                messages.append(
                    Message(
                        role=row[0],
                        content=row[1],
                        timestamp=row[2],
                        tool_calls=json.loads(row[3]) if row[3] else None,
                        tokens_in=row[4],
                        tokens_out=row[5],
                        latency_ms=row[6],
                        cost_usd=row[7],
                    )
                )

            return Conversation(
                id=conv_row[0],
                title=conv_row[1],
                model=conv_row[2],
                created_at=conv_row[3],
                updated_at=conv_row[4],
                messages=messages,
            )

    def list_conversations(
        self,
        limit: int = 50,
        offset: int = 0,
        include_archived: bool = False,
    ) -> list[dict]:
        """List recent conversations.

        Args:
            limit: Maximum number of conversations to return.
            offset: Number of conversations to skip (for pagination).
            include_archived: Whether to include archived conversations.

        Returns:
            List of conversation summaries with pinned conversations first.
        """
        with self._connection() as conn:
            archive_filter = (
                ""
                if include_archived
                else "WHERE COALESCE(c.is_archived, FALSE) = FALSE"
            )

            rows = conn.execute(
                f"""
                SELECT 
                    c.id, 
                    c.title, 
                    c.model, 
                    c.updated_at,
                    COUNT(m.id) as message_count,
                    COALESCE(c.is_pinned, FALSE) as is_pinned,
                    COALESCE(c.is_archived, FALSE) as is_archived
                FROM conversations c
                LEFT JOIN messages m ON c.id = m.conversation_id
                {archive_filter}
                GROUP BY c.id, c.title, c.model, c.updated_at, c.is_pinned, c.is_archived
                ORDER BY COALESCE(c.is_pinned, FALSE) DESC, c.updated_at DESC
                LIMIT ? OFFSET ?
                """,
                [limit, offset],
            ).fetchall()

            return [
                {
                    "id": row[0],
                    "title": row[1],
                    "model": row[2],
                    "updated_at": row[3],
                    "message_count": row[4],
                    "is_pinned": row[5],
                    "is_archived": row[6],
                }
                for row in rows
            ]

    def search_conversations(self, query: str, limit: int = 20) -> list[dict]:
        """Search conversations by title or message content.

        Args:
            query: Search query string.
            limit: Maximum number of results.

        Returns:
            List of matching conversations with matched content preview.
        """
        if not query or not query.strip():
            return []

        search_term = f"%{query.strip()}%"

        with self._connection() as conn:
            rows = conn.execute(
                """
                SELECT DISTINCT
                    c.id,
                    c.title,
                    c.model,
                    c.updated_at,
                    (SELECT COUNT(*) FROM messages WHERE conversation_id = c.id) as message_count,
                    COALESCE(c.is_pinned, FALSE) as is_pinned,
                    COALESCE(
                        (SELECT content FROM messages 
                         WHERE conversation_id = c.id AND content ILIKE ? 
                         LIMIT 1),
                        c.title
                    ) as matched_content
                FROM conversations c
                LEFT JOIN messages m ON c.id = m.conversation_id
                WHERE c.title ILIKE ? OR m.content ILIKE ?
                ORDER BY c.updated_at DESC
                LIMIT ?
                """,
                [search_term, search_term, search_term, limit],
            ).fetchall()

            return [
                {
                    "id": row[0],
                    "title": row[1],
                    "model": row[2],
                    "updated_at": row[3],
                    "message_count": row[4],
                    "is_pinned": row[5],
                    "matched_content": row[6][:100] + "..."
                    if row[6] and len(row[6]) > 100
                    else row[6],
                }
                for row in rows
            ]

    def pin_conversation(self, conversation_id: str, pinned: bool = True) -> None:
        """Pin or unpin a conversation.

        Args:
            conversation_id: ID of the conversation.
            pinned: True to pin, False to unpin.
        """
        with self._connection() as conn:
            conn.execute(
                "UPDATE conversations SET is_pinned = ? WHERE id = ?",
                [pinned, conversation_id],
            )

    def archive_conversation(self, conversation_id: str, archived: bool = True) -> None:
        """Archive or unarchive a conversation.

        Args:
            conversation_id: ID of the conversation.
            archived: True to archive, False to unarchive.
        """
        with self._connection() as conn:
            conn.execute(
                "UPDATE conversations SET is_archived = ? WHERE id = ?",
                [archived, conversation_id],
            )

    def get_conversation_count(self, include_archived: bool = False) -> int:
        """Get total number of conversations.

        Args:
            include_archived: Whether to include archived conversations.

        Returns:
            Total conversation count.
        """
        with self._connection() as conn:
            if include_archived:
                row = conn.execute("SELECT COUNT(*) FROM conversations").fetchone()
            else:
                row = conn.execute(
                    "SELECT COUNT(*) FROM conversations WHERE COALESCE(is_archived, FALSE) = FALSE"
                ).fetchone()
            return row[0] if row else 0

    def delete_conversation(self, conversation_id: str) -> bool:
        """Delete a conversation and all its messages.

        Args:
            conversation_id: ID of the conversation to delete.

        Returns:
            True if deleted, False if not found.
        """
        with self._connection() as conn:
            # Check if conversation exists first
            exists = conn.execute(
                "SELECT 1 FROM conversations WHERE id = ?",
                [conversation_id],
            ).fetchone()

            if not exists:
                return False

            # Delete messages first (foreign key)
            conn.execute(
                "DELETE FROM messages WHERE conversation_id = ?",
                [conversation_id],
            )
            conn.execute(
                "DELETE FROM conversations WHERE id = ?",
                [conversation_id],
            )

            return True

    def update_conversation_model(self, conversation_id: str, model: str) -> None:
        """Update the model used for a conversation.

        Args:
            conversation_id: ID of the conversation.
            model: New model name.
        """
        with self._connection() as conn:
            conn.execute(
                "UPDATE conversations SET model = ? WHERE id = ?",
                [model, conversation_id],
            )

    def get_conversation_stats(self, conversation_id: str) -> dict:
        """Get aggregated stats for a conversation.

        Args:
            conversation_id: ID of the conversation.

        Returns:
            Dictionary with total tokens, cost, and latency.
        """
        with self._connection() as conn:
            row = conn.execute(
                """
                SELECT 
                    SUM(tokens_in) as total_tokens_in,
                    SUM(tokens_out) as total_tokens_out,
                    SUM(cost_usd) as total_cost,
                    AVG(latency_ms) as avg_latency,
                    COUNT(*) as message_count
                FROM messages
                WHERE conversation_id = ?
                """,
                [conversation_id],
            ).fetchone()

            return {
                "total_tokens_in": row[0] or 0,
                "total_tokens_out": row[1] or 0,
                "total_cost_usd": row[2] or 0.0,
                "avg_latency_ms": row[3] or 0,
                "message_count": row[4] or 0,
            }


# Global instance for convenience
_store: HistoryStore | None = None


def get_history_store(db_path: Path | str = DEFAULT_DB_PATH) -> HistoryStore:
    """Get or create the global history store instance.

    Args:
        db_path: Path to the database file.

    Returns:
        The history store instance.
    """
    global _store
    if _store is None:
        _store = HistoryStore(db_path)
    return _store
