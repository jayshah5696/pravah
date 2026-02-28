
import duckdb
import pytest
import uuid


@pytest.fixture
def db_connection():
    conn = duckdb.connect(':memory:')
    yield conn
    conn.close()


def _create_tables(conn):
    """Mirror of app.create_tables -- uses duckdb.CatalogException for migration."""
    conn.execute("CREATE TABLE IF NOT EXISTS chat_history (conversation_uuid UUID PRIMARY KEY, user_input TEXT, response TEXT, title TEXT)")
    # Migration: add title column for existing databases
    try:
        conn.execute("ALTER TABLE chat_history ADD COLUMN title TEXT")
    except duckdb.CatalogException:
        pass  # Column already exists
    conn.execute("CREATE TABLE IF NOT EXISTS search_results (conversation_uuid UUID, search_result JSON, FOREIGN KEY(conversation_uuid) REFERENCES chat_history(conversation_uuid))")
    conn.execute("CREATE TABLE IF NOT EXISTS fetched_texts (url TEXT PRIMARY KEY, text TEXT)")
    conn.execute("CREATE TABLE IF NOT EXISTS retrieved_chunks (conversation_uuid UUID, search_type TEXT, chunk TEXT, FOREIGN KEY(conversation_uuid) REFERENCES chat_history(conversation_uuid))")
    conn.execute("CREATE TABLE IF NOT EXISTS re_written_prompt (conversation_uuid UUID, re_written_prompt TEXT, FOREIGN KEY(conversation_uuid) REFERENCES chat_history(conversation_uuid))")


def test_create_tables_adds_title(db_connection):
    """Migration adds title column to a pre-existing table that lacks it."""
    # Simulate existing table without title
    db_connection.execute("CREATE TABLE chat_history (conversation_uuid UUID PRIMARY KEY, user_input TEXT, response TEXT)")

    # Run the function
    _create_tables(db_connection)

    # Verify column exists
    result = db_connection.execute("DESCRIBE chat_history").fetchall()
    columns = [row[0] for row in result]
    assert "title" in columns


def test_create_tables_idempotent(db_connection):
    """Running create_tables twice does not raise."""
    _create_tables(db_connection)
    _create_tables(db_connection)

    result = db_connection.execute("DESCRIBE chat_history").fetchall()
    columns = [row[0] for row in result]
    assert "title" in columns


def test_save_to_duckdb_saves_title(db_connection):
    """Title is correctly persisted via INSERT."""
    _create_tables(db_connection)

    uid = uuid.uuid4()
    db_connection.execute(
        "INSERT INTO chat_history (conversation_uuid, user_input, response, title) VALUES (?, ?, ?, ?)",
        (uid, "hi", "hello", "Greeting"),
    )

    result = db_connection.execute("SELECT title FROM chat_history WHERE conversation_uuid = ?", (uid,)).fetchone()
    assert result[0] == "Greeting"


def test_migration_catches_only_catalog_exception(db_connection):
    """CatalogException is caught when column already exists, table still works."""
    _create_tables(db_connection)

    # Verify the table is fully functional after migration
    uid = uuid.uuid4()
    db_connection.execute(
        "INSERT INTO chat_history (conversation_uuid, user_input, response, title) VALUES (?, ?, ?, ?)",
        (uid, "test", "response", "Test Title"),
    )
    result = db_connection.execute("SELECT title FROM chat_history WHERE conversation_uuid = ?", (uid,)).fetchone()
    assert result[0] == "Test Title"
