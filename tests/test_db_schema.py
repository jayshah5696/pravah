
import duckdb
import pytest
import uuid

@pytest.fixture
def db_connection():
    conn = duckdb.connect(':memory:')
    yield conn
    conn.close()

def create_tables_v2(conn):
    conn.execute("CREATE TABLE IF NOT EXISTS chat_history (conversation_uuid UUID PRIMARY KEY, user_input TEXT, response TEXT)")
    conn.execute("CREATE TABLE IF NOT EXISTS search_results (conversation_uuid UUID, search_result JSON, FOREIGN KEY(conversation_uuid) REFERENCES chat_history(conversation_uuid))")
    conn.execute("CREATE TABLE IF NOT EXISTS fetched_texts (url TEXT PRIMARY KEY, text TEXT)")
    conn.execute("CREATE TABLE IF NOT EXISTS retrieved_chunks (conversation_uuid UUID, search_type TEXT, chunk TEXT, FOREIGN KEY(conversation_uuid) REFERENCES chat_history(conversation_uuid))")
    conn.execute("CREATE TABLE IF NOT EXISTS re_written_prompt (conversation_uuid UUID, re_written_prompt TEXT, FOREIGN KEY(conversation_uuid) REFERENCES chat_history(conversation_uuid))")

    # Migration to add title
    try:
        conn.execute("ALTER TABLE chat_history ADD COLUMN title TEXT")
    except:
        pass

def test_create_tables_adds_title(db_connection):
    # Simulate existing table without title
    db_connection.execute("CREATE TABLE chat_history (conversation_uuid UUID PRIMARY KEY, user_input TEXT, response TEXT)")

    # Run the function
    create_tables_v2(db_connection)

    # Verify column exists
    result = db_connection.execute("DESCRIBE chat_history").fetchall()
    columns = [row[0] for row in result]
    assert "title" in columns

def test_create_tables_idempotent(db_connection):
    # Run twice
    create_tables_v2(db_connection)
    create_tables_v2(db_connection)

    result = db_connection.execute("DESCRIBE chat_history").fetchall()
    columns = [row[0] for row in result]
    assert "title" in columns

def save_to_duckdb_v2(conn, conversation_uuid, prompt, full_response, search_results, texts, urls, context_keyword, context_reranker, re_written_prompt, title):
    conn.execute("INSERT INTO chat_history (conversation_uuid, user_input, response, title) VALUES (?, ?, ?, ?)", (conversation_uuid, prompt, full_response, title))
    # Simplified other inserts for test
    pass

def test_save_to_duckdb_saves_title(db_connection):
    create_tables_v2(db_connection)

    uid = uuid.uuid4()
    save_to_duckdb_v2(db_connection, uid, "hi", "hello", "{}", [], [], [], [], "hi", "Greeting")

    result = db_connection.execute("SELECT title FROM chat_history WHERE conversation_uuid = ?", (uid,)).fetchone()
    assert result[0] == "Greeting"
