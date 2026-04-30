"""
Tests for File Upload / Local RAG functionality.

Architecture: Uploads are processed into LanceDB, then searched via
a `search_uploads` tool that the agent can call (mirroring `search_memory`).

Best Practice: Two-tool pattern with pagination
- search_uploads: Returns top-k previews with chunk_ids
- read_upload_chunk: Returns full chunk with prev/next navigation

These tests verify that:
1. Files can be uploaded and parsed (markitdown for PDF/DOCX/PPTX)
2. Content is chunked and indexed in LanceDB
3. search_uploads returns paginated previews
4. read_upload_chunk enables scrolling through content
5. Uploads are scoped to conversation (thread_id)
"""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
import tempfile


class TestFileProcessing:
    """Test file upload processing with markitdown (pravah/uploads.py)."""

    def test_process_txt_file(self):
        """Should extract text from .txt files and index in LanceDB."""
        from pravah.uploads import FileUploadManager

        manager = FileUploadManager()

        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("This is test content for the file upload feature.")
            f.flush()

            result = manager.process_file(
                file_path=Path(f.name),
                filename="test.txt",
                conversation_id="test-123"
            )

            assert result["success"] is True
            assert result["chunk_count"] > 0

    def test_process_md_file(self):
        """Should extract text from .md files."""
        from pravah.uploads import FileUploadManager

        manager = FileUploadManager()

        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            f.write("# Heading\n\nThis is **markdown** content with details.")
            f.flush()

            result = manager.process_file(
                file_path=Path(f.name),
                filename="readme.md",
                conversation_id="test-123"
            )

            assert result["success"] is True

    def test_process_csv_file(self):
        """Should extract rows from .csv files as text chunks."""
        from pravah.uploads import FileUploadManager

        manager = FileUploadManager()

        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("name,value,description\ntest,123,A test row\nexample,456,Another row")
            f.flush()

            result = manager.process_file(
                file_path=Path(f.name),
                filename="data.csv",
                conversation_id="test-123"
            )

            assert result["success"] is True

    def test_unsupported_file_type_raises_error(self):
        """Should raise error for unsupported file types like .exe."""
        from pravah.uploads import FileUploadManager, UnsupportedFileTypeError

        manager = FileUploadManager()

        with tempfile.NamedTemporaryFile(suffix='.exe', delete=False) as f:
            with pytest.raises(UnsupportedFileTypeError):
                manager.process_file(
                    file_path=Path(f.name),
                    filename="virus.exe",
                    conversation_id="test-123"
                )


class TestSearchUploadsTool:
    """Test the search_uploads tool (tools.py integration)."""

    def test_search_returns_relevant_chunks(self):
        """search_uploads should return chunks matching query."""
        from pravah.uploads import FileUploadManager

        manager = FileUploadManager()
        conversation_id = "test-search-123"

        # Index some content first
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Python is a programming language used for web development and data science.")
            f.flush()
            manager.process_file(Path(f.name), "python_intro.txt", conversation_id)

        results = manager.search("programming language", conversation_id=conversation_id)

        assert len(results) > 0
        assert any("python" in r["content"].lower() for r in results)

    def test_search_scoped_to_conversation(self):
        """Search should only return results from the same conversation/thread."""
        from pravah.uploads import FileUploadManager

        manager = FileUploadManager()

        # Upload to conversation A
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Secret document only for conversation A.")
            f.flush()
            manager.process_file(Path(f.name), "secret.txt", conversation_id="conv-A")

        # Search in conversation B should NOT find it
        results = manager.search("Secret document", conversation_id="conv-B")

        assert len(results) == 0

    def test_search_empty_when_no_uploads(self):
        """Search should return empty when no files uploaded for thread."""
        from pravah.uploads import FileUploadManager

        manager = FileUploadManager()
        results = manager.search("anything", conversation_id="empty-conversation")

        assert len(results) == 0

    def test_search_pagination_with_offset(self):
        """Search should support offset for pagination through results."""
        from pravah.uploads import FileUploadManager

        manager = FileUploadManager()
        conversation_id = "test-pagination-123"

        # Index enough content to have multiple chunks
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Python programming. " * 100)  # Long enough for multiple chunks
            f.flush()
            manager.process_file(Path(f.name), "large.txt", conversation_id)

        # Get first page
        page1 = manager.search("Python", conversation_id=conversation_id, top_k=3, offset=0)
        # Get second page
        page2 = manager.search("Python", conversation_id=conversation_id, top_k=3, offset=3)

        # Pages should be different
        if len(page2) > 0:
            page1_ids = {r["chunk_id"] for r in page1}
            page2_ids = {r["chunk_id"] for r in page2}
            assert page1_ids.isdisjoint(page2_ids)

    def test_search_returns_chunk_ids(self):
        """Search results should include chunk_id for read_upload_chunk."""
        from pravah.uploads import FileUploadManager

        manager = FileUploadManager()
        conversation_id = "test-chunkid-123"

        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Test content for chunk ID verification.")
            f.flush()
            manager.process_file(Path(f.name), "test.txt", conversation_id)

        results = manager.search("chunk", conversation_id=conversation_id)

        assert len(results) > 0
        assert "chunk_id" in results[0]
        assert results[0]["chunk_id"] is not None


class TestReadUploadChunk:
    """Test the read_upload_chunk tool for full content retrieval."""

    def test_read_chunk_returns_full_content(self):
        """read_upload_chunk should return full chunk text."""
        from pravah.uploads import FileUploadManager

        manager = FileUploadManager()
        conversation_id = "test-read-123"

        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("This is detailed content that should be fully returned.")
            f.flush()
            manager.process_file(Path(f.name), "detail.txt", conversation_id)

        results = manager.search("detailed", conversation_id=conversation_id)
        chunk_id = results[0]["chunk_id"]

        full_chunk = manager.read_chunk(chunk_id)

        assert "content" in full_chunk
        assert len(full_chunk["content"]) > len(results[0].get("preview", ""))

    def test_read_chunk_includes_navigation(self):
        """read_upload_chunk should include prev/next chunk_ids for scrolling."""
        from pravah.uploads import FileUploadManager

        manager = FileUploadManager()
        conversation_id = "test-nav-123"

        # Create long content that will span multiple chunks
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Section one content. " * 50 + "\n\n")
            f.write("Section two content. " * 50 + "\n\n")
            f.write("Section three content. " * 50)
            f.flush()
            manager.process_file(Path(f.name), "long.txt", conversation_id)

        results = manager.search("Section two", conversation_id=conversation_id)
        if len(results) > 0:
            chunk_id = results[0]["chunk_id"]
            full_chunk = manager.read_chunk(chunk_id)

            # Should have navigation info
            assert "prev_chunk_id" in full_chunk or "next_chunk_id" in full_chunk

    def test_read_invalid_chunk_returns_error(self):
        """read_upload_chunk should handle invalid chunk_id gracefully."""
        from pravah.uploads import FileUploadManager

        manager = FileUploadManager()
        result = manager.read_chunk("nonexistent-chunk-id-xyz")

        assert result.get("error") is not None or result.get("content") is None


class TestUploadManagement:
    """Test upload lifecycle management."""

    def test_list_uploads_for_conversation(self):
        """Should list all uploaded files for a conversation."""
        from pravah.uploads import FileUploadManager

        manager = FileUploadManager()
        conversation_id = "test-list-123"

        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Some content")
            f.flush()
            manager.process_file(Path(f.name), "file1.txt", conversation_id)

        uploads = manager.list_uploads(conversation_id)

        assert len(uploads) >= 1
        assert any(u["filename"] == "file1.txt" for u in uploads)

    def test_delete_conversation_uploads(self):
        """Should be able to delete all uploads for a conversation."""
        from pravah.uploads import FileUploadManager

        manager = FileUploadManager()
        conversation_id = "test-delete-123"

        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Content to be deleted.")
            f.flush()
            manager.process_file(Path(f.name), "todelete.txt", conversation_id)

        manager.delete_conversation_uploads(conversation_id)

        uploads = manager.list_uploads(conversation_id)
        assert len(uploads) == 0

