"""
File Upload Manager with markitdown and LanceDB integration.

Features:
- Parse PDF, DOCX, PPTX, TXT, MD, CSV using markitdown
- Chunk content and index in LanceDB
- Conversation-scoped search with pagination
- Two-tool pattern: search_uploads + read_upload_chunk
"""

from pathlib import Path
from typing import Optional
from dataclasses import dataclass
from datetime import datetime
import csv

import lancedb


# Supported file extensions
SUPPORTED_EXTENSIONS = {
    ".txt",
    ".md",
    ".csv",
    ".json",
    ".html",
    ".pdf",
    ".docx",
    ".pptx",
    ".xlsx",
}

# Chunk configuration
CHUNK_SIZE = 1000  # characters per chunk
CHUNK_OVERLAP = 100  # overlap between chunks


class UnsupportedFileTypeError(Exception):
    """Raised when file type is not supported."""

    pass


@dataclass
class UploadChunk:
    """A chunk of uploaded content."""

    chunk_id: str
    conversation_id: str
    filename: str
    content: str
    chunk_index: int
    total_chunks: int
    created_at: datetime


class FileUploadManager:
    """Manages file uploads with LanceDB storage."""

    def __init__(self, db_path: str = ".uploads_db"):
        """Initialize the upload manager.

        Args:
            db_path: Path to LanceDB database
        """
        self.db_path = db_path
        self._db = None
        self._table = None

    @property
    def db(self):
        """Lazy initialization of LanceDB connection."""
        if self._db is None:
            self._db = lancedb.connect(self.db_path)
        return self._db

    def _get_or_create_table(self):
        """Get or create the uploads table."""
        if self._table is not None:
            return self._table

        table_name = "uploads"
        if table_name in self.db.list_tables():
            self._table = self.db.open_table(table_name)
        else:
            # Create table with simple schema (FTS only, no vectors for simplicity)
            import pyarrow as pa

            schema = pa.schema([
                pa.field("content", pa.utf8()),
                pa.field("chunk_id", pa.utf8()),
                pa.field("conversation_id", pa.utf8()),
                pa.field("filename", pa.utf8()),
                pa.field("chunk_index", pa.int32()),
                pa.field("total_chunks", pa.int32()),
                pa.field("created_at", pa.utf8()),
            ])
            self._table = self.db.create_table(table_name, schema=schema, exist_ok=True)
            self._table.create_fts_index("content", replace=True)

        return self._table

    def process_file(
        self,
        file_path: Path,
        filename: str,
        conversation_id: str,
    ) -> dict:
        """Process and index a file.

        Args:
            file_path: Path to the file
            filename: Original filename
            conversation_id: ID of the conversation

        Returns:
            Dict with 'success', 'chunk_count', 'filename'

        Raises:
            UnsupportedFileTypeError: If file type not supported
        """
        # Check extension
        ext = Path(filename).suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            raise UnsupportedFileTypeError(
                f"File type {ext} not supported. Supported: {', '.join(SUPPORTED_EXTENSIONS)}"
            )

        content = self._extract_content(file_path, ext)

        if not content or not content.strip():
            return {"success": False, "chunk_count": 0, "filename": filename}

        # Chunk the content
        chunks = self._chunk_content(content)
        total_chunks = len(chunks)

        # Prepare records for LanceDB
        records = []
        now = datetime.now().isoformat()

        for i, chunk_text in enumerate(chunks):
            chunk_id = f"{conversation_id}:{filename}:{i}"
            records.append(
                {
                    "content": chunk_text,
                    "chunk_id": chunk_id,
                    "conversation_id": conversation_id,
                    "filename": filename,
                    "chunk_index": i,
                    "total_chunks": total_chunks,
                    "created_at": now,
                }
            )

        # Add to table
        table = self._get_or_create_table()
        table.add(records)

        return {"success": True, "chunk_count": total_chunks, "filename": filename}

    def _extract_content(self, file_path: Path, ext: str) -> str:
        """Extract text from a file.

        Uses lightweight built-in parsing for simple text formats and
        falls back to MarkItDown for richer document types.
        """
        if ext in {".txt", ".md", ".json", ".html"}:
            return file_path.read_text(encoding="utf-8", errors="ignore")

        if ext == ".csv":
            rows: list[str] = []
            with file_path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
                reader = csv.reader(f)
                for row in reader:
                    rows.append(", ".join(cell.strip() for cell in row))
            return "\n".join(rows)

        from markitdown import MarkItDown

        result = MarkItDown().convert(str(file_path))
        return result.text_content or ""

    def _chunk_content(self, content: str) -> list[str]:
        """Split content into overlapping chunks.

        Args:
            content: Full text content

        Returns:
            List of chunk strings
        """
        chunks = []
        start = 0

        while start < len(content):
            end = min(start + CHUNK_SIZE, len(content))
            chunk = content[start:end]
            chunks.append(chunk)
            start = end - CHUNK_OVERLAP if end < len(content) else end

        return chunks if chunks else [content]

    def search(
        self,
        query: str,
        conversation_id: str,
        top_k: int = 5,
        offset: int = 0,
    ) -> list[dict]:
        """Search uploaded content.

        Args:
            query: Search query
            conversation_id: Scope to this conversation
            top_k: Number of results per page
            offset: Skip this many results (pagination)

        Returns:
            List of dicts with 'chunk_id', 'preview', 'filename', 'relevance'
        """
        table = self._get_or_create_table()

        try:
            # FTS search with conversation filter
            results = (
                table.search(query, query_type="fts")
                .where(f"conversation_id = '{conversation_id}'")
                .limit(max(top_k + offset, top_k) * 3)
                .to_list()
            )

            # Deduplicate by chunk_id before pagination because FTS backends may
            # return repeated rows for highly repetitive content.
            deduped_results = []
            seen_chunk_ids = set()
            for result in results:
                chunk_id = result.get("chunk_id")
                if chunk_id in seen_chunk_ids:
                    continue
                seen_chunk_ids.add(chunk_id)
                deduped_results.append(result)

            # Apply offset
            results = deduped_results[offset : offset + top_k]

            # Format results with previews
            formatted = []
            for r in results:
                preview = (
                    r["content"][:200] + "..."
                    if len(r["content"]) > 200
                    else r["content"][:50]
                )
                formatted.append(
                    {
                        "chunk_id": r["chunk_id"],
                        "content": r["content"],
                        "preview": preview,
                        "filename": r["filename"],
                        "chunk_index": r["chunk_index"],
                        "total_chunks": r["total_chunks"],
                    }
                )

            return formatted
        except Exception:
            return []

    def read_chunk(self, chunk_id: str) -> dict:
        """Read full content of a specific chunk.

        Args:
            chunk_id: The chunk ID to read

        Returns:
            Dict with 'content', 'prev_chunk_id', 'next_chunk_id', or 'error'
        """
        table = self._get_or_create_table()

        try:
            # Find the chunk
            results = table.search().where(f"chunk_id = '{chunk_id}'").limit(1).to_list()

            if not results:
                return {"error": f"Chunk not found: {chunk_id}"}

            chunk = results[0]
            conversation_id = chunk["conversation_id"]
            filename = chunk["filename"]
            chunk_index = chunk["chunk_index"]
            total_chunks = chunk["total_chunks"]

            # Build navigation
            prev_chunk_id = None
            next_chunk_id = None

            if chunk_index > 0:
                prev_chunk_id = f"{conversation_id}:{filename}:{chunk_index - 1}"
            if chunk_index < total_chunks - 1:
                next_chunk_id = f"{conversation_id}:{filename}:{chunk_index + 1}"

            return {
                "content": chunk["content"],
                "filename": filename,
                "chunk_index": chunk_index,
                "total_chunks": total_chunks,
                "prev_chunk_id": prev_chunk_id,
                "next_chunk_id": next_chunk_id,
            }
        except Exception as e:
            return {"error": str(e)}

    def list_uploads(self, conversation_id: str) -> list[dict]:
        """List all uploads for a conversation.

        Args:
            conversation_id: The conversation ID

        Returns:
            List of dicts with 'filename', 'chunk_count', 'created_at'
        """
        table = self._get_or_create_table()

        try:
            results = (
                table.search()
                .where(f"conversation_id = '{conversation_id}'")
                .limit(1000)
                .to_list()
            )

            # Group by filename
            files = {}
            for r in results:
                filename = r["filename"]
                if filename not in files:
                    files[filename] = {
                        "filename": filename,
                        "chunk_count": r["total_chunks"],
                        "created_at": r["created_at"],
                    }

            return list(files.values())
        except Exception:
            return []

    def delete_conversation_uploads(self, conversation_id: str) -> bool:
        """Delete all uploads for a conversation.

        Args:
            conversation_id: The conversation ID

        Returns:
            True if successful
        """
        table = self._get_or_create_table()

        try:
            table.delete(f"conversation_id = '{conversation_id}'")
            return True
        except Exception:
            return False


# Singleton instance
_upload_manager: Optional[FileUploadManager] = None


def get_upload_manager() -> FileUploadManager:
    """Get the global upload manager instance."""
    global _upload_manager
    if _upload_manager is None:
        _upload_manager = FileUploadManager()
    return _upload_manager
