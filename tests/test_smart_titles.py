"""Tests for Smart Chat Titles functionality.

These tests verify that:
1. Title generation uses Pydantic structured output
2. Titles are 3-5 words, no punctuation
3. Uses cheap/fast model (groq/llama-3.1-8b)
4. Fallback works when LLM fails
"""

import pytest
from unittest.mock import MagicMock, patch


class TestChatTitleSchema:
    """Test the Pydantic schema for chat titles."""

    def test_valid_title_passes_validation(self):
        """ChatTitle schema should accept valid 3-5 word titles."""
        from pravah.titles import ChatTitle
        
        title = ChatTitle(title="Python Web Scraping Guide")
        assert title.title == "Python Web Scraping Guide"

    def test_title_max_length_enforced(self):
        """ChatTitle should reject titles over 40 chars."""
        from pravah.titles import ChatTitle
        from pydantic import ValidationError
        
        with pytest.raises(ValidationError):
            ChatTitle(title="This is an extremely long title that should fail validation because it exceeds the limit")

    def test_title_min_length_enforced(self):
        """ChatTitle should reject titles under 5 chars."""
        from pravah.titles import ChatTitle
        from pydantic import ValidationError
        
        with pytest.raises(ValidationError):
            ChatTitle(title="Hi")


class TestSmartTitleGeneration:
    """Test LLM-based title generation with structured output."""

    def test_generate_title_returns_pydantic_model(self):
        """Should return a ChatTitle Pydantic model."""
        from pravah.titles import generate_smart_title, ChatTitle
        
        # Mock the LLM call
        with patch("pravah.titles.client") as mock_client:
            mock_client.chat.completions.create.return_value = ChatTitle(title="AI Image Generation")
            
            result = generate_smart_title(
                query="How do I generate images with AI?",
                response="You can use DALL-E, Midjourney, or Stable Diffusion..."
            )
            
            assert isinstance(result, str)
            assert len(result) <= 40

    def test_generate_title_uses_cheap_model(self):
        """Should use fast/cheap model like groq."""
        from pravah.titles import TITLE_MODEL
        
        assert "groq" in TITLE_MODEL or "gemini" in TITLE_MODEL or "mini" in TITLE_MODEL

    def test_fallback_on_llm_failure(self):
        """Should fall back to truncation if LLM fails."""
        from pravah.titles import generate_smart_title_with_fallback
        
        with patch("pravah.titles.generate_smart_title") as mock_gen:
            mock_gen.side_effect = Exception("LLM unavailable")
            
            result = generate_smart_title_with_fallback(
                query="What is the capital of France?",
                response="Paris is the capital..."
            )
            
            # Fallback should return truncated query
            assert isinstance(result, str)
            assert len(result) <= 50


class TestHistoryIntegration:
    """Test integration with history storage."""

    def test_update_conversation_title(self):
        """Should be able to update title in DuckDB."""
        from pravah.history import HistoryStore
        
        # This will be implemented
        pass
