"""Tests for Dynamic Welcome Panel functionality.

These tests verify that:
1. Welcome panel uses LLM for personalized greeting
2. Returns Pydantic WelcomeContent model
3. Falls back to static greeting on LLM failure
4. API status indicators work
"""

import pytest
from datetime import datetime
from unittest.mock import patch, MagicMock


class TestWelcomeContentSchema:
    """Test the Pydantic schema for welcome content."""

    def test_valid_welcome_content(self):
        """WelcomeContent should accept valid greeting and tip."""
        from pravah.welcome import WelcomeContent
        
        welcome = WelcomeContent(
            greeting="Good morning! Ready to explore?",
            tip="Try asking about recent news for up-to-date information."
        )
        assert "morning" in welcome.greeting.lower()
        assert len(welcome.tip) > 10


class TestLLMWelcomeGeneration:
    """Test LLM-powered welcome generation."""

    def test_generate_welcome_returns_pydantic_model(self):
        """Should return a WelcomeContent Pydantic model."""
        from pravah.welcome import generate_welcome, WelcomeContent
        
        with patch("pravah.welcome.client") as mock_client:
            mock_client.chat.completions.create.return_value = WelcomeContent(
                greeting="Good afternoon! What can I help you explore today?",
                tip="Use the search feature to find current information."
            )
            
            result = generate_welcome()
            
            assert hasattr(result, "greeting")
            assert hasattr(result, "tip")

    def test_fallback_on_llm_failure(self):
        """Should return static welcome if LLM fails."""
        from pravah.welcome import generate_welcome_with_fallback
        
        with patch("pravah.welcome.generate_welcome") as mock_gen:
            mock_gen.side_effect = Exception("LLM unavailable")
            
            result = generate_welcome_with_fallback()
            
            # Fallback should still return greeting and tip
            assert "greeting" in dir(result) or isinstance(result, dict)


class TestTimeAwareGreeting:
    """Test time-based greeting fallback."""

    def test_morning_greeting(self):
        """Should return 'Good morning' between 5am-12pm."""
        from pravah.welcome import get_time_based_greeting
        
        with patch("pravah.welcome.datetime") as mock_dt:
            mock_dt.now.return_value = datetime(2026, 1, 31, 9, 0, 0)
            greeting = get_time_based_greeting()
            assert "morning" in greeting.lower()

    def test_afternoon_greeting(self):
        """Should return 'Good afternoon' between 12pm-5pm."""
        from pravah.welcome import get_time_based_greeting
        
        with patch("pravah.welcome.datetime") as mock_dt:
            mock_dt.now.return_value = datetime(2026, 1, 31, 14, 0, 0)
            greeting = get_time_based_greeting()
            assert "afternoon" in greeting.lower()

    def test_evening_greeting(self):
        """Should return 'Good evening' after 5pm."""
        from pravah.welcome import get_time_based_greeting
        
        with patch("pravah.welcome.datetime") as mock_dt:
            mock_dt.now.return_value = datetime(2026, 1, 31, 19, 0, 0)
            greeting = get_time_based_greeting()
            assert "evening" in greeting.lower()


class TestApiStatusIndicator:
    """Test API status display."""

    def test_all_keys_present(self):
        """Should show success when all required keys present."""
        from pravah.welcome import get_api_status
        
        with patch.dict("os.environ", {
            "TVLY_API_KEY": "test",
            "OPENAI_API_KEY": "test"
        }):
            status = get_api_status(["TVLY_API_KEY", "OPENAI_API_KEY"])
            assert status["all_configured"] is True

    def test_missing_keys(self):
        """Should show warning when keys missing."""
        from pravah.welcome import get_api_status
        
        with patch.dict("os.environ", {}, clear=True):
            status = get_api_status(["TVLY_API_KEY"])
            assert status["all_configured"] is False
            assert "TVLY_API_KEY" in status["missing"]
