
from unittest.mock import MagicMock, patch
import pytest

from pravah.llm import generate_chat_title


def test_generate_chat_title():
    """Basic title generation returns the LLM output."""
    with patch('pravah.llm.completion_llm') as mock_llm:
        mock_llm.return_value = "My Title"

        title = generate_chat_title("Hi", "Hello")

        assert title == "My Title"
        mock_llm.assert_called_once()


def test_generate_chat_title_strips_quotes():
    """Quotes and whitespace are cleaned from the LLM output."""
    with patch('pravah.llm.completion_llm') as mock_llm:
        mock_llm.return_value = '  "A Great Title"  '

        title = generate_chat_title("Hi", "Hello")

        assert title == "A Great Title"


def test_generate_chat_title_fallback_on_error():
    """Falls back to truncated prompt when LLM raises an exception."""
    with patch('pravah.llm.completion_llm') as mock_llm:
        mock_llm.side_effect = Exception("API error")

        title = generate_chat_title("What is the capital of France?", "Paris")

        assert title == "What is the capital of France?"[:50]


def test_generate_chat_title_fallback_on_empty_string():
    """Falls back to truncated prompt when LLM returns empty string."""
    with patch('pravah.llm.completion_llm') as mock_llm:
        mock_llm.return_value = "   "

        title = generate_chat_title("My long prompt here", "Response")

        assert title == "My long prompt here"[:50]


def test_generate_chat_title_truncates_long_title():
    """Titles longer than 100 characters are truncated."""
    with patch('pravah.llm.completion_llm') as mock_llm:
        mock_llm.return_value = "A" * 150

        title = generate_chat_title("Hi", "Hello")

        assert len(title) == 100


def test_generate_chat_title_truncates_response_input():
    """Only the first 500 chars of response are passed to the title prompt."""
    long_response = "x" * 1000
    with patch('pravah.llm.completion_llm') as mock_llm:
        mock_llm.return_value = "Short Title"
        with patch('pravah.llm.generate_title_prompt') as mock_prompt:
            mock_prompt.return_value = "fake prompt"

            generate_chat_title("Hi", long_response)

            # Verify response was truncated to 500 chars
            call_args = mock_prompt.call_args
            assert len(call_args[0][1]) == 500


def test_generate_chat_title_non_string_fallback():
    """Falls back to truncated prompt when LLM returns non-string."""
    with patch('pravah.llm.completion_llm') as mock_llm:
        mock_llm.return_value = 42  # non-string

        title = generate_chat_title("My prompt", "Response")

        assert title == "My prompt"[:50]


def test_generate_chat_title_strips_xml_title_tags():
    """XML <title> and </title> tags are stripped from LLM output."""
    with patch('pravah.llm.completion_llm') as mock_llm:
        mock_llm.return_value = "Great Chat Title</title>"

        title = generate_chat_title("Hi", "Hello")

        assert title == "Great Chat Title"


def test_generate_chat_title_strips_opening_title_tag():
    """Opening <title> tag is also stripped from LLM output."""
    with patch('pravah.llm.completion_llm') as mock_llm:
        mock_llm.return_value = "<title>Great Chat Title</title>"

        title = generate_chat_title("Hi", "Hello")

        assert title == "Great Chat Title"
