
from unittest.mock import MagicMock, patch
import pytest

# This import will fail until I implement the function
try:
    from pravah.llm import generate_chat_title
except ImportError:
    generate_chat_title = None

def test_generate_chat_title():
    if generate_chat_title is None:
        pytest.fail("generate_chat_title not implemented yet")

    prompt = "Hi"
    response = "Hello"

    # Mock completion_llm inside pravah.llm
    with patch('pravah.llm.completion_llm') as mock_llm:
        mock_llm.return_value = "My Title"

        title = generate_chat_title(prompt, response)

        assert title == "My Title"
        # validation of arguments passed to completion_llm could be added here
