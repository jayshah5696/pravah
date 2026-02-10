
import pytest
from unittest.mock import MagicMock, patch
import os
import sys

# Set dummy environment variables to avoid import errors
os.environ['TVLY_API_KEY'] = 'test'
os.environ['OPENAI_API_KEY'] = 'test'
os.environ['COHERE_API_KEY'] = 'test'
os.environ['BRAVE_API_KEY'] = 'test'
os.environ['JINA_API_KEY'] = 'test'
os.environ['LANGCHAIN_API_KEY'] = 'test'
os.environ['LANGCHAIN_PROJECT'] = 'test'
os.environ['GROQ_API_KEY'] = 'test'

# We need to ensure app is imported with mocked streamlit and stays in sys.modules
# or we use patch.object.
# But since sys.modules['app'] is removed after the block, app.py might be re-executed if we import it again.
# To avoid re-execution issues and sys.modules manipulation issues, let's just mock streamlit in sys.modules manually without context manager for the import,
# or use a fixture that handles this.

# Simpler approach: verify config logic without importing app if possible? No, Config is in app.
# Let's use patch.dict but prevent it from clearing 'app'.
# Actually, if we use patch.dict, we should do it in a fixture or just manually modify sys.modules.

mock_streamlit = MagicMock()
sys.modules['streamlit'] = mock_streamlit
import app

def test_config_has_title_model():
    config = app.Config(search_tvly_api_key="test")
    assert hasattr(config, 'title_model')
    assert hasattr(config, 'title_model_temperature')
    assert config.title_model == 'groq/llama-3.1-8b-instant'

def test_check_api_keys_checks_title_model():
    config = app.Config(search_tvly_api_key="test")
    config.title_model = "openai/gpt-4"

    # We want to verify check_api_keys detects missing key.
    # But we set env vars above to import successfully.
    # So inside the test, we need to unset OPENAI_API_KEY temporarily.

    # We clear environ, so OPENAI_API_KEY is gone.
    with patch.dict(os.environ, {}, clear=True):
        # We mock streamlit inside app using patch.object
        with patch.object(app, 'st') as mock_st:
            mock_st.text_input.return_value = None

            app.check_api_keys(config)

            calls = [args[0] for args, _ in mock_st.info.call_args_list]
            # Since config.title_model is openai/..., it should check OPENAI_API_KEY
            assert any("OpenAI API key" in str(c) for c in calls)
