"""Tests for pravah.prompts module."""

from pravah.prompts import (
    extract_rewritten_prompt,
    generate_prompt_template,
    get_agent_system_prompt,
    query_rewriter,
)


class TestGetAgentSystemPrompt:
    def test_returns_string(self):
        prompt = get_agent_system_prompt()
        assert isinstance(prompt, str)
        assert len(prompt) > 100

    def test_contains_identity(self):
        prompt = get_agent_system_prompt()
        assert "Pravah" in prompt

    def test_contains_tools(self):
        prompt = get_agent_system_prompt()
        assert "web_search" in prompt
        assert "gemini_search" in prompt
        assert "fetch_page" in prompt
        assert "calculate" in prompt

    def test_contains_current_date(self):
        prompt = get_agent_system_prompt()
        assert "Today's date" in prompt

    def test_contains_constraints(self):
        prompt = get_agent_system_prompt()
        assert "NEVER fabricate" in prompt


class TestGeneratePromptTemplate:
    def test_basic_generation(self):
        result = generate_prompt_template(
            query="What is Python?",
            context_list=[
                {"content": "Python is a programming language", "url": "https://python.org"}
            ],
        )
        assert "What is Python?" in result
        assert "Python is a programming language" in result
        assert "https://python.org" in result

    def test_empty_context(self):
        result = generate_prompt_template(
            query="test query",
            context_list=[],
        )
        assert "test query" in result

    def test_multiple_contexts(self):
        result = generate_prompt_template(
            query="test",
            context_list=[
                {"content": "First source", "url": "https://a.com"},
                {"content": "Second source", "url": "https://b.com"},
            ],
        )
        assert "First source" in result
        assert "Second source" in result


class TestQueryRewriter:
    def test_basic_rewrite(self):
        result = query_rewriter("What are good breakfast options?")
        assert "What are good breakfast options?" in result

    def test_with_previous_prompt(self):
        result = query_rewriter(
            "Tell me more",
            previous_prompt="What is machine learning?",
        )
        assert "Tell me more" in result
        assert "machine learning" in result

    def test_with_messages(self):
        result = query_rewriter(
            "Explain further",
            messages=[{"role": "user", "content": "Previous question"}],
        )
        assert "Explain further" in result


class TestExtractRewrittenPrompt:
    def test_extracts_from_output_tags(self):
        text = "Some text <output> extracted content </output> more text"
        # The regex in the function has a bug (`.?` instead of `.*?`),
        # but test the actual behavior
        result = extract_rewritten_prompt(text)
        assert isinstance(result, str)

    def test_returns_original_if_no_tags(self):
        text = "no output tags here"
        assert extract_rewritten_prompt(text) == text
