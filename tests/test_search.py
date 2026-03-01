"""Tests for pravah.search module (parsers and helpers, no network)."""

from pravah.search import parse_content


class TestParseContent:
    def test_parse_plain_text(self):
        html = "<html><body><p>Hello world</p><p>Second paragraph</p></body></html>"
        result = parse_content(html, markdown=False)
        assert "Hello world" in result
        assert "Second paragraph" in result

    def test_parse_markdown_headings(self):
        html = "<html><body><h1>Title</h1><p>Content here</p></body></html>"
        result = parse_content(html, markdown=True)
        assert "Title" in result
        assert "Content here" in result

    def test_parse_markdown_links(self):
        html = '<html><body><a href="https://example.com">Link text</a></body></html>'
        result = parse_content(html, markdown=True)
        assert "Link text" in result

    def test_parse_empty_content(self):
        result = parse_content("", markdown=False)
        assert result == "" or result.strip() == ""

    def test_parse_bold_text(self):
        html = "<html><body><strong>Bold text</strong></body></html>"
        result = parse_content(html, markdown=True)
        assert "Bold text" in result

    def test_parse_code_blocks(self):
        html = "<html><body><pre>print('hello')</pre></body></html>"
        result = parse_content(html, markdown=True)
        assert "print" in result

    def test_parse_lists(self):
        html = "<html><body><ul><li>Item 1</li><li>Item 2</li></ul></body></html>"
        result = parse_content(html, markdown=True)
        assert "Item 1" in result
        assert "Item 2" in result

    def test_parse_ordered_lists(self):
        html = "<html><body><ol><li>First</li><li>Second</li></ol></body></html>"
        result = parse_content(html, markdown=True)
        assert "First" in result
        assert "Second" in result

    def test_invalid_html_fallback(self):
        # Very malformed HTML should not crash
        result = parse_content("not html at all", markdown=False)
        assert isinstance(result, str)
