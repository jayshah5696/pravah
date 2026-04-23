"""
Pravah v2 Tools - Agent-friendly tools for search and retrieval

Features:
- All tools are sync (LangGraph ToolNode handles async internally)
- Chunked reading for large pages
- Summarization for long content
- Proper error handling with informative messages
- Output limits to prevent context overflow
"""

import os
import asyncio
from typing import Optional
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig

from pravah.search import search_query, search_query_gemini, get_text_from_url
from pravah.llm import completion_llm


# ============================================================================
# Configuration
# ============================================================================

MAX_SEARCH_RESULTS = 5
MAX_PAGE_CHARS_DIRECT = 8000  # Return directly if under this limit
MAX_PAGE_CHARS_SUMMARIZE = 50000  # Summarize if under this limit
MAX_PAGE_CHARS_CHUNK = 100000  # Chunk and select if over summarize limit
CHUNK_SIZE = 4000  # Size of chunks for reading
SUMMARY_MODEL = "groq/llama-3.1-8b-instant"  # Fast model for summarization


# ============================================================================
# Helper Functions
# ============================================================================


def _run_async(coro):
    """Run an async coroutine synchronously."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If we're in an async context, create a new thread
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as pool:
                return pool.submit(asyncio.run, coro).result()
        else:
            return loop.run_until_complete(coro)
    except RuntimeError:
        return asyncio.run(coro)


def _summarize_content(content: str, query: str = "", max_length: int = 4000) -> str:
    """Summarize long content using a fast LLM.

    Args:
        content: The content to summarize
        query: Optional query to focus the summary on
        max_length: Maximum length of the summary

    Returns:
        Summarized content
    """
    if len(content) <= max_length:
        return content

    # Truncate content if too long for the summarizer
    truncated = content[:MAX_PAGE_CHARS_SUMMARIZE]

    focus = f" Focus on information relevant to: {query}" if query else ""

    prompt = f"""Summarize the following content concisely.{focus}
Keep the most important facts, data, and key points.
Include any URLs, citations, or references.
Maximum {max_length} characters.

Content:
{truncated}

Summary:"""

    try:
        summary = completion_llm(
            prompt, model=SUMMARY_MODEL, temperature=0.1, stream=False
        )
        return summary[:max_length]
    except Exception as e:
        # If summarization fails, return truncated content
        return (
            content[:max_length]
            + f"\n\n[Content truncated, {len(content)} total chars. Summarization failed: {e}]"
        )


def _chunk_content(content: str, chunk_size: int = CHUNK_SIZE) -> list[dict]:
    """Split content into chunks with metadata.

    Args:
        content: The content to chunk
        chunk_size: Size of each chunk

    Returns:
        List of dicts with 'index', 'content', 'start_char', 'end_char'
    """
    chunks = []
    for i in range(0, len(content), chunk_size):
        chunk = content[i : i + chunk_size]
        chunks.append(
            {
                "index": len(chunks),
                "content": chunk,
                "start_char": i,
                "end_char": min(i + chunk_size, len(content)),
            }
        )
    return chunks


# ============================================================================
# Tools
# ============================================================================


@tool
def web_search(query: str) -> str:
    """Search the web for current information.

    Use this tool when you need to find:
    - Current events or recent news
    - Up-to-date facts and data
    - Information that may have changed since your training
    - External documentation or resources

    Do NOT use this tool for:
    - Greetings or casual conversation
    - Basic facts you already know (capitals, math, etc.)
    - Questions about yourself

    Args:
        query: A specific, keyword-rich search query.
               Good: "Python 3.13 release features October 2024"
               Bad: "Python" (too vague)

    Returns:
        Top search results with titles, URLs, and snippets.
        Use the URLs with fetch_page if you need more detail.
    """
    try:
        api_key = os.getenv("TVLY_API_KEY")
        if not api_key:
            return "Error: TVLY_API_KEY not configured. I cannot search the web without this. Please ask the user to configure the Tavily API key."

        results = search_query(query, api_key=api_key, num_results=MAX_SEARCH_RESULTS)

        if not results:
            return f"Search returned no results for: '{query}'. Try rephrasing your query with different keywords."

        if "results" not in results or not results["results"]:
            return f"No web results found for: '{query}'. This might be a very niche topic. Try broader terms."

        # Format results
        formatted = []
        for i, r in enumerate(results["results"][:MAX_SEARCH_RESULTS], 1):
            title = r.get("title", "No Title")
            url = r.get("url", "")
            content = r.get("content", "")[:500]  # Snippet

            formatted.append(f"""**[{i}] {title}**
URL: {url}
{content}""")

        result_text = "\n\n".join(formatted)

        # Add usage hint
        result_text += "\n\n---\n*To read a full article, use fetch_page with the URL.*"

        return result_text

    except Exception as e:
        return f"Search failed with error: {str(e)}. Please try a different query or check your internet connection."


@tool
def gemini_search(query: str) -> str:
    """Search the web using Google Gemini with grounded search.

    Use this tool when you need:
    - AI-synthesized answers with source citations from Google Search
    - Quick answers that combine information from multiple sources
    - Responses grounded in Google's search index
    - Up-to-date information with source verification

    This differs from web_search (Tavily) in that:
    - Returns an AI-generated answer + source URLs (vs raw search results)
    - Uses Google's search index directly via Gemini
    - Provides synthesized, reasoned responses
    - Better for questions requiring information synthesis

    Use web_search (Tavily) instead when:
    - You need to fetch full page content afterward
    - You want raw search snippets without AI synthesis
    - You prefer more control over which sources to explore

    Args:
        query: A natural language question or search query.
               Good: "What are the latest features in Python 3.13?"
               Good: "Compare React vs Vue for building SPAs"

    Returns:
        AI-synthesized answer with source URLs for verification.
    """
    try:
        # Check for API key - support both GOOGLE_API_KEY and GEMINI_API_KEY
        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not api_key:
            return "Error: GOOGLE_API_KEY (or GEMINI_API_KEY) not configured. I cannot use Gemini search without this. Please use web_search instead, or ask the user to configure the Google API key."

        results = search_query_gemini(
            query, api_key=api_key, num_results=MAX_SEARCH_RESULTS
        )

        answer = results.get("answer", "")
        sources = results.get("results", [])

        # Check if the answer is an error message from the search function
        if answer.startswith("Error"):
            return f"Gemini search failed: {answer}. Try using web_search instead."

        if not answer:
            return f"Gemini search returned no answer for: '{query}'. Try rephrasing your query or use web_search instead."

        # Format output: Answer + Sources
        output = f"**Answer:**\n{answer}\n"

        if sources:
            output += "\n**Sources:**\n"
            for i, source in enumerate(sources[:MAX_SEARCH_RESULTS], 1):
                title = source.get("title", "Untitled")
                url = source.get("url", "")
                output += f"[{i}] {title}\n    URL: {url}\n"
        else:
            output += "\n*Note: No source URLs were provided for this response.*"

        output += "\n---\n*To read a full article, use fetch_page with a source URL.*"

        return output

    except Exception as e:
        return f"Gemini search failed with error: {str(e)}. Please try a different query or use web_search instead."


@tool
def fetch_page(url: str, query: str = "") -> str:
    """Fetch and read content from a URL.

    Use this tool when you need to:
    - Read the full content of an article or documentation
    - Get detailed information from a search result
    - Access specific data on a webpage

    The content is automatically:
    - Converted to markdown for readability
    - Summarized if very long (keeping key facts)
    - Chunked if extremely long (with navigation info)
    - Stored in memory for later search with search_memory

    Args:
        url: The URL to fetch (from web_search results)
        query: Optional - your original query to focus the summary

    Returns:
        The page content in markdown format.
        For very long pages, returns a summary with the option to read specific sections.
    """
    from pravah.memory import get_memory_store, get_current_thread_id

    try:
        # Fetch the content asynchronously
        content = _run_async(get_text_from_url(url, markdown=True))

        if not content:
            return f"Could not fetch content from {url}. The page may be:\n- Behind a paywall\n- Blocked by robots.txt\n- Temporarily unavailable\n\nTry a different source."

        if not content.strip():
            return f"The page at {url} appears to be empty or contains only non-text content (images, videos, etc.)."

        # Store in session memory for later search
        thread_id = get_current_thread_id()
        if thread_id:
            # Extract title from content (first line or URL)
            title = content.split("\n")[0][:100] if content else url
            title = title.strip("#").strip()  # Remove markdown headers
            get_memory_store().add_document(thread_id, url, content, title)

        # Handle based on content length
        content_len = len(content)

        if content_len <= MAX_PAGE_CHARS_DIRECT:
            # Short enough to return directly
            return f"**Content from {url}** ({content_len} chars):\n\n{content}"

        elif content_len <= MAX_PAGE_CHARS_SUMMARIZE:
            # Summarize the content
            summary = _summarize_content(
                content, query, max_length=MAX_PAGE_CHARS_DIRECT
            )
            return f"""**Summary of {url}** (original: {content_len} chars):

{summary}

---
*This is a summarized version. Key information has been preserved.*
*Full content stored in memory - use search_memory to find specific details.*"""

        else:
            # Very long content - chunk and summarize
            summary = _summarize_content(
                content, query, max_length=MAX_PAGE_CHARS_DIRECT // 2
            )
            chunks = _chunk_content(content)

            return f"""**Summary of {url}** (original: {content_len} chars, {len(chunks)} chunks):

{summary}

---
*This page is very long. The above is a summary of the key points.*
*Use read_page_chunk to read specific sections if needed.*
*Use search_memory to find specific information in this document.*
*Total chunks available: {len(chunks)} (each ~{CHUNK_SIZE} chars)*"""

    except Exception as e:
        return f"Failed to fetch {url}: {str(e)}. The site may be blocking automated access or experiencing issues."


@tool
def read_page_chunk(url: str, chunk_index: int = 0) -> str:
    """Read a specific chunk of a long webpage.

    Use this after fetch_page indicates a page is too long and was summarized.
    This lets you read specific sections of very long documents.

    Args:
        url: The URL to read from (same as used with fetch_page)
        chunk_index: Which chunk to read (0-based). Start with 0 for the beginning.

    Returns:
        The content of the specified chunk, with navigation info.
    """
    try:
        content = _run_async(get_text_from_url(url, markdown=True))

        if not content:
            return f"Could not fetch content from {url}."

        chunks = _chunk_content(content)
        total_chunks = len(chunks)

        if chunk_index < 0 or chunk_index >= total_chunks:
            return f"Invalid chunk index. This page has {total_chunks} chunks (0 to {total_chunks - 1})."

        chunk = chunks[chunk_index]

        nav_info = []
        if chunk_index > 0:
            nav_info.append(f"Previous: chunk {chunk_index - 1}")
        if chunk_index < total_chunks - 1:
            nav_info.append(f"Next: chunk {chunk_index + 1}")

        nav_str = " | ".join(nav_info) if nav_info else "This is the only chunk."

        return f"""**Chunk {chunk_index + 1} of {total_chunks}** from {url}
Characters {chunk["start_char"]} - {chunk["end_char"]}

{chunk["content"]}

---
*Navigation: {nav_str}*"""

    except Exception as e:
        return f"Failed to read chunk from {url}: {str(e)}"


@tool
def search_memory(query: str) -> str:
    """Search content already fetched in this conversation session.

    Use this tool to:
    - Find specific information in pages you've already read
    - Avoid re-fetching the same URLs
    - Search across multiple previously fetched documents

    This is faster than re-fetching and helps stay within context limits.

    Args:
        query: What to search for in previously fetched content

    Returns:
        Relevant excerpts from previously fetched content, with source URLs.
        Returns empty if no content has been fetched yet.
    """
    from pravah.memory import get_memory_store, get_current_thread_id

    thread_id = get_current_thread_id()
    if not thread_id:
        return "Memory search is not available: no active session context."

    store = get_memory_store()
    results = store.search(thread_id, query, top_k=5)

    if not results:
        # Check if we have any documents at all
        urls = store.get_all_urls(thread_id)
        if not urls:
            return f"No documents have been fetched yet in this session. Use web_search and fetch_page first, then search_memory can find information in those pages."
        else:
            return f"No matches found for '{query}' in {len(urls)} fetched document(s). Try different search terms or fetch more relevant pages."

    # Format results
    output_parts = [f"**Found {len(results)} match(es) for '{query}':**\n"]

    for i, result in enumerate(results, 1):
        output_parts.append(f"""**[{i}] {result["title"]}**
URL: {result["url"]}
Matched terms: {", ".join(result["matched_terms"])}
Snippet: {result["snippet"]}
""")

    output_parts.append("---")
    output_parts.append(
        "*Use fetch_page with a URL above to re-read the full content.*"
    )

    return "\n".join(output_parts)


# ============================================================================
# Additional Utility Tools
# ============================================================================


@tool
def calculate(expression: str) -> str:
    """Perform a mathematical calculation.

    Use for basic math operations when you need exact answers.

    Args:
        expression: A mathematical expression like "2 + 2" or "sqrt(16)"

    Returns:
        The calculated result
    """
    import math
    import ast
    import operator

    # Safe evaluation with only math functions
    allowed_names = {
        "abs": abs,
        "round": round,
        "min": min,
        "max": max,
        "sum": sum,
        "pow": pow,
        "sqrt": math.sqrt,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "log": math.log,
        "log10": math.log10,
        "exp": math.exp,
        "pi": math.pi,
        "e": math.e,
    }

    operators = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
    }

    def _eval(node):
        if isinstance(node, ast.Constant):
            return node.value
        elif isinstance(node, ast.BinOp):
            return operators[type(node.op)](_eval(node.left), _eval(node.right))
        elif isinstance(node, ast.UnaryOp):
            return operators[type(node.op)](_eval(node.operand))
        elif isinstance(node, ast.Call):
            func = _eval(node.func)
            return func(*[_eval(arg) for arg in node.args])
        elif isinstance(node, ast.Name):
            if node.id in allowed_names:
                return allowed_names[node.id]
            raise ValueError(f"Name '{node.id}' is not allowed")
        raise TypeError(f"Unsupported operation: {type(node).__name__}")

    try:
        # Remove any potentially dangerous characters
        safe_expr = expression.replace("^", "**")

        # Parse the expression
        tree = ast.parse(safe_expr, mode="eval")

        # Evaluate safely
        result = _eval(tree.body)
        return f"{expression} = {result}"
    except Exception as e:
        return f"Could not calculate '{expression}': {str(e)}"
