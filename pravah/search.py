import asyncio
import io
import logging
import os
import random
import re
import warnings

import aiohttp
import pymupdf4llm
from bs4 import BeautifulSoup, MarkupResemblesLocatorWarning
from tavily import TavilyClient

from tenacity import retry, stop_after_attempt, wait_fixed

from pravah.exceptions import FetchError, SearchError

logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)

# Google GenAI imports for Gemini grounded search
try:
    from google import genai
    from google.genai.types import (
        GenerateContentConfig,
        GoogleSearch,
        Tool as GenAITool,
    )

    GOOGLE_GENAI_AVAILABLE = True
except ImportError:
    GOOGLE_GENAI_AVAILABLE = False

def search_query(query: str, api_key: str, num_results: int = 5) -> dict:
    tavily_client = TavilyClient(api_key=api_key)
    results = tavily_client.search(
        query, include_raw_content=False, max_results=num_results
    )
    return results


# User-agent pool for HTTP requests
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/54.0.2840.99 Safari/537.3",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/60.0.3112.113 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/64.0.3282.140 Safari/537.36 Edge/17.17134",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_6) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/11.1.2 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:61.0) Gecko/20100101 Firefox/61.0",
    "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:52.0) Gecko/20100101 Firefox/52.0",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 11_0 like Mac OS X) AppleWebKit/604.1.38 (KHTML, like Gecko) Version/11.0 Mobile/15A372 Safari/604.1",
    "Mozilla/5.0 (Linux; Android 8.0.0; SM-G960F Build/R16NW) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/62.0.3202.84 Mobile Safari/537.36",
    "Mozilla/5.0 (Windows NT 6.1; WOW64; Trident/7.0; AS; rv:11.0) like Gecko",
]

jina_api_key = os.getenv("JINA_API_KEY", "")
JINA_HEADERS = {
    "Authorization": "Bearer {}".format(jina_api_key),
    "X-Return-Format": "markdown",
}


@retry(
    stop=stop_after_attempt(3),
    wait=wait_fixed(0.1),
    retry_error_callback=lambda retry_state: "",
)
async def fetch_content(url: str) -> str:
    """Fetches the content from a given URL asynchronously.

    Args:
        url: The URL to fetch the content from.
    Returns:
        The content of the page, or an empty string if an error occurs.
    """
    try:
        headers = {"User-Agent": random.choice(USER_AGENTS)}
        connector = aiohttp.TCPConnector()
        async with aiohttp.ClientSession(connector=connector) as session:
            if "https://arxiv.org/abs/" in url:
                url = url.replace("abs", "pdf")
            async with session.get(
                url, headers=headers, timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                response.raise_for_status()
                if url.endswith(".pdf") or "https://arxiv.org/" in url:
                    # Handle PDF content
                    content = await response.read()
                    md_text = pymupdf4llm.to_markdown(io.BytesIO(content))
                    return md_text
                else:
                    return await response.text()
    except aiohttp.ClientError as e:
        logger.warning("Failed to fetch content from %s: %s", url, e)
        return ""
    except asyncio.TimeoutError:
        logger.warning("Timeout while fetching content from %s", url)
        return ""
    except UnicodeDecodeError as e:
        logger.warning("Decoding error for content from %s: %s", url, e)
        return ""


def parse_content(content: str, markdown: bool = False) -> str:
    """Parses the HTML content to extract text and convert it to markdown format if enabled.

    Args:
        content: The raw HTML content.
        markdown: A flag to indicate whether to convert to markdown format.
    Returns:
        The text content of the page, or markdown formatted text if enabled.
    """
    try:
        soup = BeautifulSoup(content, "html.parser")

        if markdown:
            # Handle headings
            for heading in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6"]):
                try:
                    heading.string = f"{'#' * int(heading.name[1:])} {heading.get_text(strip=True)}\n"
                except (AttributeError, ValueError) as e:
                    logger.debug("Skipping malformed heading element: %s", e)

            # Handle paragraphs
            for p in soup.find_all("p"):
                try:
                    p.string = f"{p.get_text(strip=True)}\n\n"
                except AttributeError as e:
                    logger.debug("Skipping malformed paragraph element: %s", e)

            # Handle links
            for a in soup.find_all("a"):
                try:
                    if "href" in a.attrs:
                        a.string = f"[{a.get_text(strip=True)}]({a['href']})"
                except (AttributeError, KeyError) as e:
                    logger.debug("Skipping malformed link element: %s", e)

            # Handle bold and italic text
            for strong in soup.find_all("strong"):
                try:
                    strong.string = f"**{strong.get_text(strip=True)}**"
                except AttributeError as e:
                    logger.debug("Skipping malformed bold element: %s", e)
            for em in soup.find_all("em"):
                try:
                    em.string = f"*{em.get_text(strip=True)}*"
                except AttributeError as e:
                    logger.debug("Skipping malformed italic element: %s", e)

            # Handle unordered lists
            for ul in soup.find_all("ul"):
                try:
                    for li in ul.find_all("li"):
                        li.string = f"- {li.get_text(strip=True)}\n"
                except AttributeError as e:
                    logger.debug("Skipping malformed unordered list: %s", e)

            # Handle ordered lists
            for ol in soup.find_all("ol"):
                try:
                    for i, li in enumerate(ol.find_all("li")):
                        li.string = f"{i + 1}. {li.get_text(strip=True)}\n"
                except AttributeError as e:
                    logger.debug("Skipping malformed ordered list: %s", e)

            # Handle code blocks
            for pre in soup.find_all("pre"):
                try:
                    pre.string = f"```\n{pre.get_text()}\n```"
                except AttributeError as e:
                    logger.debug("Skipping malformed code block: %s", e)

            # Handle images
            for img in soup.find_all("img"):
                try:
                    alt_text = img.get("alt", "")
                    img.replace_with(f"![{alt_text}]({img['src']})")
                except (AttributeError, KeyError) as e:
                    logger.debug("Skipping malformed image element: %s", e)

            # Remove empty tags
            for tag in soup.find_all():
                try:
                    if not tag.get_text(strip=True):
                        tag.decompose()
                except AttributeError as e:
                    logger.debug("Error removing empty tag: %s", e)

            # Get the final markdown text
            markdown_text = soup.get_text()

            # Clean up extra newlines
            markdown_text = re.sub(r"\n{3,}", "\n\n", markdown_text)

            return markdown_text

        else:
            # Default parsing to extract plain text
            try:
                text = " ".join([s.get_text(strip=True) for s in soup.find_all()])
                return text
            except Exception as e:
                logger.warning("Plain text extraction failed: %s", e)
                return ""

    except Exception as e:
        logger.warning("HTML parsing failed, trying fallbacks: %s", e)
        for fallback in [
            fallback_to_plain_text,
            fallback_to_partial_markdown,
            fallback_to_simplified_parsing,
        ]:
            try:
                markdown_text = fallback(content)
                logger.debug("Fallback successful using %s", fallback.__name__)
                return markdown_text
            except Exception as fallback_error:
                logger.debug(
                    "Fallback %s also failed: %s",
                    fallback.__name__,
                    fallback_error,
                )
        return ""


def fallback_to_plain_text(content):
    soup = BeautifulSoup(content, "html.parser")
    return soup.get_text()


def fallback_to_partial_markdown(content):
    soup = BeautifulSoup(content, "html.parser")
    markdown_text = ""
    for heading in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6"]):
        markdown_text += (
            f"{'#' * int(heading.name[1:])} {heading.get_text(strip=True)}\n"
        )
    for p in soup.find_all("p"):
        markdown_text += f"{p.get_text(strip=True)}\n\n"
    return markdown_text


def fallback_to_simplified_parsing(content):
    soup = BeautifulSoup(content, "html.parser")
    markdown_text = ""
    for element in soup.find_all():
        if element.name in ["h1", "h2", "h3", "h4", "h5", "h6", "p"]:
            markdown_text += f"{element.get_text(strip=True)}\n\n"
    return markdown_text


async def get_text_from_url(
    url: str, search_type: str = "default", markdown: bool = False
) -> str:
    """Fetches and parses the text content from a given URL asynchronously.

    Args:
        url: The URL to fetch the content from.
        search_type: The type of search to perform ('default' or 'jina').
        markdown: A flag to indicate whether to convert to markdown format.
    Returns:
        The text content of the page, or an empty string if an error occurs.
    """
    if search_type == "jina":
        headers = JINA_HEADERS  # Use the predefined JINA headers
        content = await fetch_jina_content(url, headers)
    else:
        content = await fetch_content(url)

    if content:
        return parse_content(content, markdown) if search_type == "default" else content
    return ""


async def fetch_jina_content(url: str, headers: dict) -> str:
    url = f"https://r.jina.ai/{url}"
    async with aiohttp.ClientSession() as session:
        async with session.get(url, headers=headers) as response:
            response.raise_for_status()
            return await response.text()


def search_query_gemini(query: str, api_key: str, num_results: int = 5) -> dict:
    """Search using Google Gemini with grounded search (Google Search tool).

    Uses the google-genai SDK to perform grounded searches via Gemini.
    Returns results in a format similar to Tavily for consistency.

    Args:
        query: The search query string.
        api_key: Google API key (GOOGLE_API_KEY).
        num_results: Maximum number of source URLs to return (default 5).

    Returns:
        Dict with 'answer' (synthesized response) and 'results' (source URLs).
        Returns empty results if google-genai is not installed.
    """
    if not GOOGLE_GENAI_AVAILABLE:
        return {
            "answer": "Error: google-genai package not installed. Run: pip install google-genai",
            "results": [],
        }

    try:
        # Configure the client - uses GOOGLE_API_KEY env var or passed key
        # Note: Don't specify api_version for compatibility with all models
        client = genai.Client(api_key=api_key)

        # Generate content with Google Search grounding
        # Using gemini-2.5-flash-lite for fast, cost-effective grounded search
        response = client.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents=query,
            config=GenerateContentConfig(
                tools=[GenAITool(google_search=GoogleSearch())],
            ),
        )

        # Extract the answer text
        answer = response.text if hasattr(response, "text") else ""

        # Extract grounding metadata (source URLs)
        results = []
        if hasattr(response, "candidates") and response.candidates:
            candidate = response.candidates[0]
            if (
                hasattr(candidate, "grounding_metadata")
                and candidate.grounding_metadata
            ):
                metadata = candidate.grounding_metadata

                # Extract grounding chunks (source URLs)
                if hasattr(metadata, "grounding_chunks") and metadata.grounding_chunks:
                    for chunk in metadata.grounding_chunks[:num_results]:
                        if hasattr(chunk, "web") and chunk.web:
                            results.append(
                                {
                                    "url": getattr(chunk.web, "uri", ""),
                                    "title": getattr(chunk.web, "title", "Untitled"),
                                    "content": "",  # Gemini doesn't provide snippets in chunks
                                }
                            )

                # Also check grounding_supports for additional context
                if (
                    hasattr(metadata, "grounding_supports")
                    and metadata.grounding_supports
                ):
                    for support in metadata.grounding_supports:
                        if hasattr(support, "grounding_chunk_indices"):
                            # This links text segments to sources
                            pass  # Already captured in grounding_chunks

                # Check search_entry_point for rendered search suggestions
                if (
                    hasattr(metadata, "search_entry_point")
                    and metadata.search_entry_point
                ):
                    # Contains rendered_content for Google Search Suggestions
                    pass  # Optional: could include search suggestion HTML

        return {"answer": answer, "results": results}

    except Exception as e:
        error_msg = str(e)
        # Provide more helpful error messages for common issues
        if "API key" in error_msg or "api_key" in error_msg.lower():
            error_msg = f"API key issue: {error_msg}. Ensure GOOGLE_API_KEY is valid."
        elif "quota" in error_msg.lower() or "rate" in error_msg.lower():
            error_msg = f"Rate limit or quota exceeded: {error_msg}"
        elif "permission" in error_msg.lower() or "403" in error_msg:
            error_msg = f"Permission denied: {error_msg}. Check API key permissions."
        return {"answer": f"Error performing Gemini search: {error_msg}", "results": []}
