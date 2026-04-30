"""
Smart Chat Title Generation with Pydantic Structured Output.

Uses instructor library for structured LLM responses to ensure
titles are 3-5 words without punctuation.
"""

from pydantic import BaseModel, Field

# Model to use for title generation (fast & cheap)
TITLE_MODEL = "groq/llama-3.1-8b-instant"

# Lazily initialized to keep imports lightweight in tests.
client = None


def _get_client():
    """Create the instructor client on first use."""
    global client
    if client is None:
        import instructor
        from litellm import completion

        client = instructor.from_litellm(completion)
    return client


class ChatTitle(BaseModel):
    """A short, descriptive title for a chat conversation."""

    title: str = Field(
        ...,
        min_length=5,
        max_length=40,
        description="3-5 word title summarizing the conversation topic, no punctuation",
    )


def generate_smart_title(query: str, response: str = "") -> str:
    """Generate a smart title using LLM with structured output.

    Args:
        query: The user's first message/query
        response: Optional first assistant response

    Returns:
        A 3-5 word title string
    """
    # Truncate inputs to save tokens
    query_preview = query[:200] if query else ""
    response_preview = response[:200] if response else ""

    prompt = f"""Create a concise 3-5 word title for this conversation.
No punctuation, no quotes, just a brief topic summary.

User: {query_preview}
{"Assistant: " + response_preview if response_preview else ""}

Title:"""

    try:
        result = _get_client().chat.completions.create(
            model=TITLE_MODEL,
            response_model=ChatTitle,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=20,
        )
        return result.title
    except Exception as e:
        # Fallback to truncation
        return _fallback_title(query)


def generate_smart_title_with_fallback(query: str, response: str = "") -> str:
    """Generate title with fallback on any failure.

    Always returns a valid title, never raises.
    """
    try:
        return generate_smart_title(query, response)
    except Exception:
        return _fallback_title(query)


def _fallback_title(query: str) -> str:
    """Create fallback title by truncating query.

    Args:
        query: Original user query

    Returns:
        First 5 words of query, max 50 chars
    """
    # Take first 5 words
    words = query.split()[:5]
    title = " ".join(words)

    # Clean punctuation
    for char in ["?", "!", ".", ",", '"', "'"]:
        title = title.replace(char, "")

    # Truncate if still too long
    if len(title) > 50:
        title = title[:47] + "..."

    return title.strip() or "New Chat"
