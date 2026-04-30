"""
Dynamic Welcome Panel with LLM-powered personalization.

Uses Pydantic structured output for consistent welcome content.
Falls back to time-based greeting if LLM fails.
"""

import os
import random
from datetime import datetime
from pydantic import BaseModel, Field

# Model for welcome generation (fast & cheap)
WELCOME_MODEL = "groq/llama-3.1-8b-instant"

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


class WelcomeContent(BaseModel):
    """Content for the dynamic welcome panel."""

    greeting: str = Field(
        ...,
        description="A warm, personalized greeting based on time of day",
    )
    tip: str = Field(
        ...,
        description="A helpful tip about using the search engine",
    )


# Fallback tips for when LLM is unavailable
FALLBACK_TIPS = [
    "Try asking about recent news for up-to-date information.",
    "You can upload documents and I'll search within them.",
    "Use specific keywords for more accurate search results.",
    "I can compare multiple topics side-by-side.",
    "Ask follow-up questions to dive deeper into any topic.",
    "I cite all my sources so you can verify information.",
]


def get_time_based_greeting() -> str:
    """Get a greeting based on current time of day.

    Returns:
        Appropriate greeting string
    """
    hour = datetime.now().hour

    if 5 <= hour < 12:
        return "Good morning"
    elif 12 <= hour < 17:
        return "Good afternoon"
    else:
        return "Good evening"


def get_api_status(required_keys: list[str]) -> dict:
    """Check which API keys are configured.

    Args:
        required_keys: List of environment variable names to check

    Returns:
        Dict with 'all_configured' bool and 'missing' list
    """
    missing = [key for key in required_keys if not os.getenv(key)]
    return {
        "all_configured": len(missing) == 0,
        "missing": missing,
    }


def generate_welcome(history_summary: str = "") -> WelcomeContent:
    """Generate personalized welcome content using LLM.

    Args:
        history_summary: Optional summary of user's past conversations

    Returns:
        WelcomeContent with greeting and tip
    """
    hour = datetime.now().hour
    time_context = (
        "morning" if 5 <= hour < 12 else "afternoon" if hour < 17 else "evening"
    )

    prompt = f"""Generate a warm welcome for an AI search engine user.
Time of day: {time_context}
{"Previous topics: " + history_summary if history_summary else "New user"}

Create a brief, friendly greeting and one helpful tip about using the search."""

    result = _get_client().chat.completions.create(
        model=WELCOME_MODEL,
        response_model=WelcomeContent,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=100,
    )
    return result


def generate_welcome_with_fallback(history_summary: str = "") -> WelcomeContent:
    """Generate welcome with fallback on any failure.

    Always returns valid content, never raises.
    """
    try:
        return generate_welcome(history_summary)
    except Exception:
        # Fallback to static content
        greeting = f"{get_time_based_greeting()}! Ready to explore?"
        tip = random.choice(FALLBACK_TIPS)
        return WelcomeContent(greeting=greeting, tip=tip)


def render_welcome_html(content: WelcomeContent, api_status: dict) -> str:
    """Render welcome content as HTML for Streamlit.

    Args:
        content: WelcomeContent with greeting and tip
        api_status: Dict from get_api_status()

    Returns:
        HTML string for st.markdown()
    """
    status_icon = "✅" if api_status["all_configured"] else "⚠️"
    status_text = (
        "All APIs configured"
        if api_status["all_configured"]
        else f"Missing: {', '.join(api_status['missing'])}"
    )

    return f"""
### {content.greeting}

An AI search engine that finds and synthesizes information from the web.

**💡 Tip:** {content.tip}

---
{status_icon} {status_text}
"""
