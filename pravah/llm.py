"""LLM completion wrapper using LiteLLM with retry logic."""

import os

import litellm
from litellm import completion
from tenacity import retry, stop_after_attempt, wait_fixed

# Configure Vertex AI from environment if available
litellm.vertex_project = os.getenv("VERTEX_PROJECT", "")
litellm.vertex_location = os.getenv("VERTEX_LOCATION", "us-east5")

os.environ["TOKENIZERS_PARALLELISM"] = "false"


@retry(stop=stop_after_attempt(3), wait=wait_fixed(1))
def completion_llm(
    text: str,
    model: str = "groq/llama3-70b-8192",
    temperature: float = 0.7,
    stream: bool = False,
) -> str:
    """Send a single-turn completion request via LiteLLM.

    Args:
        text: The prompt text to send.
        model: LiteLLM model identifier.
        temperature: Sampling temperature.
        stream: If True, returns a streaming response iterator.

    Returns:
        The response content string, or a streaming iterator if stream=True.
    """
    messages = [{"content": text, "role": "user"}]
    response = completion(
        model=model, messages=messages, temperature=temperature, stream=stream
    )
    if stream:
        return response
    return response.choices[0].message.content
