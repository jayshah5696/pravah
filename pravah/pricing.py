"""
Model Pricing for Pravah v2

Provides cost estimation based on token usage.
Prices are in USD per 1M tokens (input/output).

Updated: January 2026
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelPricing:
    """Pricing for a single model."""

    input_per_million: float  # USD per 1M input tokens
    output_per_million: float  # USD per 1M output tokens


# Model pricing data (USD per 1M tokens)
# Format: "provider/model" -> ModelPricing(input, output)
MODEL_PRICING: dict[str, ModelPricing] = {
    # OpenAI Models
    "openai/gpt-4o": ModelPricing(2.50, 10.00),
    "openai/gpt-4o-mini": ModelPricing(0.15, 0.60),
    "openai/gpt-4-turbo": ModelPricing(10.00, 30.00),
    "openai/gpt-4": ModelPricing(30.00, 60.00),
    "openai/gpt-3.5-turbo": ModelPricing(0.50, 1.50),
    "openai/o1": ModelPricing(15.00, 60.00),
    "openai/o1-mini": ModelPricing(3.00, 12.00),
    "openai/o1-preview": ModelPricing(15.00, 60.00),
    "openai/o3-mini": ModelPricing(1.10, 4.40),
    # Anthropic Models
    "anthropic/claude-3-5-sonnet-20241022": ModelPricing(3.00, 15.00),
    "anthropic/claude-3-5-haiku-20241022": ModelPricing(0.80, 4.00),
    "anthropic/claude-3-opus-20240229": ModelPricing(15.00, 75.00),
    "anthropic/claude-3-sonnet-20240229": ModelPricing(3.00, 15.00),
    "anthropic/claude-3-haiku-20240307": ModelPricing(0.25, 1.25),
    # Aliases
    "anthropic/claude-3.5-sonnet": ModelPricing(3.00, 15.00),
    "anthropic/claude-3.5-haiku": ModelPricing(0.80, 4.00),
    "anthropic/claude-sonnet-4-20250514": ModelPricing(3.00, 15.00),
    # Google/Gemini Models
    "gemini/gemini-2.0-flash": ModelPricing(0.10, 0.40),
    "gemini/gemini-2.0-flash-exp": ModelPricing(0.10, 0.40),
    "gemini/gemini-1.5-pro": ModelPricing(1.25, 5.00),
    "gemini/gemini-1.5-flash": ModelPricing(0.075, 0.30),
    "gemini/gemini-pro": ModelPricing(0.50, 1.50),
    "google/gemini-2.0-flash": ModelPricing(0.10, 0.40),
    # Groq Models (very cheap/free tier)
    "groq/llama-3.3-70b-versatile": ModelPricing(0.59, 0.79),
    "groq/llama-3.1-70b-versatile": ModelPricing(0.59, 0.79),
    "groq/llama-3.1-8b-instant": ModelPricing(0.05, 0.08),
    "groq/llama3-70b-8192": ModelPricing(0.59, 0.79),
    "groq/llama3-8b-8192": ModelPricing(0.05, 0.08),
    "groq/mixtral-8x7b-32768": ModelPricing(0.24, 0.24),
    "groq/gemma2-9b-it": ModelPricing(0.20, 0.20),
    # DeepSeek Models
    "deepseek/deepseek-chat": ModelPricing(0.14, 0.28),
    "deepseek/deepseek-reasoner": ModelPricing(0.55, 2.19),
    # Together AI Models
    "together/meta-llama/Llama-3-70b-chat-hf": ModelPricing(0.90, 0.90),
    "together/meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo": ModelPricing(0.88, 0.88),
    "together/mistralai/Mixtral-8x7B-Instruct-v0.1": ModelPricing(0.60, 0.60),
    # Fireworks AI
    "fireworks_ai/accounts/fireworks/models/llama-v3p1-70b-instruct": ModelPricing(
        0.90, 0.90
    ),
    # Cohere Models
    "cohere/command-r-plus": ModelPricing(2.50, 10.00),
    "cohere/command-r": ModelPricing(0.50, 1.50),
    # Mistral Models
    "mistral/mistral-large-latest": ModelPricing(2.00, 6.00),
    "mistral/mistral-medium-latest": ModelPricing(2.70, 8.10),
    "mistral/mistral-small-latest": ModelPricing(0.20, 0.60),
    "mistral/open-mistral-7b": ModelPricing(0.25, 0.25),
    "mistral/open-mixtral-8x7b": ModelPricing(0.70, 0.70),
}


def get_model_pricing(model: str) -> Optional[ModelPricing]:
    """Get pricing for a model.

    Args:
        model: The model string (e.g., "openai/gpt-4o", "anthropic/claude-3.5-sonnet")

    Returns:
        ModelPricing if found, None otherwise.
    """
    # Direct match
    if model in MODEL_PRICING:
        return MODEL_PRICING[model]

    # Try without version suffixes
    model_base = model.split("-")[0] if "-" in model else model
    for key, pricing in MODEL_PRICING.items():
        if key.startswith(model_base):
            return pricing

    # Try partial match on model name
    model_lower = model.lower()
    for key, pricing in MODEL_PRICING.items():
        if model_lower in key.lower() or key.lower() in model_lower:
            return pricing

    return None


def calculate_cost(
    model: str,
    tokens_in: int,
    tokens_out: int,
) -> Optional[float]:
    """Calculate the cost of a request.

    Args:
        model: The model string
        tokens_in: Number of input tokens
        tokens_out: Number of output tokens

    Returns:
        Cost in USD, or None if pricing not available.
    """
    pricing = get_model_pricing(model)
    if not pricing:
        return None

    input_cost = (tokens_in / 1_000_000) * pricing.input_per_million
    output_cost = (tokens_out / 1_000_000) * pricing.output_per_million

    return input_cost + output_cost


def format_cost(cost: Optional[float]) -> str:
    """Format cost for display.

    Args:
        cost: Cost in USD

    Returns:
        Formatted string (e.g., "$0.0023", "N/A")
    """
    if cost is None:
        return "N/A"

    if cost < 0.0001:
        return "<$0.0001"
    elif cost < 0.01:
        return f"${cost:.4f}"
    elif cost < 1.00:
        return f"${cost:.3f}"
    else:
        return f"${cost:.2f}"


def get_model_cost_tier(model: str) -> str:
    """Get a cost tier label for a model.

    Args:
        model: The model string

    Returns:
        Cost tier: "Free", "$", "$$", "$$$", "$$$$", or "Unknown"
    """
    pricing = get_model_pricing(model)
    if not pricing:
        return "Unknown"

    # Use output price as primary indicator (usually higher)
    output_price = pricing.output_per_million

    if output_price < 0.10:
        return "Free"
    elif output_price < 1.00:
        return "$"
    elif output_price < 5.00:
        return "$$"
    elif output_price < 20.00:
        return "$$$"
    else:
        return "$$$$"
