"""Tests for pravah.pricing module."""

import pytest
from pravah.pricing import (
    ModelPricing,
    calculate_cost,
    format_cost,
    get_model_cost_tier,
    get_model_pricing,
)


class TestGetModelPricing:
    def test_direct_match(self):
        pricing = get_model_pricing("openai/gpt-4o")
        assert pricing is not None
        assert pricing.input_per_million == 2.50
        assert pricing.output_per_million == 10.00

    def test_unknown_model_returns_none(self):
        assert get_model_pricing("nonexistent/model-xyz") is None

    def test_groq_model(self):
        pricing = get_model_pricing("groq/llama-3.1-8b-instant")
        assert pricing is not None
        assert pricing.input_per_million == 0.05

    def test_anthropic_alias(self):
        pricing = get_model_pricing("anthropic/claude-3.5-sonnet")
        assert pricing is not None

    def test_gemini_model(self):
        pricing = get_model_pricing("gemini/gemini-2.0-flash")
        assert pricing is not None


class TestCalculateCost:
    def test_basic_calculation(self):
        cost = calculate_cost("openai/gpt-4o", tokens_in=1000, tokens_out=500)
        assert cost is not None
        expected = (1000 / 1_000_000) * 2.50 + (500 / 1_000_000) * 10.00
        assert abs(cost - expected) < 1e-10

    def test_zero_tokens(self):
        cost = calculate_cost("openai/gpt-4o", tokens_in=0, tokens_out=0)
        assert cost == 0.0

    def test_unknown_model_returns_none(self):
        assert calculate_cost("fake/model", tokens_in=100, tokens_out=100) is None

    def test_large_token_count(self):
        cost = calculate_cost("openai/gpt-4o-mini", tokens_in=1_000_000, tokens_out=1_000_000)
        assert cost is not None
        assert cost == pytest.approx(0.15 + 0.60)


class TestFormatCost:
    def test_none_returns_na(self):
        assert format_cost(None) == "N/A"

    def test_very_small_cost(self):
        assert format_cost(0.00001) == "<$0.0001"

    def test_small_cost(self):
        result = format_cost(0.005)
        assert result.startswith("$")
        assert "0.005" in result

    def test_medium_cost(self):
        result = format_cost(0.50)
        assert result == "$0.500"

    def test_large_cost(self):
        result = format_cost(1.50)
        assert result == "$1.50"


class TestGetModelCostTier:
    def test_cheap_model(self):
        tier = get_model_cost_tier("groq/llama-3.1-8b-instant")
        assert tier in ("Free", "$")

    def test_expensive_model(self):
        tier = get_model_cost_tier("anthropic/claude-3-opus-20240229")
        assert tier == "$$$$"

    def test_unknown_model(self):
        assert get_model_cost_tier("fake/model") == "Unknown"

    def test_mid_tier_model(self):
        tier = get_model_cost_tier("openai/gpt-4o")
        assert tier in ("$$", "$$$")
