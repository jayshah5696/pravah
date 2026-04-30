import unittest
from unittest.mock import patch
from pravah.pricing import get_model_pricing, calculate_cost, format_cost, get_model_cost_tier, ModelPricing

class TestPricing(unittest.TestCase):
    @patch("litellm.model_cost", {})
    def test_get_model_pricing(self):
        # Exact match
        pricing = get_model_pricing("openai/gpt-4o")
        self.assertEqual(pricing, ModelPricing(2.5, 10.0))

        # Version suffix fallback
        pricing = get_model_pricing("openai/gpt-4o-2024-05-13")
        self.assertEqual(pricing, ModelPricing(2.5, 10.0))

        # Partial case-insensitive match
        pricing = get_model_pricing("GPT-4O")
        self.assertEqual(pricing, ModelPricing(2.5, 10.0))

        # Non-existent model
        pricing = get_model_pricing("nonexistent/model")
        self.assertIsNone(pricing)

    @patch("litellm.model_cost", {})
    def test_calculate_cost(self):
        # Known model
        cost = calculate_cost("openai/gpt-4o", 1000000, 1000000)
        self.assertEqual(cost, 12.5) # 2.5 + 10.0

        # Unknown model
        cost = calculate_cost("nonexistent/model", 1000, 1000)
        self.assertIsNone(cost)

    def test_format_cost(self):
        self.assertEqual(format_cost(None), "N/A")
        self.assertEqual(format_cost(0.00001), "<$0.0001")
        self.assertEqual(format_cost(0.005), "$0.0050")
        self.assertEqual(format_cost(0.5), "$0.500")
        self.assertEqual(format_cost(5.0), "$5.00")

    @patch("litellm.model_cost", {})
    def test_get_model_cost_tier(self):
        # Free (< 0.10)
        self.assertEqual(get_model_cost_tier("groq/llama-3.1-8b-instant"), "Free") # output 0.08

        # $ (0.10 <= output < 1.00)
        self.assertEqual(get_model_cost_tier("gemini/gemini-1.5-flash"), "$") # output 0.30

        # $$ (1.00 <= output < 5.00)
        self.assertEqual(get_model_cost_tier("openai/o3-mini"), "$$") # output 4.40

        # $$$ (5.00 <= output < 20.00)
        self.assertEqual(get_model_cost_tier("openai/gpt-4o"), "$$$") # output 10.00

        # $$$$ (>= 20.00)
        self.assertEqual(get_model_cost_tier("openai/gpt-4"), "$$$$") # output 60.00

        # Unknown
        self.assertEqual(get_model_cost_tier("nonexistent/model"), "Unknown")

if __name__ == "__main__":
    unittest.main()
