import litellm
import os
from dotenv import load_dotenv

load_dotenv()

print("--- LiteLLM Model Cost ---")
try:
    # Check if model_cost is exposed
    if hasattr(litellm, "model_cost"):
        print(f"litellm.model_cost type: {type(litellm.model_cost)}")
        # Print a sample
        sample_key = next(iter(litellm.model_cost))
        print(f"Sample key: {sample_key}")
        print(f"Sample value: {litellm.model_cost[sample_key]}")
    else:
        print("litellm.model_cost not found")
except Exception as e:
    print(f"Error checking model_cost: {e}")

print("\n--- Model Listing (Simulation) ---")
# checking if litellm has a helper for listing models from providers
try:
    # litellm doesn't have a unified 'list_models()' that calls APIs,
    # but let's check if there are provider-specific helpers or if we need to use the SDKs.
    print("Checking for list_models helper...")
    # This is likely not available in litellm as a unified call, but good to verify
    # We will likely rely on openai.models.list(), etc.
    pass
except Exception as e:
    print(f"Error: {e}")
