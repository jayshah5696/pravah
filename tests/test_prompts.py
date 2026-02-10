
import pytest
from pravah.prompts import generate_title_prompt

def test_generate_title_prompt():
    prompt = "What is the capital of France?"
    response = "The capital of France is Paris."

    title_prompt = generate_title_prompt(prompt, response)

    assert prompt in title_prompt
    assert response in title_prompt
    assert "title" in title_prompt.lower()
