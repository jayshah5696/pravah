from litellm import completion
from rich.pretty import pprint
import os
# Load environment variables from .env file
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_fixed
from pravah.prompts import generate_title_prompt

# Safely get the API key
# api_key = os.getenv('GROQ_API_KEY')
# if not api_key:
#     raise ValueError("GROQ_API_KEY not found in environment variables")

# print(api_key)

os.environ['TOKENIZERS_PARALLELISM'] = 'false'
@retry(stop=stop_after_attempt(3), wait=wait_fixed(1))
def completion_llm(text, model='groq/llama3-70b-8192', temperature=0.7, stream=False):
    messages = [{ "content": text, "role": "user" }]
    response = completion(model=model, messages=messages, temperature=temperature,stream=stream)
    if stream:
        return response
    return response.choices[0].message.content

@retry(stop=stop_after_attempt(3), wait=wait_fixed(1))
def chat_llm(input, model='groq/llama3-70b-8192', temperature=0.7):
    messages = [{ "content": input, "role": "user" }]
    response = completion(model=model, messages=messages, temperature=temperature)
    # Append response as a new message with role 'assistant'
    messages.append({ "content": response.choices[0].message.content, "role": "assistant" })
    return messages

@retry(stop=stop_after_attempt(3), wait=wait_fixed(1))
def chat_llm_v2_updated(messages, input, model='groq/llama3-70b-8192', system_prompt=None, temperature=0.7):
    # Check if system prompt already exists in the conversation
    if system_prompt and not any(msg.get("role") == "system" for msg in messages):
        messages.insert(0, {"content": system_prompt, "role": "system"})
    # Add new user message to the existing conversation
    messages.append({"content": input, "role": "user"})
    response = completion(model=model, messages=messages, temperature=temperature)
    # Append response as a new message with role 'assistant'
    print(response.choices[0].message.content)
    messages.append({"content": response.choices[0].message.content, "role": "assistant"})
    return messages

@retry(stop=stop_after_attempt(3), wait=wait_fixed(1))
def generate_chat_title(prompt, response, model='groq/llama-3.1-8b-instant', temperature=0.5):
    """
    Generates a chat title using an LLM.
    """
    title_prompt = generate_title_prompt(prompt, response)
    # Ensure completion_llm is called synchronously
    try:
        title = completion_llm(title_prompt, model=model, temperature=temperature, stream=False)
    except Exception as e:
        print(f"Error generating title: {e}")
        return prompt[:50] # Fallback to truncated prompt

    # Cleanup title (remove quotes, newlines)
    if isinstance(title, str):
        return title.strip().replace('"', '').replace("'", "")
    return title # Should be string