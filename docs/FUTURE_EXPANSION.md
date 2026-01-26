# Future Expansion & Improvement Roadmap

This document outlines planned improvements and technical specifications for the next phase of Pravah's development. These features focus on UX enhancements, dynamic configuration, and extended capabilities.

## 1. Dynamic Welcome Experience

**Goal:** Transform the static welcome message into a context-aware, dynamic dashboard.

### Features
- **Time-aware Greeting:** "Good Morning/Afternoon/Evening" based on user's local time.
- **Live Context:** Display current date (e.g., "Monday, Jan 26, 2026") to help ground the user and the agent.
- **System Pulse:** Status indicators for connected APIs (e.g., "🟢 OpenAI Ready", "🔴 Tavily Disconnected").
- **"Did You Know?"**: Random pro-tips or capability highlights rotated on each session load.

### Technical Implementation
```python
import datetime

def render_dynamic_welcome():
    now = datetime.datetime.now()
    hour = now.hour
    
    if 5 <= hour < 12:
        greeting = "Good Morning"
    elif 12 <= hour < 18:
        greeting = "Good Afternoon"
    else:
        greeting = "Good Evening"
        
    date_str = now.strftime("%A, %B %d, %Y")
    
    st.markdown(f"""
    # {greeting}, Human.
    ### It is {date_str}.
    
    System Status:
    - {'🟢' if os.getenv("OPENAI_API_KEY") else '🔴'} OpenAI
    - {'🟢' if os.getenv("TVLY_API_KEY") else '🔴'} Tavily Search
    """)
```

## 2. Dynamic Model Discovery

**Goal:** Eliminate hardcoded model lists by fetching available models directly from providers.

### Features
- **Auto-Discovery:** Query provider APIs (OpenAI `GET /models`, Anthropic, Gemini) to populate the model dropdown.
- **Capability Filtering:** Only show models that support `chat` or `function_calling`.
- **Smart Defaults:** If a configured model is deprecated or missing, fallback gracefully to a known stable model.
- **Settings Hiding:** Hide "Temperature" or "Top P" settings if the selected model doesn't support them (e.g., reasoning models like o1).

### Technical Implementation
- Use `litellm.model_list` where supported, or direct SDK calls.
- Cache the list for 24 hours to avoid API latency on every startup.
- **filtering logic:**
  ```python
  # Example for OpenAI
  models = client.models.list()
  chat_models = [m.id for m in models if "gpt" in m.id]
  ```

## 3. Automated Cost & Pricing

**Goal:** Use a single source of truth for model pricing instead of maintaining a manual dictionary.

### Features
- **Litellm Integration:** Leverage `litellm.model_cost` dictionary which is community-maintained and frequently updated.
- **Real-time Estimates:** Calculate cost per session based on actual token usage + live pricing.
- **Budget Alerts:** (Optional) Warn users if a session exceeds a certain dollar amount.

### Technical Implementation
```python
from litellm import model_cost

def get_real_cost(model_name, prompt_tokens, completion_tokens):
    if model_name in model_cost:
        input_price = model_cost[model_name]["input_cost_per_token"]
        output_price = model_cost[model_name]["output_cost_per_token"]
        return (prompt_tokens * input_price) + (completion_tokens * output_price)
    return None
```

## 4. Unified API Key & Connection Manager

**Goal:** Centralized validation for *all* required services, not just search.

### Features
- **Global Health Check:** On app startup, dry-run all configured keys.
- **Granular Feedback:** Instead of "API Error", show specific errors like "Invalid Permissions", "Quota Exceeded", or "Expired Key".
- **Provider-Specific Checks:** 
  - LLM: Simple "Hello" generation.
  - Search: Simple "Test" query.
  - Vector DB: Connection ping.

### Technical Implementation
- Create a `ConnectionManager` class.
- Run checks in parallel threads to not slow down startup.
- Store status in `st.session_state` to show/hide relevant features (e.g., disable "Search" tool if Tavily fails, but keep Chat active).

## 5. Smart Chat Titles (Auto-Slug)**

**Goal:** Replace verbatim first-query titles with concise, summarized topics.

### Features
- **Auto-Summarization:** After the first 2-3 turns, generate a 3-5 word title.
- **Async Generation:** Don't block the user; generate the title in the background or on the next refresh.
- **Editable Titles:** Allow users to manually rename chats (already supported in backend, need UI).

### Technical Implementation
- Trigger a "Title Generation" chain after the first assistant response.
- Use a small, cheap model (e.g., `gpt-4o-mini` or `llama-3-8b`) for this specific task.
- **Prompt:** "Summarize the following conversation start into a 3-5 word title. Do not use quotes."
- Update the `conversations` table in DuckDB with the new slug.

## 6. Document Uploads & RAG

**Goal:** Allow users to chat with their own data.

### Features
- **Multi-format Support:** PDF, CSV, TXT, MD.
- **Session-scoped RAG:** Uploads are temporary for the current chat session (stored in `pravah.memory`).
- **Hybrid Search:** Combine Web Search (Tavily) results with Local Document results.

### Technical Implementation
- **UI:** `st.file_uploader`.
- **Processing:** 
  - `pymupdf` for PDFs.
  - `pandas` for CSVs.
- **Storage:** 
  - Chunk text and store in the existing `MemoryStore` (from `pravah/memory.py`).
  - Use `search_memory` tool to retrieve from these docs.
- **Agent Update:** Provide a new tool `search_uploaded_docs` to the agent.

---
*Created: January 2026*
