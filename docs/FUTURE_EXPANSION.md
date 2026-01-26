# Future expansion roadmap

This is a planning document. No code changes in this file.

## Roadmap todos

1. Dynamic welcome panel
2. Dynamic model discovery and settings gating
3. Pricing from litellm source of truth
4. Unified API key validation
5. Smart chat titles (slug, not raw query)
6. File upload and local RAG
7. Adaptive summarization tool for large tool outputs

## 1. Dynamic welcome panel

Goal: replace the static welcome copy with a time-aware panel.

Scope
- Time-based greeting and date stamp.
- Small status row for key services (LLM provider, search provider).
- Short rotating tips (1 line) pulled from a local list.

Notes
- Keep it light and factual. No marketing copy.
- Date should be local to the user session.

## 2. Dynamic model discovery and settings gating

Goal: avoid hardcoded model lists and hide controls that do not apply.

Scope
- Fetch model lists from provider APIs when available.
- Cache results for a day to avoid slow startup.
- Filter to chat-capable and tool-capable models.
- Hide or disable settings such as Temperature or Top P when the model ignores them.

Notes
- OpenAI: list models via SDK and filter by id patterns and metadata.
- Anthropic and Gemini: use their model list endpoints where available.
- Keep a safe fallback list if discovery fails.

## 3. Pricing from litellm source of truth

Goal: use litellm pricing data instead of a manual table.

Scope
- Read from litellm model pricing registry.
- Convert to per-request cost using input and output tokens.
- Show a single formatted cost in the debug panel.

Notes
- If litellm does not have a model entry, show "N/A" and do not guess.

## 4. Unified API key validation

Goal: validate all required keys in one place, not only search.

Scope
- Centralize required keys for each provider and tool.
- Run a lightweight validation call per provider.
- Surface clear status in the sidebar and the welcome panel.

Notes
- Validation should be fast and non-billing where possible.
- Do not block the UI on slow providers; show a pending state.

## 5. Smart chat titles (slug, not raw query)

Goal: replace raw user queries in the sidebar with short titles.

Scope
- Generate a 3-5 word title after the first assistant response.
- Use a small, low-cost model.
- Store the slug in DuckDB and allow manual edit later.

Notes
- Avoid quotes and punctuation in the title.
- Keep titles stable once set.

## 6. File upload and local RAG

Goal: let users attach local files and query them during the session.

Scope
- Add a file uploader in the sidebar.
- Parse PDF, TXT, MD, CSV.
- Chunk and index content per conversation.
- Provide a local search tool for retrieval.

Notes
- Keep uploaded data session-scoped by default.
- Provide a clear delete action for uploads.

## 7. Adaptive summarization tool for large tool outputs

Goal: when a tool response is too large, summarize with a smaller model and return only the needed parts.

Scope
- Add a tool wrapper for oversized outputs.
- Use a fast, low-cost summarizer model.
- Summarize to a target length and preserve citations.
- Provide a "view full output" path on demand.

Notes
- Apply this to fetch_page and any future heavy tools.
- Keep raw text in memory for follow-up questions.

## UI follow-up from this request

Question
- The chat history in the sidebar grows without a scrollbar. Should it scroll?

Answer
- Yes. Use a fixed-height sidebar container and let the history list scroll inside it. Keep the header, search box, and footer controls fixed.

---
Created: January 2026
