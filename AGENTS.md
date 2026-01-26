# Repository Guidelines

## Project Structure & Module Organization
Root scripts `app.py` (Streamlit UI) and `main.py` (CLI workflow) orchestrate search and retrieval flows. Core modules live in `pravah/`—notably `search.py` for source gathering, `retrieval.py` for chunking and ranking, `llm.py` for model access, and `prompts.py` for Jinja templating. Static assets such as the demo GIF reside under `assets/`. Dependency manifests (`pyproject.toml`, `requirements.txt`) and the Dockerfile stay at the repo root; local data artifacts like `pravah.db` should remain git-ignored when regenerated.

## Build, Test, and Development Commands
This project uses **uv** as the package manager (not poetry).

- `uv sync` — install Python 3.11 dependencies into the managed virtualenv.
- `uv run streamlit run app.py` — launch the default web UI on port 8501.
- `uv run python main.py` — exercise the terminal agent loop against live services.
- `uv run python -c "..."` — run inline Python scripts with the project environment.
- `uv run pytest` — run tests (when available).
- `docker build -t pravah-local .` and `docker run -p 8501:8501 pravah-local` — verify containerized deployments match local behavior.

**Important:** Always use `uv run` prefix to ensure the correct virtual environment is used.

## Coding Style & Naming Conventions
Stick to idiomatic Python (PEP 8) with 4-space indents, descriptive snake_case for functions and modules, and CapWords for classes or dataclasses. Prefer explicit type hints on new public functions. Keep Streamlit UI code declarative, and limit side effects in `pravah/` modules to preserve reusability in agent workflows.

## Testing & Evaluation

### Eval Suite
The project includes an evaluation suite to test agent behavior:

```bash
# Run full eval suite
uv run python scripts/eval.py --model "gemini/gemini-2.0-flash"

# Run with a specific model and limit
uv run python scripts/eval.py --model "openai/gpt-4o-mini" --limit 10
```

**Eval files:**
- `tests/eval_set.csv` — test cases with expected behaviors
- `tests/eval_results.csv` — output results after running eval
- `tests/traces/` — full trajectory JSON files for debugging

**Eval categories:**
- `happy_path` — normal use cases (search, no-search, greetings)
- `edge_case` — ambiguous queries, no results scenarios
- `adversarial` — prompt injection, hallucination tests

### Unit Tests
New contributions should introduce targeted tests alongside features, ideally with `pytest` under a `tests/` directory. Mock external APIs (Tavily, Gemini, LLM endpoints) to keep runs deterministic. Document any long-running or network-dependent checks in the PR so reviewers can reproduce them.

## Commit & Pull Request Guidelines
Recent history favors concise, present-tense commit subjects ("Add retry logic", "Update prompts"). Use a single focus per commit, and include relevant files only. Pull requests should summarize intent, list manual validation commands, call out new environment keys, and attach UI screenshots or console transcripts when UX changes occur. Cross-link GitHub issues or research threads for context.

## Environment & Secrets
Create a `.env` in the repo root with keys such as `TVLY_API_KEY`, `OPENAI_API_KEY`, `COHERE_API_KEY`, and optional LangSmith settings before running the app. Never commit secrets; rely on `.env` locally and repository secrets in CI/CD. Update documentation if new providers or ports are introduced.

### Required API Keys

| Key | Purpose | Required |
|-----|---------|----------|
| `TVLY_API_KEY` | Tavily web search (`web_search` tool) | Yes (for search) |
| `GOOGLE_API_KEY` or `GEMINI_API_KEY` | Gemini grounded search (`gemini_search` tool) | Optional |
| `OPENAI_API_KEY` | OpenAI models | For OpenAI models |
| `ANTHROPIC_API_KEY` | Claude models | For Anthropic models |
| `COHERE_API_KEY` | Cohere reranking | Optional |

### Search Tools

The agent has two search tools available:

1. **`web_search`** (Tavily) — Returns raw search results with URLs and snippets
   - Best for: fetching full pages afterward, raw search snippets
   - Requires: `TVLY_API_KEY`

2. **`gemini_search`** (Gemini Grounded Search) — Returns AI-synthesized answers with citations
   - Best for: quick answers, information synthesis, fact-checking
   - Model: `gemini-2.5-flash-lite`
   - Requires: `GOOGLE_API_KEY` or `GEMINI_API_KEY`

The agent decides which tool to use based on the query type.
