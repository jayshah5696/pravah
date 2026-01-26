# Pravah v2

Pravah is a search-first chat app that combines web results and agent tools. The name means "flow" in Sanskrit.

![Pravah Demo](assets/demo.gif)

## What it does

- Web search with source-backed answers
- Page fetching and long-document chunking
- Multi-provider LLM support via LiteLLM
- Persistent chat history in DuckDB
- Streaming responses and tool visibility

## Quick start

### Prerequisites

- Python 3.11+
- uv (recommended) or pip

### Install

```bash
git clone https://github.com/jayshah5696/pravah.git
cd pravah

uv sync
```

### Configure

Create a `.env` file in the repo root:

```bash
TVLY_API_KEY=your_tavily_api_key

OPENAI_API_KEY=your_openai_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key
GEMINI_API_KEY=your_gemini_api_key
GROQ_API_KEY=your_groq_api_key
DEEPSEEK_API_KEY=your_deepseek_api_key

LANGCHAIN_API_KEY=your_langsmith_api_key
LANGCHAIN_PROJECT=pravah
```

If you want to customize the model list or UI, edit `config.yaml`.

### Run

```bash
uv run streamlit run app.py
```

Open http://localhost:8501

## Project structure

```
pravah/
├── app.py                 # Streamlit UI
├── config.yaml            # Model and UI configuration
├── pravah/
│   ├── agent.py           # LangGraph agent
│   ├── tools.py           # Tool layer (search, fetch, calculate)
│   ├── prompts.py         # System prompt builder
│   ├── history.py         # DuckDB history storage
│   ├── search.py          # Search providers
│   ├── retrieval.py       # Chunking and reranking
│   └── llm.py             # LLM helpers
├── scripts/
│   └── eval.py            # Evaluation runner
├── tests/
│   └── eval_set.csv       # Eval dataset
├── .streamlit/
│   └── config.toml        # Theme
└── docs/
    └── EVAL_GUIDE.md      # Evaluation notes
```

## Models and providers

The UI reads available models from `config.yaml`. Keep that file as the source of truth for the UI list. If you use a custom model string, enter it in the sidebar.

## Agent tools

| Tool | Purpose |
|------|---------|
| `web_search` | Web search via Tavily |
| `fetch_page` | Fetch and extract page text |
| `read_page_chunk` | Navigate long pages |
| `search_memory` | Search previously fetched content |
| `calculate` | Safe math evaluation |

## Evaluation

```bash
uv run python scripts/eval.py
uv run python scripts/eval.py --limit 10
uv run python scripts/eval.py --verbose
```

Results are written to `tests/eval_results.csv` and `tests/traces/`.

## Docker

```bash
docker build -t pravah .
docker run -p 8501:8501 --env-file .env pravah
```

## Development

```bash
uv sync --group dev
uv run pytest
uv run ruff check .
uv run ruff format .
```

## Contributing

Open an issue or PR with a focused change and a short description of how you tested it.

## License

MIT. See `LICENSE`.
