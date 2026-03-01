"""Pravah v1 CLI pipeline - legacy search-retrieve-generate workflow.

This module provides a terminal-based interface using the original
search -> retrieve -> chunk -> rerank -> generate pipeline.
For the agent-based workflow, use app.py instead.
"""

import asyncio
import os
from dataclasses import dataclass

import aiohttp
from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown

from pravah.llm import completion_llm
from pravah.prompts import generate_prompt_template
from pravah.retrieval import RetrievalEngine
from pravah.search import get_text_from_url, search_query

load_dotenv()


@dataclass
class Config:
    search_tvly_api_key: str
    model: str = "openai/gpt-4o-mini"
    temperature: float = 0.5
    tokens: bool = True
    chunk_size: int = 1500
    overlap: int = 300
    keyword_search_limit: int = 20
    rerank_limit: int = 10


async def fetch_all_texts(urls: list[str]) -> list[str]:
    """Fetch text content from multiple URLs concurrently."""
    async with aiohttp.ClientSession():
        tasks = [get_text_from_url(url) for url in urls]
        return await asyncio.gather(*tasks)


async def main() -> None:
    console = Console()
    config = Config(search_tvly_api_key=os.environ["TVLY_API_KEY"])

    while True:
        input_query = input("Enter your query (or 'quit' to exit): ")
        if input_query.lower() == "quit":
            break

        console.print("Searching for relevant context...")
        search_results = search_query(
            input_query, api_key=config.search_tvly_api_key
        )

        console.print("Fetching texts from search results...")
        urls = [result["url"] for result in search_results["results"]]
        texts = await fetch_all_texts(urls)
        dict_of_texts = [
            {"content": text, "url": url} for text, url in zip(texts, urls)
        ]

        console.print(f"Fetched {len(dict_of_texts)} texts")

        console.print("Initializing RetrievalEngine...")
        retrieval = RetrievalEngine(
            dict_of_texts,
            uuid_input="cli",
            tokens=config.tokens,
            chunk_size=config.chunk_size,
            overlap=config.overlap,
        )

        console.print("Performing keyword search...")
        context = await retrieval.keyword_search(
            input_query, config.keyword_search_limit
        )

        console.print("Ranking the context...")
        context = await retrieval.rerank_chunks(
            input_query, context, config.rerank_limit
        )

        console.print("Generating response...")
        prompt = generate_prompt_template(input_query, context)
        output = completion_llm(
            prompt, model=config.model, temperature=config.temperature
        )

        console.print(Markdown(output))


if __name__ == "__main__":
    asyncio.run(main())
