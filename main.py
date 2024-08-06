import asyncio
import aiohttp
from rich.console import Console
from rich.markdown import Markdown
from functools import lru_cache
from dataclasses import dataclass
from dotenv import load_dotenv
import os
load_dotenv()
from pravah.llm import completion_llm
from pravah.prompts import generate_prompt_template
from pravah.retrieval import RetrievalEngine
from pravah.search import search_query, get_text_from_url

search_tvly_api_key = os.environ['TVLY_API_KEY']

@lru_cache(maxsize=128)
def cached_search_query(query):
    return search_query(query)

@lru_cache(maxsize=128)
async def fetch_text(session, url):
    return await get_text_from_url(url)

async def fetch_all_texts(urls):
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_text(session, url) for url in urls]
        return await asyncio.gather(*tasks)



@dataclass
class Config:
    search_tvly_api_key: str
    model: str = 'openai/gpt-4o-mini'
    temperature: float = 0.5
    tokens: bool = True
    chunk_size: int = 1500
    overlap: int = 300
    keyword_search_limit: int = 20
    rerank_limit: int = 10

config = Config(search_tvly_api_key=os.environ['TVLY_API_KEY'])

async def main():
    console = Console()
    while True:
        input_query = input("Enter your query (or 'quit' to exit): ")
        if input_query.lower() == 'quit':
            break

        console.print("Searching for relevant context...")
        search_results = search_query(input_query, api_key=config.search_tvly_api_key)
        console.print(f"Search results")

        console.print("Fetching texts from search results...")
        urls = [result['url'] for result in search_results['results']]
        texts = await fetch_all_texts(urls)
        dict_of_texts = [{'content': text, 'url': url} for text, url in zip(texts, urls)]
        
        console.print(f"Fetched texts: " ,len(dict_of_texts))
        console.print([len(text['content']) for text in dict_of_texts])
        
        console.print("Initializing RetrievalEngine with fetched texts...")
        retrival = RetrievalEngine(dict_of_texts, tokens=config.tokens, chunk_size=config.chunk_size, overlap=config.overlap)
        
        console.print("Performing keyword search on the input query...")
        context = await retrival.keyword_search(input_query, config.keyword_search_limit)

        console.print('Ranking the context...')
        context = await retrival.rerank_chunks(input_query, context, config.rerank_limit)
        console.print(f"Retrieved context:")

        console.print("Generating prompt template...")
        prompt = generate_prompt_template(input_query, context)

        console.print("Getting completion from LLM...")
        output = completion_llm(prompt, model=config.model, temperature=config.temperature)
        console.print(f"LLM output:")
        
        md = Markdown(output)
        console.print(md)

if __name__ == "__main__":
    asyncio.run(main())