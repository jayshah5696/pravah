from pravah.llm import completion_llm
from pravah.prompts import generate_prompt_template
from pravah.retrieval import RetrievalEngine
from pravah.search import search_query, get_text_from_url
import asyncio
import aiohttp
from rich.console import Console
from rich.markdown import Markdown
from functools import lru_cache

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

async def main():
    console = Console()
    while True:
        input_query = input("Enter your query (or 'quit' to exit): ")
        if input_query.lower() == 'quit':
            break
        # input_query = "What is an example use case of anymoaly detection algorithms?"

        # Search for relevant context using the input query
        console.print("Searching for relevant context...")
        search_results = search_query(input_query)
        console.print(f"Search results")

        console.print("Fetching texts from search results...")
        # get list of urls
        urls = [result['url'] for result in search_results['results']]
        texts = await fetch_all_texts(urls)
        dict_of_texts = [{'content': text, 'url': url} for text, url in zip(texts, urls)]
        
        console.print(f"Fetched texts: " ,len(dict_of_texts))
        # len of each fetched text
        console.print([len(text['content']) for text in dict_of_texts])
        
        console.print("Initializing RetrievalEngine with fetched texts...")
        retrival = RetrievalEngine(dict_of_texts, tokens=True,chunk_size=1500, overlap=300)
        
        console.print("Performing keyword search on the input query...")
        context = await retrival.keyword_search(input_query,10)
        # context = await retrival.combined_search(input_query,10)
        console.print(f"Retrieved context:")

        console.print("Generating prompt template...")
        prompt = generate_prompt_template(input_query, context)
        console.print(f"Generated prompt: {prompt}")

        console.print("Getting completion from LLM...")
        output = completion_llm(prompt,model='openai/gpt-4o-mini',temperature=0.5)
        console.print(f"LLM output:")
        
        # Render the Markdown output using rich
        md = Markdown(output)
        console.print(md)

if __name__ == "__main__":
    asyncio.run(main())

