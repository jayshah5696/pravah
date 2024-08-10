from tavily import TavilyClient
from dotenv import load_dotenv
import os
import requests
import random
import time
from requests import get
from bs4 import BeautifulSoup
import aiohttp
import asyncio
import warnings
from bs4 import MarkupResemblesLocatorWarning

warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)

# load_dotenv()
# api_key=os.environ['TVLY_API_KEY']

def search_query(query:str, api_key):
    tavily_client = TavilyClient(api_key=api_key)
    results = tavily_client.search(query,
                               include_raw_content=True)
    return results


# List of user-agents
USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3', 
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/54.0.2840.99 Safari/537.3',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/60.0.3112.113 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/64.0.3282.140 Safari/537.36 Edge/17.17134',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_6) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/11.1.2 Safari/605.1.15',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:61.0) Gecko/20100101 Firefox/61.0',
    'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:52.0) Gecko/20100101 Firefox/52.0',
    'Mozilla/5.0 (iPhone; CPU iPhone OS 11_0 like Mac OS X) AppleWebKit/604.1.38 (KHTML, like Gecko) Version/11.0 Mobile/15A372 Safari/604.1',
    'Mozilla/5.0 (Linux; Android 8.0.0; SM-G960F Build/R16NW) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/62.0.3202.84 Mobile Safari/537.36',
    'Mozilla/5.0 (Windows NT 6.1; WOW64; Trident/7.0; AS; rv:11.0) like Gecko'
]

# Proxy details
PROXIES = {
    'http': 'http://your_proxy_server:port',
    'https': 'https://your_proxy_server:port'
}

async def fetch_content(url: str) -> str:
    """Fetches the raw HTML content from a given URL asynchronously.

    Args:
        url: The URL to fetch the content from.
    Returns:
        The raw HTML content of the page, or an empty string if an error occurs.
    """
    # await asyncio.sleep(random.randint(2, 5))  # Simulate human-like behavior with random sleep durations
    try:
        headers = {'User-Agent': random.choice(USER_AGENTS)}
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers) as response:
                response.raise_for_status()
                return await response.text()
    except aiohttp.ClientError as e:
        print(f"An error occurred while fetching content from {url}: {e}")
        return ''

def parse_content(content: str) -> str:
    """Parses the HTML content to extract text.

    Args:
        content: The raw HTML content.
    Returns:
        The text content of the page.
    """
    soup = BeautifulSoup(content, 'html.parser')
    text = ' '.join([s.get_text(strip=True) for s in soup.find_all()])
    return text

async def get_text_from_url(url: str) -> str:
    """Fetches and parses the text content from a given URL asynchronously.

    Args:
        url: The URL to fetch the content from.
    Returns:
        The text content of the page, or an empty string if an error occurs.
    """
    content = await fetch_content(url)
    if content:
        return parse_content(content)
    return ''