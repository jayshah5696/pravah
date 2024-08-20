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
import PyPDF2
import io
import re
from brave import AsyncBrave
from duckduckgo_search import AsyncDDGS
warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)

# load_dotenv()
# api_key=os.environ['TVLY_API_KEY']

def search_query(query:str, api_key, num_results=5):
    tavily_client = TavilyClient(api_key=api_key, max_results=num_results)
    results = tavily_client.search(query,
                               include_raw_content=False)
    return results

async def search_query_brave(query, api_key, num_results=5):
    brave = AsyncBrave(api_key=api_key)
    search_results = await brave.search(q=query, count=num_results)
    web_results = search_results.web_results
    urls = [x['url'].unicode_string() for x in web_results]
    return {'results':[{'url':url} for url in urls]}

async def search_query_duckduckgo(query, num_results=5):
    search_results = await AsyncDDGS().atext(query, max_results=num_results)
    return {'results':[{'url': result['href']} for result in search_results]}

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

jina_api_key = os.environ['JINA_API_KEY']
JINA_HEADERS = {
    'Authorization': 'Bearer {}'.format(jina_api_key),
    'X-Return-Format': 'markdown'
}
async def fetch_content(url: str) -> str:
    """Fetches the content from a given URL asynchronously.

    Args:
        url: The URL to fetch the content from.
    Returns:
        The content of the page, or an empty string if an error occurs.
    """
    # await asyncio.sleep(random.randint(2, 5))  # Simulate human-like behavior with random sleep durations
    try:
        headers = {'User-Agent': random.choice(USER_AGENTS)}
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers) as response:
                response.raise_for_status()
                if url.endswith(".pdf"):
                    # Handle PDF content
                    content = await response.read()
                    pdf_reader = PyPDF2.PdfReader(io.BytesIO(content))
                    text = ""
                    for page in range(len(pdf_reader.pages)):
                        text += pdf_reader.pages[page].extract_text()
                    return text
                else:
                    # Handle other content types as before
                    return await response.text()
    except aiohttp.ClientError as e:
        print(f"An error occurred while fetching content from {url}: {e}")
        return ''
    except UnicodeDecodeError as e:
        print(f"An error occurred while decoding content from {url}: {e}")
        return ''
def parse_content(content: str, markdown: bool = False) -> str:
    """Parses the HTML content to extract text and convert it to markdown format if enabled.

    Args:
        content: The raw HTML content.
        markdown: A flag to indicate whether to convert to markdown format.
    Returns:
        The text content of the page, or markdown formatted text if enabled.
    """
    try:
        soup = BeautifulSoup(content, 'html.parser')

        if markdown:
            # Handle headings
            for heading in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
                try:
                    heading.string = f"{'#' * int(heading.name[1:])} {heading.get_text(strip=True)}\n"
                except Exception as e:
                    print(f"Error handling heading: {e}")

            # Handle paragraphs
            for p in soup.find_all('p'):
                try:
                    p.string = f"{p.get_text(strip=True)}\n\n"
                except Exception as e:
                    print(f"Error handling paragraph: {e}")

            # Handle links
            for a in soup.find_all('a'):
                try:
                    if 'href' in a.attrs:  # Check if 'href' exists
                        a.string = f"[{a.get_text(strip=True)}]({a['href']})"
                except Exception as e:
                    print(f"Error handling link: {e}")

            # Handle bold and italic text
            for strong in soup.find_all('strong'):
                try:
                    strong.string = f"**{strong.get_text(strip=True)}**"
                except Exception as e:
                    print(f"Error handling bold text: {e}")
            for em in soup.find_all('em'):
                try:
                    em.string = f"*{em.get_text(strip=True)}*"
                except Exception as e:
                    print(f"Error handling italic text: {e}")

            # Handle unordered lists
            for ul in soup.find_all('ul'):
                try:
                    for li in ul.find_all('li'):
                        li.string = f"- {li.get_text(strip=True)}\n"
                except Exception as e:
                    print(f"Error handling unordered list: {e}")

            # Handle ordered lists
            for ol in soup.find_all('ol'):
                try:
                    for i, li in enumerate(ol.find_all('li')):
                        li.string = f"{i+1}. {li.get_text(strip=True)}\n"
                except Exception as e:
                    print(f"Error handling ordered list: {e}")

            # Handle code blocks
            for pre in soup.find_all('pre'):
                try:
                    pre.string = f"```\n{pre.get_text()}\n```"
                except Exception as e:
                    print(f"Error handling code block: {e}")

            # Handle images
            for img in soup.find_all('img'):
                try:
                    alt_text = img.get('alt', '')
                    img.replace_with(f"![{alt_text}]({img['src']})")
                except Exception as e:
                    print(f"Error handling image: {e}")

            # Remove empty tags
            for tag in soup.find_all():
                try:
                    if not tag.get_text(strip=True):
                        tag.decompose()
                except Exception as e:
                    print(f"Error removing empty tag: {e}")

            # Get the final markdown text
            markdown_text = soup.get_text()

            # Clean up extra newlines
            markdown_text = re.sub(r'\n{3,}', '\n\n', markdown_text)

            return markdown_text

        else:
            # Default parsing to extract plain text
            try:
                text = ' '.join([s.get_text(strip=True) for s in soup.find_all()])
                return text
            except Exception as e:
                print(f"Error during plain text extraction: {e}")
                return ""

    except Exception as e:
        print(f"An error occurred during parsing: {e}")
        for fallback in [fallback_to_plain_text, fallback_to_partial_markdown, fallback_to_simplified_parsing]:
            try:
                markdown_text = fallback(content)
                print(f"Fallback successful using {fallback.__name__}.")
                return markdown_text
            except Exception as e:
                print(f"Error during fallback {fallback.__name__}: {e}")
        return ""
    
def fallback_to_plain_text(content):
    soup = BeautifulSoup(content, 'html.parser')
    return soup.get_text()

def fallback_to_partial_markdown(content):
    soup = BeautifulSoup(content, 'html.parser')
    markdown_text = ""
    for heading in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
        markdown_text += f"{'#' * int(heading.name[1:])} {heading.get_text(strip=True)}\n"
    for p in soup.find_all('p'):
        markdown_text += f"{p.get_text(strip=True)}\n\n"
    return markdown_text

def fallback_to_simplified_parsing(content):
    soup = BeautifulSoup(content, 'html.parser')
    markdown_text = ""
    for element in soup.find_all():
        if element.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p']:
            markdown_text += f"{element.get_text(strip=True)}\n\n"
    return markdown_text
    

async def get_text_from_url(url: str, search_type: str = 'default', markdown: bool = False) -> str:
    """Fetches and parses the text content from a given URL asynchronously.

    Args:
        url: The URL to fetch the content from.
        search_type: The type of search to perform ('default' or 'jina').
        markdown: A flag to indicate whether to convert to markdown format.
    Returns:
        The text content of the page, or an empty string if an error occurs.
    """
    if search_type == 'jina':
        headers = JINA_HEADERS  # Use the predefined JINA headers
        content = await fetch_jina_content(url, headers)
    else:
        content = await fetch_content(url)

    if content:
        return parse_content(content, markdown) if search_type == 'default' else content
    return ''

async def fetch_jina_content(url: str, headers: dict) -> str:

    url = f'https://r.jina.ai/{url}'
    async with aiohttp.ClientSession() as session:
        async with session.get(url, headers=headers) as response:
            response.raise_for_status()
            return await response.text()