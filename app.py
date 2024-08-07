import streamlit as st
import asyncio
import aiohttp
import duckdb
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



# Load API key from environment
search_tvly_api_key = os.environ['TVLY_API_KEY']

# Define configuration dataclass
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

config = Config(search_tvly_api_key=search_tvly_api_key)

# Initialize DuckDB connection
conn = duckdb.connect(database='pravah.db')
conn.execute("CREATE TABLE IF NOT EXISTS chat_history (id INTEGER PRIMARY KEY, user_input TEXT, response TEXT)")

# Cache search query
@lru_cache(maxsize=128)
def cached_search_query(query):
    return search_query(query,api_key = config.search_tvly_api_key)

# Fetch text from URL
@lru_cache(maxsize=128)
async def fetch_text(session, url):
    return await get_text_from_url(url)

# Fetch all texts from URLs
async def fetch_all_texts(urls):
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_text(session, url) for url in urls]
        return await asyncio.gather(*tasks)


def display_intermediate_result(message):
    with st.empty():
        st.info(message)
# Main Streamlit app
def main():
    st.title("Pravaha")

    # Sidebar configuration
    st.sidebar.header("Configuration")
    config.model = st.sidebar.text_input("Model", config.model)
    config.temperature = st.sidebar.slider("Temperature", 0.0, 1.0, config.temperature)
    config.chunk_size = st.sidebar.number_input("Chunk Size", value=config.chunk_size)
    config.overlap = st.sidebar.number_input("Overlap", value=config.overlap)
    config.keyword_search_limit = st.sidebar.number_input("Keyword Search Limit", value=config.keyword_search_limit)
    config.rerank_limit = st.sidebar.number_input("Rerank Limit", value=config.rerank_limit)

    # Chat input
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Add a button to reset chat messages
    if st.sidebar.button("Reset Chat"):
        st.session_state.messages = []
        st.session_state.current_context = []
        st.experimental_rerun()
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("What is your question?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            # Create a placeholder for intermediate results
            intermediate_placeholder = st.empty()
            
            # Create a placeholder for the final response
            response_placeholder = st.empty()

            full_response = ""

            # Function to update intermediate results
            def update_intermediate(message):
                intermediate_placeholder.info(message)

            # Search for relevant context
            update_intermediate("Searching for relevant context...")
            search_results = cached_search_query(prompt)

            # Fetch texts from search results
            update_intermediate("Fetching texts from search results...")
            urls = [result['url'] for result in search_results['results']]
            texts = asyncio.run(fetch_all_texts(urls))
            dict_of_texts = [{'content': text, 'url': url} for text, url in zip(texts, urls)]

            update_intermediate(f"Fetched {len(dict_of_texts)} texts")

            # Initialize RetrievalEngine
            update_intermediate("Initializing RetrievalEngine with fetched texts...")
            retrieval = RetrievalEngine(dict_of_texts, tokens=config.tokens, chunk_size=config.chunk_size, overlap=config.overlap)

            # Perform keyword search
            update_intermediate("Performing keyword search on the input query...")
            context = asyncio.run(retrieval.keyword_search(prompt, config.keyword_search_limit))

            # Rank the context
            update_intermediate('Ranking the context...')
            context = asyncio.run(retrieval.rerank_chunks(prompt, context, config.rerank_limit))

            # Generate prompt template
            update_intermediate("Generating prompt template...")
            prompt_template = generate_prompt_template(prompt, context)

            # Get completion from LLM
            update_intermediate("Getting completion from LLM...")
            stream = completion_llm(prompt_template, model=config.model, temperature=config.temperature, stream=True)

            # Clear the intermediate placeholder
            intermediate_placeholder.empty()

            # Display the streaming response
            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    full_response += chunk.choices[0].delta.content
                    response_placeholder.markdown(full_response + "▌")

            response_placeholder.empty()
            # Display the final response
            response_placeholder.markdown(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})

        # Save chat history to DuckDB
        max_id = conn.execute("SELECT COALESCE(MAX(id), 0) + 1 FROM chat_history").fetchone()[0]
        conn.execute("INSERT INTO chat_history (id, user_input, response) VALUES (?, ?, ?)", (max_id, prompt, full_response))

    # Display chat history
    st.sidebar.header("Chat History")
    chat_history = conn.execute("SELECT * FROM chat_history").fetchall()
    for chat in chat_history:
        st.sidebar.write(f"User: {chat[1]}")
        st.sidebar.write(f"Response: {chat[2]}")

    # Right-side panel for visualizing and bringing history back
    st.sidebar.header("Visualize and Use History")
    chat_history = conn.execute("SELECT id, user_input, response FROM chat_history").fetchall()
    previous_queries = [f"ID: {chat[0]} - {chat[1]}" for chat in chat_history]
    selected_history = st.sidebar.selectbox("Select a history to use", previous_queries)
    if st.sidebar.button("Use Selected History"):
        selected_id = int(selected_history.split()[1])
        selected_chat = conn.execute("SELECT * FROM chat_history WHERE id = ?", (selected_id,)).fetchone()
        st.session_state.current_context = selected_chat
        st.session_state.messages.append({"role": "user", "content": selected_chat[1]})
        st.session_state.messages.append({"role": "assistant", "content": selected_chat[2]})
        st.experimental_rerun()

    # Display current context
    # if 'current_context' in st.session_state and st.session_state['current_context']:
    #     st.sidebar.write("Current Context:")
    #     st.sidebar.write(st.session_state['current_context'])

if __name__ == "__main__":
    main()