import streamlit as st
import asyncio
import aiohttp
import duckdb
from functools import lru_cache
from dataclasses import dataclass
from dotenv import load_dotenv
import os
import uuid
load_dotenv()
from pravah.llm import completion_llm
from pravah.prompts import generate_prompt_template, query_rewriter, extract_rewritten_prompt
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
    rewrite_model: str = 'groq/llama-3.1-8b-instant'
    rewrite_model_temperature: float = 0.1

config = Config(search_tvly_api_key=search_tvly_api_key)

# Initialize DuckDB connection

def create_tables(conn):
    # Create tables without explicit transaction management
    conn.execute("CREATE TABLE IF NOT EXISTS chat_history (conversation_uuid UUID PRIMARY KEY, user_input TEXT, response TEXT)")
    conn.execute("CREATE TABLE IF NOT EXISTS search_results (conversation_uuid UUID, search_result JSON, FOREIGN KEY(conversation_uuid) REFERENCES chat_history(conversation_uuid))")
    conn.execute("CREATE TABLE IF NOT EXISTS fetched_texts (url TEXT PRIMARY KEY, text TEXT)")
    conn.execute("CREATE TABLE IF NOT EXISTS retrieved_chunks (conversation_uuid UUID, search_type TEXT, chunk TEXT, FOREIGN KEY(conversation_uuid) REFERENCES chat_history(conversation_uuid))")
    conn.execute("CREATE TABLE IF NOT EXISTS re_written_prompt (conversation_uuid UUID, re_written_prompt TEXT, FOREIGN KEY(conversation_uuid) REFERENCES chat_history(conversation_uuid))")

def save_to_duckdb(conn, conversation_uuid, prompt, full_response, search_results, texts, urls, context_keyword, context_reranker, re_written_prompt):
    # Save data without explicit transaction management
    conn.execute("INSERT INTO chat_history (conversation_uuid, user_input, response) VALUES (?, ?, ?)", (conversation_uuid, prompt, full_response))
    # Save search results to DuckDB
    conn.execute("INSERT INTO search_results (conversation_uuid, search_result) VALUES (?, ?)", (conversation_uuid, search_results))
    # Save fetched texts to DuckDB
    for text, url in zip(texts, urls):
        # Check if the URL already exists in the database
        existing_text = conn.execute("SELECT text FROM fetched_texts WHERE url = ?", (url,)).fetchone()
        if existing_text is None:
            conn.execute("INSERT INTO fetched_texts (url, text) VALUES (?, ?)", (url, text))
    # Save retrieved chunks (keyword search) to DuckDB
    for chunk in context_keyword:
        conn.execute("INSERT INTO retrieved_chunks (conversation_uuid, search_type, chunk) VALUES (?, ?, ?)", (conversation_uuid, 'keyword_search', chunk))
    # Save retrieved chunks (reranked) to DuckDB
    for chunk in context_reranker:
        conn.execute("INSERT INTO retrieved_chunks (conversation_uuid, search_type, chunk) VALUES (?, ?, ?)", (conversation_uuid, 'reranked', chunk))
    # Save re-written prompt to DuckDB
    conn.execute("INSERT INTO re_written_prompt (conversation_uuid, re_written_prompt) VALUES (?, ?)", (conversation_uuid, re_written_prompt))

with duckdb.connect(database='pravah.db') as conn:  # Use context manager for connection
    create_tables(conn)
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
    config.rewrite_model = st.sidebar.text_input("Rewrite Model", config.rewrite_model)
    config.rewrite_model_temperature = st.sidebar.slider("Rewrite Model Temperature", 0.0, 1.0, config.rewrite_model_temperature)

    # Chat input
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "previous_prompt" not in st.session_state:
        st.session_state.previous_prompt = ""
    previous_prompt = st.session_state.previous_prompt
    # Add a button to reset chat messages
    if st.sidebar.button("Reset Chat"):
        st.session_state.messages = []
        st.session_state.current_context = []
        st.session_state.previous_prompt = ""
        st.rerun()
        
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("What is your question?"):
        # Generate UUID for the conversation
        conversation_uuid = uuid.uuid4()
        
        st.session_state.messages.append({"role": "user", "content": prompt})
        if previous_prompt!='':
            re_written_prompt = extract_rewritten_prompt(completion_llm(query_rewriter(prompt,
                                                                previous_prompt,st.session_state.messages),
                                                                model=config.rewrite_model,
                                                                temperature=config.rewrite_model_temperature, stream=False))
            
        else:
            re_written_prompt = extract_rewritten_prompt(completion_llm(query_rewriter(prompt,None, None)))
        print("************")
        print(re_written_prompt)
        print("************")
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
            search_results = cached_search_query(re_written_prompt)

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
            context_keyword = asyncio.run(retrieval.keyword_search(prompt, config.keyword_search_limit))

            # Rank the context
            update_intermediate('Ranking the context...')
            context_reranker = asyncio.run(retrieval.rerank_chunks(prompt, context_keyword, config.rerank_limit))
                
            # Generate prompt template
            update_intermediate("Generating prompt template...")
            prompt_template = generate_prompt_template(prompt, context_reranker, extra_context={'search_query': re_written_prompt})

            # Get completion from LLM
            update_intermediate("Getting completion from LLM...")
            stream = completion_llm(prompt_template,
                                    model=config.model,
                                    temperature=config.temperature, stream=True)

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
        st.session_state.previous_prompt = re_written_prompt
        # Save chat history to DuckDB
        with duckdb.connect(database='pravah.db') as conn:  
            save_to_duckdb(conn, conversation_uuid, prompt, full_response, search_results, texts, urls, context_keyword, context_reranker, re_written_prompt)

    # Display chat history
    # st.sidebar.header("Chat History")
    # chat_history = conn.execute("SELECT * FROM chat_history").fetchall()
    # for chat in chat_history:
    #     st.sidebar.write(f"User: {chat[1]}")
    #     st.sidebar.write(f"Response: {chat[2]}")

    # Right-side panel for visualizing and bringing history back
    st.sidebar.header("Visualize and Use History")
    with duckdb.connect(database='pravah.db') as conn:  
        chat_history = conn.execute("SELECT user_input, response FROM chat_history").fetchall()
    previous_queries = [f"{chat[0]}" for chat in chat_history]
    selected_history = st.sidebar.selectbox("Select a history to use", previous_queries)
    if st.sidebar.button("Use Selected History"):
        with duckdb.connect(database='pravah.db') as conn:  
            selected_chat = conn.execute("SELECT * FROM chat_history WHERE user_input = ?", (selected_history,)).fetchone()
        st.session_state.current_context = selected_chat
        st.session_state.messages.append({"role": "user", "content": selected_chat[1]})
        st.session_state.messages.append({"role": "assistant", "content": selected_chat[2]})
        st.session_state.previous_prompt = selected_chat[1]
        st.rerun()
        previous_prompt = selected_chat[1]

    # Display current context
    # if 'current_context' in st.session_state and st.session_state['current_context']:
    #     st.sidebar.write("Current Context:")
    #     st.sidebar.write(st.session_state['current_context'])

if __name__ == "__main__":
    main()
