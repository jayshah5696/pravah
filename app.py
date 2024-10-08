import streamlit as st
import asyncio
import aiohttp
import duckdb
from functools import lru_cache
from dataclasses import dataclass
from dotenv import load_dotenv, set_key
import os
import uuid
from rerankers import Reranker
from langsmith import traceable, Client
load_dotenv()
from pravah.llm import completion_llm
from pravah.prompts import generate_prompt_template, query_rewriter, extract_rewritten_prompt
from pravah.retrieval import RetrievalEngine, LiteLLMEmbeddingClient
from pravah.search import search_query, get_text_from_url, search_query_brave, search_query_duckduckgo
from langsmith.run_trees import RunTree
# Load API key from environment
search_tvly_api_key = os.environ['TVLY_API_KEY']
if search_tvly_api_key is None:
    raise ValueError("Please set the TVLY_API_KEY environment variable")

# Define configuration dataclass
@dataclass
class Config:
    search_tvly_api_key: str
    model: str = 'openai/gpt-4o-mini'
    temperature: float = 0.5
    chunking_method: str = 'tokens'
    chunk_size: int = 1500
    overlap: int = 300
    keyword_search_limit: int = 20
    rerank_limit: int = 10
    rewrite_model: str = 'groq/llama-3.1-8b-instant'
    rewrite_model_temperature: float = 0.1
    search_type: str = 'default' # or jina
    markdown: bool = True
    search_engine: str = 'tvly'
    reranker: str = 'cohere' # or flashrank

config = Config(search_tvly_api_key=search_tvly_api_key)

def update_env_file(key, value):
    env_path = os.path.join(os.path.dirname(__file__), '.env')
    set_key(env_path, key, value)
    os.environ[key] = value

def check_model_key(model):
    if 'openai' in model.lower():
        return 'OPENAI_API_KEY'
    elif 'anthropic' in model.lower():
        return 'ANTHROPIC_API_KEY'
    elif 'groq' in model.lower():
        return 'GROQ_API_KEY'
    elif 'cohere' in model.lower():
        return 'COHERE_API_KEY'
    return None

def check_api_keys(config):
    required_keys = []
    api_key_info = {
        'TVLY_API_KEY': {
            'description': "To set up the TVLY API key, please visit the TVLY developer portal and create an account. After that, you can generate your API key from the dashboard.",
            'link': "https://app.tavily.com/sign-in"
        },
        'BRAVE_API_KEY': {
            'description': "To obtain the Brave API key, go to the Brave Search API page, sign up, and follow the instructions to generate your API key.",
            'link': "https://brave.com/search/api/"
        },
        'COHERE_API_KEY': {
            'description': "For the Cohere API key, visit the Cohere website, create an account, and generate your API key from the API section.",
            'link': "https://cohere.ai"
        },
        'JINA_API_KEY': {
            'description': "To get the Jina API key, sign up on the Jina AI website and navigate to the API section to create your key.",
            'link': "https://jina.ai"
        },
        'LANGCHAIN_API_KEY': {
            'description': "You can obtain the LangChain API key by signing up on the Langsmith website and generating it from your account settings.",
            'link': "https://langchain.com"
        },
        'LANGCHAIN_PROJECT': {
            'description': "You can obtain the LangChain API key by signing up on the Langsmith website and generating it from your account settings.",
            'link': "https://langchain.com"
        },
        'LANGCHAIN_ENDPOINT': {
            'description': "You can obtain the LangChain API key by signing up on the Langsmith website and generating it from your account settings.",
            'link': "https://langchain.com"
        },
        'OPENAI_API_KEY': {
            'description': "To acquire the OpenAI API key, visit the OpenAI website, create an account, and generate your API key from the API section.",
            'link': "https://openai.com"
        },
        'ANTHROPIC_API_KEY': {
            'description': "To obtain the Anthropic API key, sign up on the Anthropic website and follow the instructions to generate your API key.",
            'link': "https://anthropic.com"
        },
        'GROQ_API_KEY': {
            'description': "For the Groq API key, visit the Groq website, create an account, and generate your API key from the API section.",
            'link': "https://groq.com"
        }
    }

    if config.search_engine == 'tvly':
        required_keys.append('TVLY_API_KEY')
    elif config.search_engine == 'brave':
        required_keys.append('BRAVE_API_KEY')

    # Check API keys for both model and rewrite_model
    for model in [config.model, config.rewrite_model]:
        key = check_model_key(model)
        if key:
            required_keys.append(key)
    
    if config.reranker == 'cohere':
        required_keys.append('COHERE_API_KEY')
    
    if config.search_type == 'jina':
        required_keys.append('JINA_API_KEY')

    # Check for LangSmith related API keys
    if 'LANGCHAIN_API_KEY' not in os.environ:
        required_keys.append('LANGCHAIN_API_KEY')
    if 'LANGCHAIN_PROJECT' not in os.environ:
        required_keys.append('LANGCHAIN_PROJECT')

    missing_keys = [key for key in required_keys if not os.environ.get(key)]

    if missing_keys:
        st.warning("Some required API keys are missing. Please enter them below:")
        for key in missing_keys:
            st.info(api_key_info[key]['description'])
            st.markdown(f"[More Info]({api_key_info[key]['link']})")
            value = st.text_input(f"Enter {key}", type="password")
            if value:
                update_env_file(key, value)
        
        if st.button("Save API Keys"):
            st.success("API keys saved successfully. Please restart the app.")
            st.stop()

def setup_config_and_check_api_keys():
    st.sidebar.header("Configuration")

    with st.sidebar.expander("Model Configuration", expanded=True):
        config.model = st.text_input("Model", config.model, key='model_input')
        config.temperature = st.slider("Temperature", 0.0, 1.0, config.temperature, key='temperature_slider')

    with st.sidebar.expander("Chunking Configuration", expanded=False):
        config.chunk_size = st.number_input("Chunk Size", value=config.chunk_size, key='chunk_size_input')
        config.chunking_method = st.selectbox("Chunking Method", ["tokens", "text", 'regex'], key='chunking_method_select')
        config.overlap = st.number_input("Overlap", value=config.overlap, key='overlap_input')

    with st.sidebar.expander("Search Configuration", expanded=False):
        config.keyword_search_limit = st.number_input("Keyword Search Limit", value=config.keyword_search_limit, key='keyword_search_limit_input')
        config.rerank_limit = st.number_input("Rerank Limit", value=config.rerank_limit, key='rerank_limit_input')
        config.search_type = st.selectbox("Search Type", ["default", "jina"], key='search_type_select')
        config.markdown = st.checkbox("Markdown", value=config.markdown, key='markdown_checkbox')
        config.search_engine = st.selectbox("Search Engine", ["duckduckgo", "brave", "tvly"], key='search_engine_select')
        config.use_lancedb = st.checkbox("Use LanceDB", value=True, key='use_lancedb_checkbox')

    with st.sidebar.expander("Rewrite Configuration", expanded=False):
        config.rewrite_model = st.text_input("Rewrite Model", config.rewrite_model, key='rewrite_model_input')
        config.rewrite_model_temperature = st.slider("Rewrite Model Temperature", 0.0, 1.0, config.rewrite_model_temperature, key='rewrite_model_temperature_slider')

    with st.sidebar.expander("Reranker Configuration", expanded=False):
        config.reranker = st.selectbox("Reranker", ["cohere", "flashrank"], key='reranker_select')

    # Check for required API keys based on configuration
    check_api_keys(config)

    return config

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
@traceable  # Add tracing to the search query function
def cached_search_query(query, num_results=10):
    if config.search_engine == 'tvly':
        return search_query(query, api_key=config.search_tvly_api_key, num_results=num_results)
    elif config.search_engine == 'brave':
        brave_api_key = os.environ['BRAVE_API_KEY'] 
        if brave_api_key is None:
            raise ValueError("Please set the BRAVE_API_KEY environment variable")
        return asyncio.run(search_query_brave(query, api_key=brave_api_key, num_results=num_results))
    elif config.search_engine == 'duckduckgo':
        return asyncio.run(search_query_duckduckgo(query, num_results=num_results))
    else:
        raise ValueError("Unsupported search engine")

# Fetch text from URL
@lru_cache(maxsize=128)
@traceable  # Add tracing to the fetch text function
async def fetch_text(session, url):
    return await get_text_from_url(url, search_type=config.search_type, markdown=config.markdown)

# Fetch all texts from URLs
@traceable  # Add tracing to the fetch all texts function
async def fetch_all_texts(urls):
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_text(session, url) for url in urls]
        return await asyncio.gather(*tasks)

def display_intermediate_result(message):
    with st.empty():
        st.info(message)


def main():
    st.set_page_config(page_title="Pravaha", page_icon=":ocean:", layout="wide")
    
    # Add custom CSS
    st.markdown(
        """
        <style>
        .css-18e3th9 {
            padding-top: 2rem;
        }
        .css-1d391kg {
            padding-top: 2rem;
        }
        .css-1v3fvcr {
            padding-top: 2rem;
        }
        .css-1cpxqw2 {
            padding-top: 2rem;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Add logo/image
    st.image("assets/pravha.png", width=200)  # Update the path to your logo

    st.title("Pravaha")
    
    # Setup configuration and check API keys
    config = setup_config_and_check_api_keys()

    # Initialize the main RunTree
    main_run = RunTree(name="pravah Run", run_type="chain", inputs={"config": config})

    # Chat input
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "previous_prompt" not in st.session_state:
        st.session_state.previous_prompt = ""
    previous_prompt = st.session_state.previous_prompt
    
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
        
        # Step 1: Rewriting the Prompt
        rewrite_run = main_run.create_child(name="Rewriting Prompt", run_type="llm", inputs={"prompt": prompt})
        if previous_prompt != '':
            re_written_prompt = extract_rewritten_prompt(completion_llm(query_rewriter(prompt, previous_prompt, st.session_state.messages),
                                                                        model=config.rewrite_model,
                                                                        temperature=config.rewrite_model_temperature, stream=False))
        else:
            re_written_prompt = extract_rewritten_prompt(completion_llm(query_rewriter(prompt,None, None),
                                                                model=config.rewrite_model,
                                                                temperature=config.rewrite_model_temperature, stream=False))
        print("************")
        print(re_written_prompt)
        print("************")
        rewrite_run.end(outputs={"re_written_prompt": re_written_prompt})
        
        
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
            
            # Step 2: Search Query (Keyword or Combined Search)
            search_run = main_run.create_child(name="Search Query", run_type="chain", inputs={"re_written_prompt": re_written_prompt})
            search_results = cached_search_query(re_written_prompt)
            search_run.end(outputs={"search_results": search_results})

            # Fetch texts from search results
            update_intermediate("Fetching texts from search results...")
            urls = [result['url'] for result in search_results['results']]
            
            fetch_run = main_run.create_child(name="Fetch Texts", run_type="chain", inputs={"urls": urls})
            texts = asyncio.run(fetch_all_texts(urls))
            dict_of_texts = [{'content': text, 'url': url} for text, url in zip(texts, urls)]
            fetch_run.end(outputs={"texts": dict_of_texts})

            update_intermediate(f"Fetched {len(dict_of_texts)} texts")

            # Initialize RetrievalEngine
            update_intermediate("Initializing RetrievalEngine with fetched texts...")
            retrieval = RetrievalEngine(
                dict_of_texts,
                uuid_input=str(conversation_uuid),
                chunking_method=config.chunking_method,
                chunk_size=config.chunk_size,
                overlap=config.overlap,
                use_lancedb=config.use_lancedb, 
                reranker=Reranker(
                    config.reranker, 
                    lang='en', 
                    api_key=os.environ.get('COHERE_API_KEY') if config.reranker == 'cohere' else None
                )
            ) 
            # Perform keyword search
            # update_intermediate("Performing keyword search on the input query...")
            # context_keyword = asyncio.run(retrieval.keyword_search(prompt, config.keyword_search_limit))

            # # Rank the context
            # update_intermediate('Ranking the context...')
            # context_reranker = asyncio.run(retrieval.rerank_chunks(prompt, context_keyword, config.rerank_limit))
            
            # Step 3: Combined Search
            combined_search_run = main_run.create_child(name="Combined Search", run_type="chain", inputs={"prompt": prompt})
            context_reranker = asyncio.run(retrieval.combined_search(prompt, config.rerank_limit))
            combined_search_run.end(outputs={"context_reranker": context_reranker})
            
            context_keyword = []
            update_intermediate("Generating prompt template...")
            prompt_template = generate_prompt_template(prompt, context_reranker, extra_context={'search_query': re_written_prompt})

            # Get completion from LLM
            update_intermediate("Getting completion from LLM...")
            
            # Step 4: Streaming Final Output
            stream_run = main_run.create_child(name="Streaming Final Output", run_type="llm", inputs={"prompt_template": prompt_template})
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
            stream_run.end(outputs={"final_output": full_response})
        
        st.session_state.messages.append({"role": "assistant", "content": full_response})
        st.session_state.previous_prompt = re_written_prompt
        
        with duckdb.connect(database='pravah.db') as conn:  
            save_to_duckdb(conn, conversation_uuid, prompt, full_response, search_results, texts, urls, context_keyword, context_reranker, re_written_prompt)


        main_run.end(outputs={"final_output": full_response})  # Ensure the RunTree is properly ended

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

if __name__ == "__main__":
    main()
