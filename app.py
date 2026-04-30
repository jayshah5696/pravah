"""
Pravah v2 - AI Search Engine
A Streamlit chat application powered by LangGraph ReAct Agent

Features:
- Full chat interface with message history
- Model selection via LiteLLM (OpenAI, Anthropic, Groq, etc.)
- Real-time tool execution visibility
- Streaming responses
- Session persistence
- Configuration via config.yaml
"""

import streamlit as st
import asyncio
import uuid
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

from pravah.history import get_history_store, HistoryStore

# Load environment variables
load_dotenv()


# ============================================================================
# Configuration Loading
# ============================================================================

CONFIG_PATH = Path(__file__).parent / "config.yaml"


@st.cache_data
def load_yaml_config() -> dict[str, Any]:
    """Load configuration from YAML file."""
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH) as f:
            return yaml.safe_load(f)
    return {}


def get_available_models() -> list[str]:
    """Get list of available model names from config."""
    config = load_yaml_config()
    models = config.get("models", [])
    return [m["name"] for m in models if isinstance(m, dict) and "name" in m]


def get_model_descriptions() -> dict[str, str]:
    """Get model name -> description mapping."""
    config = load_yaml_config()
    models = config.get("models", [])
    return {
        m["name"]: m.get("description", "")
        for m in models
        if isinstance(m, dict) and "name" in m
    }


def get_api_key_mapping() -> dict[str, str]:
    """Get provider prefix -> env var mapping."""
    config = load_yaml_config()
    return config.get("api_keys", {})


def get_defaults() -> dict[str, Any]:
    """Get default settings from config."""
    config = load_yaml_config()
    return config.get("defaults", {})


# Must be the first Streamlit command
_yaml_config = load_yaml_config()
_ui_config = _yaml_config.get("ui", {})

st.set_page_config(
    page_title=_ui_config.get("page_title", "Pravah - AI Search Engine"),
    page_icon=_ui_config.get("page_icon", "assets/pravha.png"),
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for cleaner UI
st.markdown(
    """
<style>
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Rounded chat input */
    .stChatInput > div > div > input {
        border-radius: 20px;
    }
    
    /* Better sidebar spacing */
    .stSidebar > div:first-child {
        padding-top: 1rem;
    }
    
    /* Conversation history buttons */
    .stSidebar button[kind="secondary"] {
        text-align: left;
        font-size: 0.85rem;
        padding: 0.4rem 0.6rem;
    }
    
    /* Tool calls expander styling */
    .streamlit-expanderHeader {
        font-size: 0.85rem;
        color: #666;
    }
    
    /* Debug panel metrics */
    [data-testid="stMetricValue"] {
        font-size: 1.1rem;
    }
    
    /* Welcome message styling */
    .welcome-message h3 {
        margin-bottom: 1rem;
    }
    
    /* Keyboard shortcut hints */
    .shortcut-hint {
        font-size: 0.7rem;
        color: #888;
        margin-left: 0.5rem;
    }
    
    /* Scrollable sidebar container styling */
    [data-testid="stSidebar"] [data-testid="stVerticalBlockBorderWrapper"] > div {
        scrollbar-width: thin;
        scrollbar-color: #888 transparent;
    }
    
    [data-testid="stSidebar"] [data-testid="stVerticalBlockBorderWrapper"] > div::-webkit-scrollbar {
        width: 6px;
    }
    
    [data-testid="stSidebar"] [data-testid="stVerticalBlockBorderWrapper"] > div::-webkit-scrollbar-track {
        background: transparent;
    }
    
    [data-testid="stSidebar"] [data-testid="stVerticalBlockBorderWrapper"] > div::-webkit-scrollbar-thumb {
        background-color: #888;
        border-radius: 3px;
    }
    
    [data-testid="stSidebar"] [data-testid="stVerticalBlockBorderWrapper"] > div::-webkit-scrollbar-thumb:hover {
        background-color: #666;
    }
</style>
""",
    unsafe_allow_html=True,
)

# Keyboard shortcuts via JavaScript
st.markdown(
    """
<script>
document.addEventListener('keydown', function(e) {
    // Cmd/Ctrl + K - Focus search box
    if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
        e.preventDefault();
        const searchInput = document.querySelector('input[placeholder="Search..."]');
        if (searchInput) {
            searchInput.focus();
            searchInput.select();
        }
    }
    
    // Cmd/Ctrl + N - New chat (click the New Chat button)
    if ((e.metaKey || e.ctrlKey) && e.key === 'n') {
        e.preventDefault();
        const newChatBtn = Array.from(document.querySelectorAll('button')).find(
            btn => btn.textContent.includes('New Chat')
        );
        if (newChatBtn) {
            newChatBtn.click();
        }
    }
    
    // Escape - Focus chat input
    if (e.key === 'Escape') {
        const chatInput = document.querySelector('[data-testid="stChatInput"] textarea, [data-testid="stChatInput"] input');
        if (chatInput) {
            chatInput.focus();
        }
    }
    
    // / - Focus chat input (when not already in input)
    if (e.key === '/' && !['INPUT', 'TEXTAREA'].includes(document.activeElement.tagName)) {
        e.preventDefault();
        const chatInput = document.querySelector('[data-testid="stChatInput"] textarea, [data-testid="stChatInput"] input');
        if (chatInput) {
            chatInput.focus();
        }
    }
});
</script>
""",
    unsafe_allow_html=True,
)


# ============================================================================
# Configuration
# ============================================================================


@dataclass
class Config:
    """Application configuration."""

    # Model settings
    model: str = field(
        default_factory=lambda: get_defaults().get("model", "openai/gpt-4o-mini")
    )
    temperature: float = field(
        default_factory=lambda: get_defaults().get("temperature", 0.3)
    )

    # Search settings
    search_engine: str = "tvly"
    max_search_results: int = field(
        default_factory=lambda: get_defaults().get("max_search_results", 5)
    )

    # Available models loaded from config.yaml
    available_models: list[str] = field(default_factory=get_available_models)


# ============================================================================
# API Key Management
# ============================================================================


def get_required_api_keys(model: str) -> list[str]:
    """Determine which API keys are required based on the selected model."""
    api_key_mapping = get_api_key_mapping()
    required = [
        api_key_mapping.get("tavily", "TVLY_API_KEY")
    ]  # Always need Tavily for search

    # Extract provider from model string (e.g., "openai/gpt-4o" -> "openai")
    provider = model.split("/")[0].lower() if "/" in model else model.lower()

    # Look up the API key for this provider
    if provider in api_key_mapping:
        required.append(api_key_mapping[provider])
    elif "claude" in model.lower():
        # Handle claude models that might not have anthropic prefix
        required.append(api_key_mapping.get("anthropic", "ANTHROPIC_API_KEY"))

    return required


def check_api_keys(required_keys: list[str]) -> dict[str, bool]:
    """Check which required API keys are present."""
    return {key: bool(os.environ.get(key)) for key in required_keys}


def render_api_key_setup():
    """Render API key setup UI in sidebar."""
    st.sidebar.markdown("---")
    st.sidebar.subheader("API Keys")

    # Get required keys for current model
    config = st.session_state.get("config", Config())
    required_keys = get_required_api_keys(config.model)
    key_status = check_api_keys(required_keys)

    # Show status
    all_present = all(key_status.values())

    if all_present:
        st.sidebar.success("All required API keys configured")
    else:
        st.sidebar.warning("Some API keys missing")

    # Expandable section for key management
    with st.sidebar.expander("Manage API Keys", expanded=not all_present):
        for key, present in key_status.items():
            col1, col2 = st.columns([3, 1])
            with col1:
                if present:
                    st.text(f"✅ {key}")
                else:
                    new_value = st.text_input(
                        f"{key}",
                        type="password",
                        key=f"input_{key}",
                        placeholder="Enter API key...",
                    )
                    if new_value:
                        os.environ[key] = new_value
                        st.rerun()


# ============================================================================
# History Sidebar
# ============================================================================


def _group_conversations_by_date(conversations: list[dict]) -> dict[str, list]:
    """Group conversations by date categories."""
    from datetime import date, timedelta

    today = date.today()
    yesterday = today - timedelta(days=1)
    week_ago = today - timedelta(days=7)
    month_ago = today - timedelta(days=30)

    grouped: dict[str, list] = {
        "Pinned": [],
        "Today": [],
        "Yesterday": [],
        "This Week": [],
        "This Month": [],
        "Older": [],
    }

    for conv in conversations:
        # Pinned conversations go to top
        if conv.get("is_pinned"):
            grouped["Pinned"].append(conv)
            continue

        conv_date = conv["updated_at"].date() if conv["updated_at"] else today

        if conv_date == today:
            grouped["Today"].append(conv)
        elif conv_date == yesterday:
            grouped["Yesterday"].append(conv)
        elif conv_date > week_ago:
            grouped["This Week"].append(conv)
        elif conv_date > month_ago:
            grouped["This Month"].append(conv)
        else:
            grouped["Older"].append(conv)

    return grouped


def render_history_sidebar():
    """Render the conversation history in the sidebar with search."""
    history = get_history_store()

    # Search box (stays fixed at top)
    search_query = st.sidebar.text_input(
        "Search chats",
        placeholder="Search...",
        key="chat_search",
        label_visibility="collapsed",
    )

    # Pagination settings
    page_size = 30

    # Handle search
    if search_query and search_query.strip():
        conversations = history.search_conversations(search_query.strip(), limit=20)
        if not conversations:
            st.sidebar.caption(f"No results for '{search_query}'")
            return
        st.sidebar.caption(f"Results for '{search_query}'")
    else:
        # Get paginated list
        offset = st.session_state.get("history_offset", 0)
        conversations = history.list_conversations(limit=page_size, offset=offset)

        if not conversations and offset == 0:
            st.sidebar.caption("No conversation history yet")
            return

    # Group by date
    grouped = _group_conversations_by_date(conversations)

    # Create a scrollable container for the conversation list
    # Height of 400px provides good visibility without pushing other content off screen
    with st.sidebar.container(height=400, border=False):
        # Render each group inside the scrollable container
        for group_name, convs in grouped.items():
            if not convs:
                continue

            # Group header with icon
            if group_name == "Pinned":
                st.markdown(f"**{group_name}**")
            else:
                st.caption(group_name)

            for conv in convs:
                _render_conversation_item_in_container(conv, history)

        # Load more button (only when not searching) - inside scrollable area
        if not search_query:
            total = history.get_conversation_count()
            shown = st.session_state.get("history_offset", 0) + len(conversations)
            if shown < total:
                if st.button(
                    f"Load more ({total - shown} remaining)",
                    use_container_width=True,
                    key="load_more_history",
                ):
                    st.session_state.history_offset = (
                        st.session_state.get("history_offset", 0) + page_size
                    )
                    st.rerun()


def _render_conversation_item(conv: dict, history):
    """Render a single conversation item in the sidebar (legacy, used outside container)."""
    conv_id = conv["id"]
    is_pinned = conv.get("is_pinned", False)

    # Truncate title
    title = conv["title"]
    if len(title) > 35:
        title = title[:35] + "..."

    # Add pin indicator
    if is_pinned:
        title = f"* {title}"

    # Main row with conversation button
    col1, col2 = st.sidebar.columns([5, 1])

    with col1:
        if st.button(
            title,
            key=f"conv_{conv_id}",
            use_container_width=True,
            help=f"{conv['message_count']} msgs | {conv['model']}",
        ):
            load_conversation(conv_id)

    with col2:
        # Context menu using popover
        with st.popover(":", help="Options"):
            if is_pinned:
                if st.button("Unpin", key=f"unpin_{conv_id}", use_container_width=True):
                    history.pin_conversation(conv_id, False)
                    st.rerun()
            else:
                if st.button("Pin", key=f"pin_{conv_id}", use_container_width=True):
                    history.pin_conversation(conv_id, True)
                    st.rerun()

            if st.button("Archive", key=f"archive_{conv_id}", use_container_width=True):
                history.archive_conversation(conv_id, True)
                st.rerun()

            if st.button(
                "Delete", key=f"del_{conv_id}", use_container_width=True, type="primary"
            ):
                history.delete_conversation(conv_id)
                st.rerun()


def _render_conversation_item_in_container(conv: dict, history):
    """Render a single conversation item inside a container (uses st. not st.sidebar)."""
    conv_id = conv["id"]
    is_pinned = conv.get("is_pinned", False)

    # Truncate title
    title = conv["title"]
    if len(title) > 35:
        title = title[:35] + "..."

    # Add pin indicator
    if is_pinned:
        title = f"* {title}"

    # Main row with conversation button (use st.columns, not st.sidebar.columns)
    col1, col2 = st.columns([5, 1])

    with col1:
        if st.button(
            title,
            key=f"conv_{conv_id}",
            use_container_width=True,
            help=f"{conv['message_count']} msgs | {conv['model']}",
        ):
            load_conversation(conv_id)

    with col2:
        # Context menu using popover
        with st.popover(":", help="Options"):
            if is_pinned:
                if st.button("Unpin", key=f"unpin_{conv_id}", use_container_width=True):
                    history.pin_conversation(conv_id, False)
                    st.rerun()
            else:
                if st.button("Pin", key=f"pin_{conv_id}", use_container_width=True):
                    history.pin_conversation(conv_id, True)
                    st.rerun()

            if st.button("Archive", key=f"archive_{conv_id}", use_container_width=True):
                history.archive_conversation(conv_id, True)
                st.rerun()

            if st.button(
                "Delete", key=f"del_{conv_id}", use_container_width=True, type="primary"
            ):
                history.delete_conversation(conv_id)
                st.rerun()


def load_conversation(conversation_id: str):
    """Load a conversation from history into session state."""
    history = get_history_store()
    conv = history.get_conversation(conversation_id)

    if conv:
        st.session_state.thread_id = conv.id
        st.session_state.messages = [
            {
                "role": msg.role,
                "content": msg.content,
                "tool_calls": msg.tool_calls,
            }
            for msg in conv.messages
        ]
        st.rerun()


# ============================================================================
# Sidebar Configuration
# ============================================================================


def render_sidebar() -> Config:
    """Render sidebar configuration and return Config object."""

    # Logo and title
    if os.path.exists("assets/pravha.png"):
        st.sidebar.image("assets/pravha.png", width=150)
    st.sidebar.title("Pravah")
    st.sidebar.caption("AI Search Engine")

    st.sidebar.markdown("---")

    # Model Configuration
    st.sidebar.subheader("Model Settings")

    config = Config()

    from pravah.pricing import get_model_cost_tier, get_model_pricing

    # Model selection
    config.model = st.sidebar.selectbox(
        "Model",
        options=config.available_models,
        index=config.available_models.index(config.model)
        if config.model in config.available_models
        else 0,
        help="Select the LLM to use. Different models have different capabilities and costs.",
    )

    # Show cost tier for selected model
    tier = get_model_cost_tier(config.model)
    pricing = get_model_pricing(config.model)
    if pricing:
        st.sidebar.caption(
            f"Cost: {tier} (${pricing.input_per_million:.2f}/${pricing.output_per_million:.2f} per 1M tokens)"
        )
    else:
        st.sidebar.caption(f"Cost: {tier}")

    # Custom model option
    custom_model = st.sidebar.text_input(
        "Custom Model (optional)",
        placeholder="e.g., together/meta-llama/Llama-3-70b",
        help="Enter a custom LiteLLM model string",
    )
    if custom_model:
        config.model = custom_model

    # Temperature
    config.temperature = st.sidebar.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.3,
        step=0.1,
        help="Higher values make output more random, lower values more deterministic.",
    )

    # API Key management
    render_api_key_setup()

    # File Upload section
    st.sidebar.markdown("---")
    st.sidebar.subheader("📁 Upload Documents")

    # Initialize processed files tracker in session state
    if "processed_files" not in st.session_state:
        st.session_state.processed_files = set()

    uploaded_files = st.sidebar.file_uploader(
        "Upload files to search",
        type=["pdf", "docx", "pptx", "xlsx", "txt", "md", "csv", "html", "json"],
        accept_multiple_files=True,
        key=f"file_uploader_{st.session_state.thread_id}",
        help="Upload documents to search within this conversation. Supports PDF, DOCX, PPTX, and more.",
    )

    if uploaded_files:
        from pravah.uploads import get_upload_manager
        import tempfile
        from pathlib import Path

        manager = get_upload_manager()
        new_uploads = 0

        for uploaded_file in uploaded_files:
            # Create unique key for this file+conversation
            file_key = f"{st.session_state.thread_id}:{uploaded_file.name}:{uploaded_file.size}"

            # Skip if already processed
            if file_key in st.session_state.processed_files:
                continue

            # Save to temp file and process
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp.flush()

                result = manager.process_file(
                    file_path=Path(tmp.name),
                    filename=uploaded_file.name,
                    conversation_id=st.session_state.thread_id,
                )

                if result["success"]:
                    st.session_state.processed_files.add(file_key)
                    new_uploads += 1

        if new_uploads > 0:
            st.sidebar.success(f"✓ Indexed {new_uploads} file(s)")

    # Show uploaded files for this conversation
    if "thread_id" in st.session_state:
        from pravah.uploads import get_upload_manager
        manager = get_upload_manager()
        uploads = manager.list_uploads(st.session_state.thread_id)
        if uploads:
            with st.sidebar.expander(f"📄 Indexed Files ({len(uploads)})", expanded=False):
                for u in uploads:
                    st.caption(f"• {u['filename']} ({u['chunk_count']} chunks)")

    # Session management
    st.sidebar.markdown("---")
    st.sidebar.subheader("Conversations")

    # New chat button
    if st.sidebar.button("New Chat", use_container_width=True, type="primary"):
        st.session_state.messages = []
        st.session_state.thread_id = str(uuid.uuid4())
        st.rerun()

    # History viewer
    render_history_sidebar()

    # Debug toggle
    st.sidebar.markdown("---")
    st.session_state.show_debug = st.sidebar.checkbox(
        "Show Debug Info",
        value=st.session_state.get("show_debug", False),
        help="Show token usage, latency, and cost estimates",
    )

    # Keyboard shortcuts help
    with st.sidebar.expander("Keyboard Shortcuts"):
        st.markdown("""
        - **Cmd/Ctrl + K** - Search chats
        - **Cmd/Ctrl + N** - New chat
        - **/** - Focus chat input
        - **Esc** - Focus chat input
        """)

    # Store config in session state
    st.session_state.config = config

    return config


# ============================================================================
# Agent Execution
# ============================================================================


async def run_agent_stream(
    query: str,
    config: Config,
    thread_id: str,
    history_messages: list[dict] | None = None,
):
    """Run the agent and yield streaming events.

    Args:
        query: The current user query.
        config: Application configuration.
        thread_id: Unique conversation identifier.
        history_messages: Previous messages in the conversation (for context).
    """
    from pravah.agent import app
    from langchain_core.messages import HumanMessage, AIMessage

    agent_config = {
        "configurable": {
            "thread_id": thread_id,
        },
        "recursion_limit": 25,
    }

    # Build message list with conversation history
    messages = []

    # Include previous messages for context
    if history_messages:
        for msg in history_messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "user":
                messages.append(HumanMessage(content=content))
            elif role == "assistant" and content:
                messages.append(AIMessage(content=content))

    # Add the current query
    messages.append(HumanMessage(content=query))

    inputs = {
        "messages": messages,
        "model": config.model,
        "temperature": config.temperature,
        "iteration_count": 0,
        "documents": [],
    }

    async for event in app.astream_events(inputs, config=agent_config, version="v2"):
        yield event


def process_agent_response(
    query: str,
    config: Config,
    thread_id: str,
    history_messages: list[dict] | None = None,
):
    """Process agent response with streaming and tool visibility.

    Args:
        query: The current user query.
        config: Application configuration.
        thread_id: Unique conversation identifier.
        history_messages: Previous messages for context.
    """
    # Create containers for dynamic content
    status_container = st.status("Thinking...", expanded=True)
    response_container = st.empty()

    full_response = ""
    tool_calls = []
    current_tool = None
    total_tokens_in = 0
    total_tokens_out = 0

    async def run():
        nonlocal \
            full_response, \
            tool_calls, \
            current_tool, \
            total_tokens_in, \
            total_tokens_out

        async for event in run_agent_stream(query, config, thread_id, history_messages):
            event_type = event.get("event", "")

            # Handle different event types
            if event_type == "on_chat_model_stream":
                # Streaming LLM output (if model supports streaming)
                data = event.get("data", {})
                chunk = data.get("chunk")
                if chunk:
                    content = getattr(chunk, "content", "")
                    if content:
                        full_response += content
                        response_container.markdown(full_response + "▌")

            elif event_type == "on_chat_model_end":
                # Capture response from chat model
                # For streaming: this fires after stream completes
                # For non-streaming: this is where we get the response
                data = event.get("data", {})
                output = data.get("output")
                if output:
                    content = ""
                    # Handle AIMessage or dict output
                    if hasattr(output, "content") and output.content:
                        content = output.content
                    elif isinstance(output, dict):
                        content = output.get("content", "")

                    # Only overwrite if we don't have streaming content yet
                    # or if this is a new response after tool calls
                    if content and (
                        not full_response or len(content) > len(full_response)
                    ):
                        full_response = content
                        response_container.markdown(full_response)

                    # Extract token usage from response_metadata
                    if hasattr(output, "response_metadata"):
                        metadata = output.response_metadata
                        if isinstance(metadata, dict):
                            usage = metadata.get("token_usage") or metadata.get("usage")
                            if usage:
                                # Handle both dict and object-style usage (litellm returns objects)
                                if hasattr(usage, "prompt_tokens"):
                                    total_tokens_in += (
                                        getattr(usage, "prompt_tokens", 0) or 0
                                    )
                                    total_tokens_out += (
                                        getattr(usage, "completion_tokens", 0) or 0
                                    )
                                elif isinstance(usage, dict):
                                    total_tokens_in += (
                                        usage.get("prompt_tokens", 0)
                                        or usage.get("input_tokens", 0)
                                        or 0
                                    )
                                    total_tokens_out += (
                                        usage.get("completion_tokens", 0)
                                        or usage.get("output_tokens", 0)
                                        or 0
                                    )

            elif event_type == "on_tool_start":
                # Tool is starting
                tool_name = event.get("name", "unknown")
                tool_input = event.get("data", {}).get("input", {})
                current_tool = {
                    "name": tool_name,
                    "input": tool_input,
                    "start_time": datetime.now().isoformat(),
                }

                with status_container:
                    st.write(f"**{tool_name}**")
                    if isinstance(tool_input, dict):
                        for k, v in tool_input.items():
                            st.text(f"  {k}: {str(v)[:100]}...")
                    else:
                        st.text(f"  Input: {str(tool_input)[:100]}...")

            elif event_type == "on_tool_end":
                # Tool finished
                tool_name = event.get("name", "unknown")
                output = event.get("data", {}).get("output", "")

                if current_tool:
                    current_tool["output"] = str(output)[:500]
                    current_tool["end_time"] = datetime.now().isoformat()
                    tool_calls.append(current_tool)

                with status_container:
                    st.write(f"Done: {tool_name}")

    # Run the async function
    asyncio.run(run())

    # Update status to complete
    status_container.update(label="Complete", state="complete", expanded=False)

    # Display final response
    response_container.markdown(full_response)

    return full_response, tool_calls, total_tokens_in, total_tokens_out


# ============================================================================
# Debug Panel
# ============================================================================


def render_debug_panel(thread_id: str):
    """Render debug panel with conversation stats."""
    if not st.session_state.get("show_debug", False):
        return

    from pravah.pricing import calculate_cost, format_cost, get_model_cost_tier

    history = get_history_store()
    stats = history.get_conversation_stats(thread_id)
    config = st.session_state.get("config", Config())

    # Calculate cost from tokens if we have them
    tokens_in = stats["total_tokens_in"]
    tokens_out = stats["total_tokens_out"]

    if tokens_in and tokens_out:
        estimated_cost = calculate_cost(config.model, tokens_in, tokens_out)
    else:
        estimated_cost = None

    with st.expander("Debug Info", expanded=False):
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Messages", stats["message_count"])
        with col2:
            total_tokens = (tokens_in or 0) + (tokens_out or 0)
            st.metric("Total Tokens", f"{total_tokens:,}" if total_tokens else "N/A")
        with col3:
            st.metric("In / Out", f"{tokens_in or 0:,} / {tokens_out or 0:,}")
        with col4:
            latency = stats["avg_latency_ms"]
            st.metric("Avg Latency", f"{latency:,.0f}ms" if latency else "N/A")
        with col5:
            cost_str = format_cost(estimated_cost)
            tier = get_model_cost_tier(config.model)
            st.metric("Est. Cost", cost_str, delta=tier, delta_color="off")


# ============================================================================
# Chat Interface
# ============================================================================


def render_chat_message(message: dict):
    """Render a single chat message."""
    role = message["role"]
    content = message["content"]

    with st.chat_message(role):
        st.markdown(content)

        # Show tool calls if present
        if "tool_calls" in message and message["tool_calls"]:
            with st.expander("Tool calls", expanded=False):
                for tool in message["tool_calls"]:
                    st.text(f"Tool: {tool['name']}")
                    if "input" in tool:
                        st.json(tool["input"])


def render_chat_history():
    """Render the chat history."""
    messages = st.session_state.get("messages", [])

    for message in messages:
        render_chat_message(message)


# ============================================================================
# Main Application
# ============================================================================


def main():
    """Main application entry point."""

    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(uuid.uuid4())

    # Render sidebar and get config
    config = render_sidebar()

    # Main content area
    st.title("Pravah")
    st.caption("AI-powered search that synthesizes information from multiple sources")

    # Check API keys
    required_keys = get_required_api_keys(config.model)
    key_status = check_api_keys(required_keys)

    if not all(key_status.values()):
        missing = [k for k, v in key_status.items() if not v]
        st.warning(
            f"Please configure the following API keys in the sidebar: {', '.join(missing)}"
        )
        st.stop()

    # Debug panel (shows conversation stats when enabled)
    render_debug_panel(st.session_state.thread_id)

    # Render chat history
    render_chat_history()

    # Chat input
    if prompt := st.chat_input("Ask me anything..."):
        history = get_history_store()
        thread_id = st.session_state.thread_id

        # Add user message to session and DB
        st.session_state.messages.append(
            {
                "role": "user",
                "content": prompt,
            }
        )
        history.add_message(thread_id, "user", prompt)
        history.update_conversation_model(thread_id, config.model)

        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            try:
                import time

                start_time = time.time()

                # Pass previous messages for conversation context
                # Exclude the just-added user message (it's in the query)
                previous_messages = (
                    st.session_state.messages[:-1]
                    if len(st.session_state.messages) > 1
                    else None
                )

                response, tool_calls, tokens_in, tokens_out = process_agent_response(
                    prompt,
                    config,
                    thread_id,
                    history_messages=previous_messages,
                )

                latency_ms = int((time.time() - start_time) * 1000)

                # Add assistant message to session and DB
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": response,
                        "tool_calls": tool_calls,
                    }
                )
                history.add_message(
                    thread_id,
                    "assistant",
                    response,
                    tool_calls=tool_calls,
                    latency_ms=latency_ms,
                    tokens_in=tokens_in if tokens_in else None,
                    tokens_out=tokens_out if tokens_out else None,
                )

            except Exception as e:
                error_msg = f"An error occurred: {str(e)}"
                st.error(error_msg)
                error_response = f"I encountered an error: {str(e)}. Please try again."
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": error_response,
                    }
                )
                history.add_message(thread_id, "assistant", error_response)

    # Welcome message if no messages
    if not st.session_state.messages:
        from pravah.welcome import generate_welcome_with_fallback, get_api_status

        # Generate dynamic welcome
        welcome = generate_welcome_with_fallback()
        api_status = get_api_status(["TVLY_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY"])

        st.markdown(f"""
        ### {welcome.greeting}
        
        An AI search engine that finds and synthesizes information from the web.
        
        **Capabilities:**
        - Search the web for current information
        - Read and summarize articles
        - Perform calculations
        - Cite all sources
        
        **💡 Tip:** {welcome.tip}
        
        **Try asking:**
        - "What are the new features in Python 3.13?"
        - "Compare React and Vue.js for web development"
        - "Latest news about AI regulation"
        
        Type your question below to get started.
        """)


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":
    main()
