"""
Pravah v2 Agent - LangGraph ReAct Agent with LiteLLM

Features:
- Official langchain-litellm integration for model flexibility
- SQLite checkpointer for conversation persistence
- Loop control with iteration limits
- Proper system prompt injection
- Message trimming for context management
"""

from typing import TypedDict, Annotated, Literal
import operator
import os
import sqlite3

from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
    ToolMessage,
    RemoveMessage,
)
from langchain_litellm import ChatLiteLLM
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

# Try to import AsyncSqliteSaver for persistent storage with async support
# Falls back to MemorySaver if not available
_USE_SQLITE_ASYNC = False
AsyncSqliteSaver = None

try:
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver as _AsyncSqliteSaver

    AsyncSqliteSaver = _AsyncSqliteSaver
    _USE_SQLITE_ASYNC = True
except ImportError:
    pass

if AsyncSqliteSaver is None:
    try:
        from langgraph_checkpoint_sqlite.aio import (
            AsyncSqliteSaver as _AsyncSqliteSaver,
        )

        AsyncSqliteSaver = _AsyncSqliteSaver
        _USE_SQLITE_ASYNC = True
    except ImportError:
        pass

# Import RemainingSteps if available (newer langgraph versions)
try:
    from langgraph.managed import RemainingSteps
except ImportError:
    # Fallback: create a simple int type alias
    RemainingSteps = int  # type: ignore

from pravah.tools import (
    web_search,
    gemini_search,
    fetch_page,
    search_memory,
    read_page_chunk,
    calculate,
)
from pravah.prompts import get_agent_system_prompt
from pravah.memory import set_current_thread_id


# ============================================================================
# State Definition
# ============================================================================


class AgentState(TypedDict):
    """State for the Pravah agent.

    Attributes:
        messages: Conversation history (appends via operator.add)
        remaining_steps: Auto-managed by LangGraph for loop control
        iteration_count: Manual counter for additional loop control
        documents: Fetched documents stored separately from messages
        model: Current model being used (from UI config)
        temperature: Current temperature setting
    """

    messages: Annotated[list[BaseMessage], operator.add]
    remaining_steps: RemainingSteps
    iteration_count: int
    documents: list[dict]
    model: str
    temperature: float


# ============================================================================
# Tools
# ============================================================================

tools = [
    web_search,
    gemini_search,
    fetch_page,
    read_page_chunk,
    search_memory,
    calculate,
]


# ============================================================================
# Model Factory
# ============================================================================


def get_chat_model(model: str = "openai/gpt-4o", temperature: float = 0.0):
    """Create a chat model with tool calling support.

    Uses the official langchain-litellm ChatLiteLLM for maximum compatibility.
    Streaming is enabled for real-time response display.
    """
    chat_model = ChatLiteLLM(
        model=model,
        temperature=temperature,
        max_tokens=4096,
        streaming=True,  # Enable streaming for real-time responses
    )
    return chat_model.bind_tools(tools)


# ============================================================================
# Agent Node
# ============================================================================


def agent_node(state: AgentState) -> dict:
    """Main agent node that invokes the LLM.

    - Injects system prompt
    - Trims messages if context is too long
    - Tracks iteration count
    - Handles wrap-up when running low on steps
    """
    # Get model configuration from state or use defaults
    model_name = state.get("model", "openai/gpt-4o")
    temperature = state.get("temperature", 0.0)

    # Get the model
    model = get_chat_model(model_name, temperature)

    # Build messages with system prompt
    system_prompt = get_agent_system_prompt()
    messages = [SystemMessage(content=system_prompt)] + list(state["messages"])

    # Check remaining steps and add wrap-up instruction if low
    remaining = state.get("remaining_steps", 10)
    if remaining <= 3:
        wrap_up = HumanMessage(
            content="[System: You are running low on steps. Provide your best answer now based on what you've found. Do not make more tool calls.]"
        )
        messages.append(wrap_up)

    # Trim messages if too long (keep system + recent)
    # Simple approach: limit to last N messages after system
    MAX_MESSAGES = 20
    if len(messages) > MAX_MESSAGES:
        # Keep system message + last (MAX_MESSAGES - 1) messages
        messages = [messages[0]] + messages[-(MAX_MESSAGES - 1) :]

    # Invoke the model
    response = model.invoke(messages)

    # Update iteration count
    current_count = state.get("iteration_count", 0) + 1

    return {
        "messages": [response],
        "iteration_count": current_count,
    }


# ============================================================================
# Routing
# ============================================================================


def should_continue(state: AgentState) -> Literal["tools", "__end__"]:
    """Determine whether to continue to tools or end.

    Ends if:
    - No tool calls in last message
    - Remaining steps <= 1
    - Iteration count >= 10 (hard limit)
    """
    # Check remaining steps
    remaining = state.get("remaining_steps", 10)
    if remaining <= 1:
        return END

    # Check iteration limit
    if state.get("iteration_count", 0) >= 10:
        return END

    # Check for tool calls
    messages = state.get("messages", [])
    if not messages:
        return END

    last_message = messages[-1]

    # Check if the last message has tool_calls
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"

    return END


# ============================================================================
# Tool Node with Thread Context
# ============================================================================


def create_tool_node_with_context(tools_list):
    """Create a tool node that sets thread context before execution.

    This allows tools like search_memory to access the current thread's data.
    """
    from langchain_core.runnables import RunnableConfig

    base_tool_node = ToolNode(tools_list)

    def tool_node_with_context(state: AgentState, config: RunnableConfig) -> dict:
        # Extract thread_id from config and set it for tools
        configurable = config.get("configurable", {}) if config else {}
        thread_id = configurable.get("thread_id", "") if configurable else ""
        if thread_id:
            set_current_thread_id(thread_id)

        # Call the base tool node
        return base_tool_node.invoke(state, config)

    return tool_node_with_context


def handle_tool_error(state: AgentState) -> dict:
    """Handle errors from tool execution by informing the agent."""
    messages = state.get("messages", [])

    # Find the last error
    error_message = "An error occurred during tool execution."
    for msg in reversed(messages):
        if isinstance(msg, ToolMessage) and "error" in str(msg.content).lower():
            error_message = msg.content
            break

    return {
        "messages": [
            AIMessage(
                content=f"I encountered an error: {error_message}. Let me try a different approach."
            )
        ]
    }


# ============================================================================
# Graph Construction
# ============================================================================


def create_agent_graph(checkpointer=None):
    """Create and compile the agent graph.

    Args:
        checkpointer: Optional checkpointer for persistence.
                     If None, uses MemorySaver (async-compatible).

    Returns:
        Compiled graph ready for invocation.
    """
    # Create the graph
    builder = StateGraph(AgentState)

    # Add nodes
    builder.add_node("agent", agent_node)
    builder.add_node("tools", create_tool_node_with_context(tools))

    # Add edges
    builder.add_edge(START, "agent")
    builder.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            END: END,
        },
    )
    builder.add_edge("tools", "agent")

    # Create checkpointer if not provided
    # Use MemorySaver by default - it supports both sync and async
    # AsyncSqliteSaver requires special handling with context managers
    if checkpointer is None:
        checkpointer = MemorySaver()

    # Compile with checkpointer
    return builder.compile(checkpointer=checkpointer)


def get_async_checkpointer():
    """Get an async-compatible SQLite checkpointer for use in async contexts.

    Usage:
        async with get_async_checkpointer() as checkpointer:
            graph = create_agent_graph(checkpointer=checkpointer)
            # use graph...

    Returns:
        AsyncSqliteSaver context manager, or MemorySaver if not available.
    """
    if _USE_SQLITE_ASYNC and AsyncSqliteSaver is not None:
        db_path = os.path.join(os.path.dirname(__file__), "..", "checkpoints.db")
        db_path = os.path.abspath(db_path)
        return AsyncSqliteSaver.from_conn_string(db_path)
    else:
        # Return a simple context manager wrapper for MemorySaver
        from contextlib import asynccontextmanager

        @asynccontextmanager
        async def _memory_saver_cm():
            yield MemorySaver()

        return _memory_saver_cm()


# ============================================================================
# Default Graph Instance
# ============================================================================


def _create_default_checkpointer():
    """Create the default checkpointer.

    Uses MemorySaver which supports both sync and async operations.
    For persistent async storage, use get_async_checkpointer() context manager.
    """
    return MemorySaver()


_checkpointer = _create_default_checkpointer()

# Compile the graph with MemorySaver (async-compatible)
app = create_agent_graph(checkpointer=_checkpointer)


# ============================================================================
# Convenience Functions
# ============================================================================


async def run_agent(
    query: str,
    thread_id: str,
    model: str = "openai/gpt-4o",
    temperature: float = 0.0,
):
    """Run the agent with a query.

    Args:
        query: User's query
        thread_id: Unique thread ID for conversation persistence
        model: LiteLLM model string (e.g., "openai/gpt-4o", "anthropic/claude-3-sonnet")
        temperature: Model temperature

    Yields:
        Events from the agent execution (for streaming)
    """
    config = {
        "configurable": {
            "thread_id": thread_id,
        },
        "recursion_limit": 25,
    }

    inputs = {
        "messages": [HumanMessage(content=query)],
        "model": model,
        "temperature": temperature,
        "iteration_count": 0,
        "documents": [],
    }

    async for event in app.astream_events(inputs, config=config, version="v2"):
        yield event


def run_agent_sync(
    query: str,
    thread_id: str,
    model: str = "openai/gpt-4o",
    temperature: float = 0.0,
) -> dict:
    """Run the agent synchronously.

    Args:
        query: User's query
        thread_id: Unique thread ID for conversation persistence
        model: LiteLLM model string
        temperature: Model temperature

    Returns:
        Final state after agent execution
    """
    config = {
        "configurable": {
            "thread_id": thread_id,
        },
        "recursion_limit": 25,
    }

    inputs = {
        "messages": [HumanMessage(content=query)],
        "model": model,
        "temperature": temperature,
        "iteration_count": 0,
        "documents": [],
    }

    return app.invoke(inputs, config=config)


def get_conversation_history(thread_id: str) -> list[BaseMessage]:
    """Get the conversation history for a thread.

    Args:
        thread_id: The thread ID to look up

    Returns:
        List of messages in the conversation
    """
    config = {"configurable": {"thread_id": thread_id}}
    state = app.get_state(config)
    return state.values.get("messages", []) if state.values else []
