import os
import json
from langchain_groq import ChatGroq
from langchain_core.runnables import RunnableConfig
from langchain.agents import create_agent, AgentState
from langchain.tools import tool
from langchain.agents.middleware import wrap_tool_call
from langchain_core.messages import ToolMessage
from langchain.agents.structured_output import ToolStrategy, ProviderStrategy

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from langgraph.graph.message import add_messages

from typing_extensions import TypedDict, Annotated
from typing import Dict, List, Any
import uuid

class State(AgentState):
    messages: Annotated[List, add_messages, "List of messages in the conversation"]
    llm_calls: Annotated[int, "Number of LLM calls made so far"]
    response_metadata: Annotated[Dict[str, str], "Metadata about the LLM responses"]
    role: Annotated[str, "User role, i.e PM, Dev, etc"]
    model_name: Annotated[str, "Name of the model that produced the response"]
    usage_metadata: Annotated[Dict[str, Any], "Token usage and timing metadata"]
    tool_calls: Annotated[List[Dict[str, Any]], "List of tool call records (name, args, id)"]
    tool_message: Annotated[str, "Content returned by tool messages (if any)"]


class ToolStructure(TypedDict):
    function_name: str
    
class Context(TypedDict):
    user_role: str
    
@tool
def search(query: str) -> str:
    """Search for information.
    query: The search query.
    Returns: The search results.
    """
    return f"Results for: {query}"

@tool
def get_weather(location: str) -> str:
    """Get weather information for a location.
    location: The location to get the weather for.
    Returns: The weather information.
    """
    # In a real tool this would call a weather API. Keep the docstring explicit so
    # the model knows when to call it.
    return f"Weather in {location}: Sunny, 72°F"

@wrap_tool_call
def handle_tool_errors(request, handler):
    """Handle tool execution errors with custom messages."""
    try:
        return handler(request)
    except Exception as e:
        # Return a custom error message to the model
        return ToolMessage(
            content=f"Tool error: Please check your input and try again. ({str(e)})",
            tool_call_id=request.tool_call["id"]
        )

def create_llm() -> ChatGroq:
    print("Creating LLM...")
    api_key = "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"
    if not api_key:
        raise RuntimeError("GROQ_API_KEY environment variable is not set. Please set it before running.")

    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0,
        max_tokens=1024,
        api_key=api_key,
    )
    return llm


def call_agent(state: State):
    print("Calling agent...")
    llm = create_llm()
    agent = create_agent(
        name="Assistant",
        model=llm,
        system_prompt=(
            "You are a helpful assistant. When the user asks for factual or external information, "
            "use the provided tools instead of inventing facts. Use `search(query)` for general "
            "information lookup and `get_weather(location)` for weather requests. When you call a tool, "
            "provide the minimal, precise arguments the tool needs. Return the tool output to the user."
        ),
        tools=[search, get_weather],
        middleware=[handle_tool_errors],
        context_schema=Context,
        # response_format=ToolStrategy(ToolStructure)
    )
    
    # Normalize messages to the expected shape: a list of {role, content} dicts.
    if isinstance(state.get("messages"), list):
        messages = state["messages"]
    else:
        # If caller passed a single string, wrap it into a proper message list.
        messages = [{"role": "user", "content": str(state.get("messages", ""))}]

    result = agent.invoke({"messages": messages}, context={"user_role": state["role"]})

    # result["messages"] items may be dicts or message objects depending on runtime; handle both.
    latest_message = result.get("messages", [])[-1] if result.get("messages") else None
    print("Agent result: ", json.dumps(result, default=str, indent=2))
    
    if "structured_response" in result:
        print("Structure response: ", result["structured_response"])
    content = latest_message.content
    metadata = latest_message.response_metadata

    return {"messages": content, "response_metadata": metadata}

def graph():
    print("Compiling graph...")
    
    chatbot = StateGraph(state_schema=State)
    chatbot.add_node("chat", call_agent)
    chatbot.add_edge(START, "chat")
    chatbot.add_edge("chat", END)
    
    chatbot = chatbot.compile(name="chatbot",
                    checkpointer=InMemorySaver(),
                    store=InMemoryStore())
    
    config = RunnableConfig(
            {
            "configurable": {
                "thread_id": uuid.uuid4()
                },
            "recursion_limit": 50,
            "run_id": uuid.uuid4(),
            "run_name": "chatbot_run"
            }
        )
    
    # Pass messages as a list of message dicts per the State schema.
    result = chatbot.invoke(
        {
            "messages": [{"role": "user", "content": "What is the weather of vancouver?"}],
            "role": "Dev",
        },
        config=config,
    )
    print("Final result: ", result)
    


if __name__ == "__main__":
    graph()
