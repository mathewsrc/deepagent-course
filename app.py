import gradio as gr
from typing import List, Tuple, Any

# Import the graph entrypoint from main.py. We'll call the compiled graph which
# runs the `call_agent` node internally.
from main import graph

# Helper to convert Gradio chat history (list of (user, bot) tuples) into the
# message list expected by main.call_agent
def build_state_messages(user_message: str, chat_history: List[Tuple[str, str]]) -> List[dict]:
    messages = []
    for u, b in chat_history:
        # preserve order: user then assistant
        messages.append({"role": "user", "content": u})
        messages.append({"role": "assistant", "content": b})
    # append the new user message last
    messages.append({"role": "user", "content": user_message})
    return messages


def send_message(user_message: str, chat_history: List[Tuple[str, str]]):
    """Send a user message to the agent and return updated chat history and empty input."""
    if not user_message:
        return chat_history, ""

    state = {
        "messages": build_state_messages(user_message, chat_history),
        # default role; could be exposed in UI later
        "role": "Dev",
    }

    try:
        resp = graph(state)
        # The graph returns what the `call_agent` node produces. We expect a dict
        # with a `messages` entry (string). Coerce to string safely.
        bot_content = resp.get("messages") if isinstance(resp.get("messages"), str) else str(resp.get("messages"))
    except Exception as e:
        bot_content = f"Error calling agent: {e}"

    chat_history = chat_history + [(user_message, bot_content)]
    return chat_history, ""


def evaluate_openapi(openapi_text: str, chat_history: List[Tuple[str, str]]):
    """Send an "evaluate OpenAPI" request to the agent using the text area content."""
    if not openapi_text or openapi_text.strip() == "":
        return chat_history

    prompt = "Please evaluate the following OpenAPI specification and provide a concise review, potential issues, and suggestions for improvement:\n\n" + openapi_text

    state = {
        "messages": build_state_messages(prompt, chat_history),
        "role": "Dev",
    }

    try:
        resp = graph(state)
        bot_content = resp.get("messages") if isinstance(resp.get("messages"), str) else str(resp.get("messages"))
    except Exception as e:
        bot_content = f"Error calling agent: {e}"

    chat_history = chat_history + [("[Evaluate OpenAPI]", bot_content)]
    return chat_history


def next_step(chat_history: List[Tuple[str, str]]):
    """Send a lightweight 'next' message to the agent to advance the conversation or workflow."""
    prompt = "Next"
    state = {
        "messages": build_state_messages(prompt, chat_history),
        "role": "Dev",
    }

    try:
        resp = graph(state)
        bot_content = resp.get("messages") if isinstance(resp.get("messages"), str) else str(resp.get("messages"))
    except Exception as e:
        bot_content = f"Error calling agent: {e}"

    chat_history = chat_history + [("[Next]", bot_content)]
    return chat_history


with gr.Blocks(title="LangGraph Chat App") as demo:
    gr.Markdown("# LangGraph Chat · Minimal Hugging Face App")

    with gr.Row():
        chatbot = gr.Chatbot(label="Conversation")

    with gr.Row():
        with gr.Column(scale=3):
            user_input = gr.Textbox(lines=2, placeholder="Type a message and press Enter or click Send")
        with gr.Column(scale=1):
            send_btn = gr.Button("Send")
            next_btn = gr.Button("Next")

    gr.Markdown("---")
    gr.Markdown("## Evaluate OpenAPI")
    openapi_text = gr.Textbox(lines=10, placeholder="Paste OpenAPI spec (JSON/YAML) here")
    eval_btn = gr.Button("Evaluate OpenAPI")

    # hidden state stored in the Chatbot component. We'll wire callbacks to update it.

    # send message on button click or when Enter pressed
    send_btn.click(fn=send_message, inputs=[user_input, chatbot], outputs=[chatbot, user_input])
    user_input.submit(fn=send_message, inputs=[user_input, chatbot], outputs=[chatbot, user_input])

    # evaluate openapi
    eval_btn.click(fn=evaluate_openapi, inputs=[openapi_text, chatbot], outputs=[chatbot])

    # next button
    next_btn.click(fn=next_step, inputs=[chatbot], outputs=[chatbot])


demo.launch()
