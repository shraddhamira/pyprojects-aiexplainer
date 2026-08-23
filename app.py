import streamlit as st
from openai import OpenAI
import anthropic
from google import genai

# App configuration
st.set_page_config(page_title="AI Concept Tutor", page_icon="🤖", layout="wide")
st.title("🤖 AI Concept Tutor")

# 1. Sidebar Setup
with st.sidebar:
    st.header("Configuration")
    provider = st.selectbox(
        "Choose Provider",
        ["Google Gemini", "OpenAI", "Anthropic Claude"]
    )
    api_key = st.text_input(f"Enter your {provider} API Key", type="password")

    model_options = {
        "Google Gemini": [
            "gemini-3.1-pro-preview",
            "gemini-3.7-flash",
            "gemini-2.5-flash",
        ],
        "OpenAI": [
            "gpt-4o",
            "gpt-4o-mini",
        ],
        "Anthropic Claude": [
            "claude-3-7-sonnet-20250219",
            "claude-3-5-haiku-20241022",
        ],
    }
    selected_model = st.selectbox("Choose Model", model_options[provider])
    
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# 2. System Prompt Definition
SYSTEM_PROMPT = (
    "You are an expert AI tutor. Explain complex artificial intelligence, "
    "machine learning, and deep learning concepts clearly, using intuitive analogies, "
    "mathematical foundations when requested, and practical real-world examples. Buy keep response crisp, in bullet points."
)

# 3. Session State Initialization
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display conversation history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 4. Universal LLM Streaming Handler
def stream_response(provider_name, key, model_name, messages):
    if provider_name == "OpenAI":
        client = OpenAI(api_key=key)
        formatted_msgs = [{"role": "system", "content": SYSTEM_PROMPT}] + [
            {"role": m["role"], "content": m["content"]} for m in messages
        ]
        response = client.chat.completions.create(
            model=model_name,
            messages=formatted_msgs,
            stream=True
        )
        for chunk in response:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    elif provider_name == "Anthropic Claude":
        client = anthropic.Anthropic(api_key=key)
        formatted_msgs = [
            {"role": m["role"], "content": m["content"]} for m in messages
        ]
        with client.messages.stream(
            model=model_name,
            max_tokens=2048,
            system=SYSTEM_PROMPT,
            messages=formatted_msgs,
        ) as stream:
            for text in stream.text_stream:
                yield text

    elif provider_name == "Google Gemini":
        client = genai.Client(api_key=key)
        # Format history as single prompt context for Gemini
        history_text = "\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in messages])
        response = client.models.generate_content_stream(
            model=model_name,
            contents=f"{history_text}\nAssistant:",
            config={"system_instruction": SYSTEM_PROMPT}
        )
        for chunk in response:
            yield chunk.text

# 5. User Input Handling
if prompt := st.chat_input("Ask a question about AI/ML (e.g., How does multi-head self-attention work?)..."):
    if not api_key:
        st.warning(f"Please enter your {provider} API key in the sidebar to proceed.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            response_generator = stream_response(
                provider, api_key, selected_model, st.session_state.messages
            )
            full_response = st.write_stream(response_generator)
            st.session_state.messages.append({"role": "assistant", "content": full_response})
        except Exception as e:
            st.error(f"Error calling {provider}: {str(e)}")