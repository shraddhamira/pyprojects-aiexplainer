import streamlit as st
from openai import OpenAI
import anthropic
from google import genai

# App configuration
st.set_page_config(page_title="AI Concept Tutor", page_icon="🤖", layout="wide")
st.title("🤖 AI Concept Tutor")

# 1. Dynamic Model Fetching & Key Validation Helper
@st.cache_data(show_spinner=False, ttl=300)
def fetch_available_models(provider_name: str, key: str):
    """
    Validates the API key by querying the provider's models endpoint
    and returns a filtered list of chat-compatible models.
    """
    if not key.strip():
        return []

    try:
        if provider_name == "Google Gemini":
            client = genai.Client(api_key=key)
            models = client.models.list()
            # Filter for generative/chat-capable Gemini models
            valid_models = [
                m.name.replace("models/", "")
                for m in models
                if "gemini" in m.name.lower() and "embed" not in m.name.lower()
            ]
            return sorted(valid_models, reverse=True)

        elif provider_name == "OpenAI":
            client = OpenAI(api_key=key)
            models = client.models.list()
            # Filter for standard GPT/Chat models and exclude embeddings/whisper/tts
            chat_keywords = ["gpt-4", "gpt-3.5", "o1", "o3", "chatgpt"]
            valid_models = [
                m.id for m in models.data
                if any(kw in m.id.lower() for kw in chat_keywords)
                and "audio" not in m.id and "realtime" not in m.id
            ]
            return sorted(valid_models, reverse=True)

        elif provider_name == "Anthropic Claude":
            client = anthropic.Anthropic(api_key=key)
            # Query the models list API to validate key and get active versions
            models = client.models.list()
            valid_models = [m.id for m in models.data if "claude" in m.id.lower()]
            return sorted(valid_models, reverse=True)

    except Exception as e:
        raise ValueError(f"Invalid API key or authentication failed: {str(e)}")

# 2. Sidebar Setup
with st.sidebar:
    st.header("Configuration")
    provider = st.selectbox(
        "Choose Provider",
        ["Google Gemini", "OpenAI", "Anthropic Claude"]
    )
    
    api_key = st.text_input(
        f"Enter your {provider} API Key", 
        type="password",
        placeholder="Paste your key here..."
    )

    available_models = []
    selected_model = None

    if api_key:
        with st.spinner("Validating key and fetching models..."):
            try:
                available_models = fetch_available_models(provider, api_key)
                if available_models:
                    st.success("API Key validated successfully!")
                    selected_model = st.selectbox("Choose Available Model", available_models)
                else:
                    st.warning("No chat-compatible models found for this key.")
            except ValueError as err:
                st.error(str(err))
    else:
        st.info("Enter your API key above to load available models.")

    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# 3. System Prompt Definition
SYSTEM_PROMPT = (
    "You are an expert AI tutor. Explain complex artificial intelligence, "
    "machine learning, and deep learning concepts clearly, using intuitive analogies, "
    "mathematical foundations when requested, and practical real-world examples. Please keep response crisp and short."
)

# 4. Session State Initialization
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display conversation history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 5. Universal LLM Streaming Handler
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
        history_text = "\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in messages])
        response = client.models.generate_content_stream(
            model=model_name,
            contents=f"{history_text}\nAssistant:",
            config={"system_instruction": SYSTEM_PROMPT}
        )
        for chunk in response:
            yield chunk.text

# 6. User Input Handling
if prompt := st.chat_input("Ask a question about AI/ML (e.g., Explain backpropagation intuitively)..."):
    if not api_key:
        st.warning(f"Please enter your {provider} API key in the sidebar.")
        st.stop()
        
    if not selected_model:
        st.warning("Please select a valid model from the dropdown to continue.")
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