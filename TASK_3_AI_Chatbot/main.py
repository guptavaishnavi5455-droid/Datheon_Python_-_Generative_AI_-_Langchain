import os
import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# Load API key
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

# Page configuration
st.set_page_config(
    page_title="AI Chatbot",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 AI Chatbot")
st.caption("Powered by Google Gemini")

if not api_key:
    st.error("Google API key not found. Check your .env file.")
    st.stop()

# Initialize Gemini
llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    google_api_key=api_key,
    temperature=0.3,
    max_output_tokens=256
)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Clear chat button
if st.sidebar.button("Clear Chat"):
    st.session_state.messages = []
    st.rerun()

st.sidebar.title("Chat Settings")
st.sidebar.write("Model: Gemini 2.5 Flash")
st.sidebar.write("Chat history is stored during this session.")

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
prompt = st.chat_input("Ask me anything...")

if prompt:
    # Display user message
    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })

    with st.chat_message("user"):
        st.markdown(prompt)

    # Prepare conversation for Gemini
    history = [
        SystemMessage(content="You are a helpful AI assistant.")
    ]

    for message in st.session_state.messages[:-1]:
        if message["role"] == "user":
            history.append(HumanMessage(content=message["content"]))
        else:
            history.append(AIMessage(content=message["content"]))

    history.append(HumanMessage(content=prompt))

    # Generate streaming response
    with st.chat_message("assistant"):
        try:
            def generate_response():
                for chunk in llm.stream(history):
                    if isinstance(chunk.content, str):
                        yield chunk.content
                    elif isinstance(chunk.content, list):
                        for item in chunk.content:
                            if isinstance(item, dict) and item.get("type") == "text":
                                yield item.get("text", "")

            response = st.write_stream(generate_response())

            st.session_state.messages.append({
                "role": "assistant",
                "content": response
            })

        except Exception as e:
            st.error(f"Error: {e}")