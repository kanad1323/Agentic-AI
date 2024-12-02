# streamlit_chatbot.py

import streamlit as st
from dotenv import load_dotenv
import os
import uuid
import bs4
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.tools.retriever import create_retriever_tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain.schema import HumanMessage

# Load environment variables
load_dotenv()

# Set User Agent
os.environ['USER_AGENT'] = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'

# Additional configurations
st.set_page_config(page_title="ChatBot", page_icon=":chat-plus-outline:", layout="wide", initial_sidebar_state="expanded", menu_items=None)

# Environment Variables
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
TAVILY_API_KEY = os.getenv('TAVILY_API_KEY')
LANGCHAIN_API_KEY = os.getenv('LANGCHAIN_API_KEY')
LANGCHAIN_TRACING_V2 = os.getenv('LANGCHAIN_TRACING_V2')

# Cache the model
@st.cache_resource
def load_model():
    return ChatOpenAI(model="gpt-4o-mini")

model = load_model()

# Cache the vector store setup
@st.cache_resource
def setup_vector_store():
    loader = WebBaseLoader(
        web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
        bs_kwargs={"parse_only": bs4.SoupStrainer(class_=("post-title", "post-header", "post-content"))},
    )
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, add_start_index=True
    )
    all_splits = text_splitter.split_documents(docs)

    embedding_wrapper = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        encode_kwargs={'batch_size': 256}  # Reduced batch size
    )
    vectorstore = FAISS.from_documents(documents=all_splits, embedding=embedding_wrapper)
    return vectorstore

vectorstore = setup_vector_store()

# Cache the retriever setup
@st.cache_resource
def setup_retriever(vectorstore):
    return vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})

retriever = setup_retriever(vectorstore)

# Cache the tools setup
@st.cache_resource
def setup_tools(retriever):
    retriever_tool = create_retriever_tool(
        retriever,
        "blog_post_retriever",
        "Searches and returns excerpts from the Autonomous Agents blog post.",
    )

    search = TavilySearchResults(max_results=2)
    wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())

    return [search, wikipedia, retriever_tool]

tools = setup_tools(retriever)

# Cache the agent executor setup
@st.cache_resource
def setup_agent(model, tools):
    memory = MemorySaver()
    unique_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": unique_id}}
    return create_react_agent(model, tools, checkpointer=memory), config

agent_executor_with_memory, agent_config = setup_agent(model, tools)

# Custom prompts
custom_prompt_template = "Keep your answers limited to one word while answering this question: {query}"

# Function to stream responses
def stream_query_response(query):
    custom_prompt = custom_prompt_template.format(query=query)

    # Stream the response from the agent
    for event in agent_executor_with_memory.stream(
        {"messages": [HumanMessage(content=custom_prompt)]},
        config=agent_config,
        stream_mode="values",
    ):
        yield event

# Initialize session state for chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Limit chat history to prevent excessive memory usage
MAX_HISTORY_LENGTH = 50  # Adjust as needed
if len(st.session_state.chat_history) > MAX_HISTORY_LENGTH:
    st.session_state.chat_history = st.session_state.chat_history[-MAX_HISTORY_LENGTH:]

# Streamlit UI
st.title("LangChain Chatbot with Streamlit Frontend")

# Display chat history
for chat in st.session_state.chat_history:
    with st.container():
        st.markdown(f"**User:** {chat['user']}")
        st.markdown(f"**Bot:** {chat['bot']}")

# User input
user_input = st.text_input("You:", key="input")

if st.button("Send") and user_input:
    # Add user message to chat history
    st.session_state.chat_history.append({"user": user_input, "bot": ""})
    latest_index = len(st.session_state.chat_history) - 1

    # Placeholder for bot response
    response_placeholder = st.empty()

    # Stream the response
    for response in stream_query_response(user_input):
        st.session_state.chat_history[latest_index]['bot'] = response
        # Update the placeholder with the latest response
        response_placeholder.markdown(f"**Bot:** {response}")
