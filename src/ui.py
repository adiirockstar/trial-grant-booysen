import streamlit as st
from rag_agent import build_vectorstore, create_agent
from helper import load_documents
import os
from pathlib import Path

from dotenv import load_dotenv
load_dotenv('.env')
st.set_page_config(page_title="Grant's Codex Agent", page_icon="📖")

@st.cache_resource
def init_agent():
    """Initialize the RAG agent with error handling"""
    try:
        # Check if data directory exists
        data_dir = "data"
        if not os.path.exists(data_dir):
            st.error(f"Data directory '{data_dir}' not found. Please create it and add your documents.")
            return None
        
        # Load documents
        docs = load_documents(data_dir)
        if not docs:
            st.warning(f"No documents found in '{data_dir}' directory.")
            return None
        
        st.success(f"Loaded {len(docs)} documents")
        
        # Build vector store
        vectorstore_data = build_vectorstore(docs)
        if vectorstore_data[0] is None:
            st.error("Failed to build vector store")
            return None
        
        # Create agent
        agent = create_agent(vectorstore_data)
        return agent
        
    except Exception as e:
        st.error(f"Error initializing agent: {str(e)}")
        return None

st.title("📖 Grant's Codex Agent")
st.write("Ask me anything about my experience, skills, and values!")

agent = init_agent()

if agent is None:
    st.error("Failed to initialize the agent. Please check your setup:")
    st.write("1. Ensure the 'data' directory exists and contains your documents")
    st.write("2. Make sure Ollama is running locally")
    st.write("3. Check that all dependencies are installed")
else:
    # Query interface
    query = st.text_input("Your question:")
    if query:
        with st.spinner("Thinking..."):
            answer = agent(query)
        st.markdown("### Answer")
        st.write(answer)