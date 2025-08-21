import streamlit as st
from rag_agent import build_vectorstore, create_agent
from helper import load_documents
import os
from pathlib import Path
import yaml
from design import load_frontend
from dotenv import load_dotenv
load_dotenv('.env')
st.set_page_config(page_title="Grant's Codex Agent", page_icon="📖")

# --------- Load config ---------
cfg = yaml.safe_load(open("config.yaml", "r"))
def mode_instructions(mode: str) -> str:
    return {
        "Interview":  "Answer concisely and professionally. Prioritize skills, impact, clarity.",
        "Story":      "Answer with a short narrative and an example grounded in the docs.",
        "FastFacts":  "Answer in tight bullet points (max 5).",
        "HumbleBrag": "Be confident and specific about achievements; avoid exaggeration.",
        "Reflect":    "Self-assess energy, collaboration style, and growth areas with candor.",
        "Projects":    "Answer exclusively about the projects I have done with excitement.",
    }.get(mode, "")


@st.cache_resource
def init_agent(mode):
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
        agent = create_agent(vectorstore_data, mode)
        return agent
        
    except Exception as e:
        st.error(f"Error initializing agent: {str(e)}")
        return None


st.title("📖 Grant's Codex Agent")
st.write("Ask me anything about my experience, skills, and values!")
mode = load_frontend()

agent = init_agent(mode=mode)

if agent is None:
    st.error("Failed to initialize the agent. Please check your setup:")
    st.write("1. Ensure the 'data' directory exists and contains your documents")
    st.write("2. Make sure Ollama is running locally")
    st.write("3. Check that all dependencies are installed")
else:

    st.write("Try questions like:")
    st.code(
        "What kind of data scientist are you?\n"
        "What are your strongest technical skills?\n"
        "What projects are you most proud of?\n"
        "What do you value in a team culture?\n"
        "How do you learn or debug something new?\n"
        "Please tell me you have a killer worship playlist? You have to share it with me. Do you have a link?\n"
        "Tell about your karate achievements. How did it shape your character? Are these achievements reported in a newspaper?\n"
        "I love the game of chess! Do you play? What is your username? Give me a friend request link!\n",
        language="text"
    )

    q = st.text_input("Your question")
    go = st.button("Ask")

    if q:
        with st.spinner("Thinking..."):
            answer = agent(q)
        st.markdown("### Answer")
        st.write(answer)
        st.caption(f"Mode: {mode} - {mode_instructions(mode)}")


                