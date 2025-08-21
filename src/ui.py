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

def combine_documents(original_docs, uploaded_docs):
    """Combine original documents from data directory with uploaded documents"""
    all_docs = []
    
    # Add original documents
    if original_docs:
        all_docs.extend(original_docs)
    
    # Add uploaded documents
    if uploaded_docs:
        all_docs.extend(uploaded_docs)
    
    return all_docs

@st.cache_resource
def init_agent_with_uploads(mode):
    """Initialize the RAG agent including uploaded documents"""
    try:
        # Load original documents from data directory (if exists)
        original_docs = []
        data_dir = "data"
        if os.path.exists(data_dir):
            original_docs = load_documents(data_dir)
        
        # Get uploaded documents from session state
        uploaded_docs = st.session_state.get('uploaded_docs', [])
        
        # Combine all documents
        all_docs = combine_documents(original_docs, uploaded_docs)
        
        if not all_docs:
            return None, "No documents available. Upload some documents to get started!"
        
        st.success(f"📚 Loaded {len(original_docs)} original + {len(uploaded_docs)} uploaded = {len(all_docs)} total documents")
        
        # Build vector store
        vectorstore_data = build_vectorstore(all_docs)
        if vectorstore_data[0] is None:
            return None, "Failed to build vector store"
        
        # Create agent
        agent = create_agent(vectorstore_data, mode)
        return agent, f"Agent ready with {len(all_docs)} documents"
        
    except Exception as e:
        return None, f"Error initializing agent: {str(e)}"
    
def main():
    st.title("📖 Grant's Codex Agent")
    st.write("Ask me anything about my experience, skills, and values!")
    
    # Get the selected mode and uploaded documents from frontend
    mode, uploaded_docs = load_frontend()
    
    # Initialize cache key for agent reloading based on document changes
    docs_changed = st.session_state.get('docs_changed', 0)
    
    # Initialize agent with current document state
    agent, status_message = init_agent_with_uploads(docs_changed)
    
    if agent is None:
        st.error("**Agent not ready**")
        st.error(status_message)
        
        # Show helpful instructions
        st.info("🚀 **Getting Started:**")
        st.write("1. Upload documents using the sidebar")
        st.write("2. Supported formats: TXT, PDF, DOCX, Markdown") 
        st.write("3. Files are processed in-memory for this session")
        st.write("4. Make sure your Groq API key is configured")
        
        # Show current status
        original_count = len([f for f in os.listdir("data") if f.lower().endswith(('.txt', '.pdf', '.docx', '.md'))]) if os.path.exists("data") else 0
        uploaded_count = len(st.session_state.get('uploaded_docs', []))
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📁 Original Docs", original_count)
        with col2:
            st.metric("📤 Uploaded Docs", uploaded_count)
        with col3:
            st.metric("📊 Total", original_count + uploaded_count)
            
    else:
        # Agent is ready
        st.success(status_message)
        
        # Show current mode
        st.info(f"**Current Mode:** {mode} - {mode_instructions(mode)}")
        
        # Show example questions
        with st.expander("💡 Example Questions", expanded=False):
            st.code(
                "What kind of data scientist are you?\n"
                "What are your strongest technical skills?\n"
                "What projects are you most proud of?\n"
                "What do you value in a team culture?\n"
                "How do you learn or debug something new?\n"
                "Tell about your karate achievements. How did it shape your character?\n"
                "I love the game of chess! Do you play? What is your username?\n",
                language="text"
            )
        
        # Quick action buttons
        st.write("**Quick Questions:**")
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("🎯 Skills Overview"):
                st.session_state.quick_question = "What are your core technical skills and strengths?"
        with col2:
            if st.button("💼 Experience"):
                st.session_state.quick_question = "Tell me about your most impactful projects"
        with col3:
            if st.button("🌟 Values"):
                st.session_state.quick_question = "What do you value in work and team culture?"
        
        # Question input
        default_question = st.session_state.get('quick_question', '')
        q = st.text_input("Your question", value=default_question, placeholder="Ask me anything...")
        
        # Clear the quick question after it's been set
        if 'quick_question' in st.session_state and default_question:
            del st.session_state.quick_question
        
        if q:
            with st.spinner("🤔 Thinking..."):
                try:
                    # Pass both question and mode to the agent
                    answer = agent(q, mode)
                    
                    st.markdown("### 💬 Answer")
                    st.write(answer)
                    
                    # Show current mode and source info
                    original_count = len([f for f in os.listdir("data") if f.lower().endswith(('.txt', '.pdf', '.docx', '.md'))]) if os.path.exists("data") else 0
                    uploaded_count = len(st.session_state.get('uploaded_docs', []))
                    
                    st.caption(f"**Mode:** {mode} | **Sources:** {original_count} original + {uploaded_count} uploaded docs")
                    
                    # Feedback section
                    st.markdown("---")
                    col1, col2, col3 = st.columns([1, 1, 3])
                    with col1:
                        if st.button("👍 Helpful"):
                            st.success("Thanks for the feedback!")
                    with col2:
                        if st.button("👎 Not helpful"):
                            st.info("Try switching modes or adding more specific documents!")
                    
                except Exception as e:
                    st.error(f"Error getting response: {str(e)}")

    # Footer with session info and tips
    st.markdown("---")
    
    # Session persistence warning
    st.warning("⚠️ **Session Persistence:** Uploaded documents are stored in memory and will be lost when the session ends. Use the download feature in the sidebar to backup your content.")
    
    # Tips
    with st.expander("💡 Tips for Best Results"):
        st.markdown("""
        **📤 Uploading Documents:**
        - Upload relevant documents about Grant's experience, projects, skills
        - Supported formats: TXT, PDF, DOCX, Markdown
        - Files are processed in-memory (no permanent storage)
        
        **🎯 Getting Better Answers:**
        - Try different modes for varied response styles
        - Be specific in your questions
        - Upload more relevant documents for better context
        
        **🔧 Troubleshooting:**
        - If responses seem off, try refreshing the knowledge base in the side bar on the left
        - Clear and re-upload documents if needed
        - Make sure your Groq API key is properly configured
        """)

if __name__ == "__main__":
    main()
                