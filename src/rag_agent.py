
from groq import Groq
import os
from dotenv import load_dotenv
from pathlib import Path
from sentence_transformers import SentenceTransformer
import faiss

# At the top of rag_agent.py
import os
from dotenv import load_dotenv
from pathlib import Path

import streamlit as st
loaded_env = False
try:
    api_key = st.secrets["GROQ_API_KEY"]
    loaded_env = True
except:
    print("Could not load env variable using streamlit")

if loaded_env == False:
    env_path = Path(__file__).parent.parent / '.env'
    print(f"Looking for .env at: {env_path}")
    print(f"File exists: {env_path.exists()}")

    load_dotenv(env_path)

    api_key = os.getenv('GROQ_API_KEY')
    print(f"Loaded API key: {api_key[:10]}..." if api_key else "No API key found")

groq_client = None
try:
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
except Exception as e:
    print(f"Failed to load embedder: {e}")
    embedder = None

def create_agent(vectorstore_data):
    """Create agent using Groq's Llama"""
    if vectorstore_data is None:
        return lambda question: "Sorry, the knowledge base is not available."
    
    index, texts = vectorstore_data
    
    def agent(question: str) -> str:
        if not os.getenv("GROQ_API_KEY"):
            return "Please set GROQ_API_KEY environment variable"
        
        # Initialize Groq client
        groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        print("Successfully loaded Groq!")
        try:
            # Search for relevant documents
            q_emb = embedder.encode([question])
            D, I = index.search(q_emb.astype('float32'), k=3)
            context = "\n\n".join([texts[i] for i in I[0] if i < len(texts)])
            system_prompt = build_system_prompt(mode)
            mode_instruction = get_mode_instruction(mode)
            prompt = f"""{system_prompt}:

            Mode Instructions: {mode_instruction}
            {context}

            Question: {question}
            Answer:"""
            
            # Use Groq's Llama model
            response = groq_client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama3-8b-8192",  # or "llama3-70b-8192" for larger model
                max_tokens=500,
                temperature=0.7
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"Sorry, I encountered an error: {str(e)}"
    
    return agent

def build_vectorstore(docs):
    """Build FAISS vector store from documents"""
    if not docs:
        print("No documents provided")
        return None, []
    
    if embedder is None:
        print("Embedder not available")
        return None, []
        
    texts = [d.page_content for d in docs]
    
    try:
        embeddings = embedder.encode(texts)
        dim = embeddings.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(embeddings.astype('float32'))  # Ensure float32 for FAISS
        return index, texts
    except Exception as e:
        print(f"Error building vector store: {e}")
        return None, []

def build_system_prompt(mode: str) -> str:
    base = (
        "You are Grant Booysen's Personal Codex Agent.\n"
        "- Voice: concise, pragmatic\n"
        "- Speak in first person ('I', 'my').\n"
        "- Ground answers in the provided context from my documents.\n"
        "- If context is insufficient, say what you need.\n"
    )
    return base + f"\nCurrent Mode: {mode}\n"

def get_mode_instruction(mode: str) -> str:
    """Get specific instructions for each mode"""
    return {
        "Interview":  "Answer concisely and professionally. Prioritize skills, impact, clarity.",
        "Story":      "Answer with a short narrative and an example grounded in the docs.",
        "FastFacts":  "Answer in tight bullet points (max 5).",
        "HumbleBrag": "Be confident and specific about achievements; avoid exaggeration.",
        "Reflect":    "Self-assess energy, collaboration style, and growth areas with candor.",
    }.get(mode, "Answer concisely and professionally.")