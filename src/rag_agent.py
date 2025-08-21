
from groq import Groq
import os
from dotenv import load_dotenv
from pathlib import Path
from sentence_transformers import SentenceTransformer
import faiss
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



def create_agent(vectorstore_data, mode):
    """Create agent using Groq's Llama"""
    if vectorstore_data is None or vectorstore_data[0] is None:
        return lambda question: "Sorry, the knowledge base is not available."
    
    index, texts = vectorstore_data
    
    def agent(question: str, mode = mode) -> str:
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
            Answer as Grant in first person, based on the context provided.
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
    """Build FAISS vector store from documents (works with in-memory docs)"""
    if not docs:
        print("No documents provided to build_vectorstore")
        return None, []
    
    if embedder is None:
        print("Embedder not available for vectorstore")
        return None, []
    
    try:
        # Extract text content from documents
        texts = []
        for doc in docs:
            if hasattr(doc, 'page_content'):
                # Split long documents into chunks
                content = doc.page_content
                chunk_size = 1000
                overlap = 200
                
                if len(content) <= chunk_size:
                    texts.append(content)
                else:
                    # Split into overlapping chunks
                    for i in range(0, len(content), chunk_size - overlap):
                        chunk = content[i:i + chunk_size]
                        if len(chunk.strip()) > 50:  # Minimum chunk size
                            texts.append(chunk)
            else:
                print(f"Document missing page_content attribute: {doc}")
        
        if not texts:
            print("No valid text content extracted from documents")
            return None, []
        
        print(f"Processing {len(texts)} text chunks for vector store")
        
        # Create embeddings
        embeddings = embedder.encode(texts, show_progress_bar=True)
        
        # Build FAISS index
        dim = embeddings.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(embeddings.astype('float32'))
        
        print(f"Successfully built vector store with {len(texts)} chunks")
        return index, texts
        
    except Exception as e:
        print(f"Error building vector store: {e}")
        return None, []

def build_system_prompt(mode: str) -> str:
    base = (
        "You are my spokesperson and need to reflect how I think and speak.\n"
        "- Voice: tone that reflects how I speak in the documents.\n"
        "- Speak in first person ('I', 'my') as if you are Grant.\n"
        "- Ground answers in the provided context from my documents.\n"
        "- If context is insufficient, say what you need.\n"
        "- Stay true to Grant's voice and experiences\n"
    )
    return base + f"\nCurrent Mode: {mode}\n"

def get_mode_instruction(mode: str) -> str:
    """Get specific instructions for each mode"""
    return {
        "Interview":  "Answer concisely and professionally. Prioritize skills, impact, clarity using my mannerisms.",
        "Story":      "Answer with a short narrative and an example grounded in the docs using my mannerisms.",
        "FastFacts":  "Answer in tight bullet points (max 5) using my mannerisms.",
        "HumbleBrag": "Be confident and specific about achievements; avoid exaggeration using my mannerisms.",
        "Reflect":    "Self-assess energy, collaboration style, and growth areas with candor using my mannerisms.",
        "Projects":    "Answer exclusively about the projects I have done with excitement using my mannerisms.",
    }.get(mode, "Answer concisely and professionally.")