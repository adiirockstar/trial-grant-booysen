
from groq import Groq
import os

# Initialize Groq client
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def create_agent(vectorstore_data):
    """Create agent using Groq's Llama"""
    if vectorstore_data is None:
        return lambda question: "Sorry, the knowledge base is not available."
    
    index, texts = vectorstore_data
    
    def agent(question: str) -> str:
        if not os.getenv("GROQ_API_KEY"):
            return "Please set GROQ_API_KEY environment variable"
        
        try:
            # Search for relevant documents
            q_emb = embedder.encode([question])
            D, I = index.search(q_emb.astype('float32'), k=3)
            context = "\n\n".join([texts[i] for i in I[0] if i < len(texts)])
            
            prompt = f"""You are Grant's personal Codex Agent. Answer based on the context below, in his own voice:

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