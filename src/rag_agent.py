from sentence_transformers import SentenceTransformer
import faiss
from ollama import Client
import os

# Embeddings
embedder = SentenceTransformer("all-MiniLM-L6-v2")

def build_vectorstore(docs):
    texts = [d.page_content for d in docs]
    embeddings = embedder.encode(texts)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index, texts

# Local Ollama client
ollama_client = Client()

def create_agent(index, texts):
    def agent(question: str) -> str:
        q_emb = embedder.encode([question])
        D, I = index.search(q_emb, k=3)
        context = "\n\n".join([texts[i] for i in I[0]])
        prompt = f"""You are Grant's personal Codex Agent. 
Answer based on the context below, in his own voice:

{context}

Question: {question}
Answer:"""
        response = ollama_client.chat(model="llama3", messages=[{"role": "user", "content": prompt}])
        return response['message']['content']
    return agent
