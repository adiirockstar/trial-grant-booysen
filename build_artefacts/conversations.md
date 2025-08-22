---

# Codex Agent Project Development Log

This file documents the workflow in the format:

**Prompt → AI Response → Final Code Snippet**

---

## 1. How do I build this project?

**Prompt:**
How do I build this:

> *Ubundi Trial Project: Build a Personal Codex Agent*
> Objective: Build a context-aware agent that can answer questions about you as a candidate—based on documents and data you’ve created...
> (Full brief provided above.)

---

**AI Response:**

Got it — you want a battle plan for **actually building this Codex Agent project**. Here’s how I’d structure it so you hit the 6–8h budget, tick all the evaluation boxes, and still have something polished to show:

---

### 🛠 Step 1: Gather Your Training Data

* Put your **CV** in `data/cv.md` (or PDF).
* Add **2–3 extras** (blogs, README, code comments, personal notes).
* Keep them authentic: they don’t have to be fancy, but they should represent your voice.

👉 Example:

* `cv.md` — professional highlights.
* `blog_debugging.md` — short post on how you learn/debug.
* `readme_feature_store.md` — a technical project doc.

---

### 🛠 Step 2: Decide on Architecture

Two lightweight paths:

1. **With RAG (recommended)**

   * Embed docs (SentenceTransformers / FAISS).
   * Retrieve top chunks as context.
   * Pass them into an LLM with your **system persona prompt**.

2. **Without RAG (fallback)**

   * Store curated snippets in JSON.
   * On each question, pull from rules (skills → CV, values → notes).
   * In README explain how you’d add RAG later.

---

### 🛠 Step 3: Persona + Modes

* **System Prompt**: “You are Grant’s Codex Agent. Speak in first person, concise, pragmatic, with dry humor.”
* **Modes** (radio button in UI):

  * Interview → concise, professional.
  * Story → longer, reflective.
  * Fast Facts → bullet points.
  * Humble Brag → confidence boosted.
  * Reflect → self-assessment.

---

### 🛠 Step 4: UI

* Use **Streamlit** (quickest to deploy).
* One-page app:

  * Sidebar → choose mode, upload docs, rebuild index.
  * Main → ask a question, see answer + citations.

---

### 🛠 Step 5: Logging & “Show Your Thinking”

* Keep `artifacts/` folder:

  * `design_notes.md` → explain choices.
  * `prompts_log.jsonl` → log prompt → AI → code.
  * `sub_agents.md` → e.g. “Helper agent for chunking text”.
* The app can also auto-log interactions to `/logs`.

---

### 🛠 Step 6: README

Include:

* Setup (`pip install -r requirements.txt`, `streamlit run app.py`).
* Sample questions & answers.
* Design choices (RAG, persona, fallback mode).
* Improvements with more time.

---

### 🛠 Step 7: Deploy

* **Easiest**: Streamlit Cloud (free, just push repo).
* Alternatives: Replit, Vercel.
* Add link to deployed app in README.

---

### 🛠 Step 8: Submission

* GitHub repo with:

  * `/data` (your CV + supporting docs).
  * `/artifacts` (AI collaboration history).
  * `/rag` (RAG code).
  * `app.py` (Streamlit).
  * `README.md` (full guide).
* Short **video walkthrough** (<5min).

---

✅ That’s it — you’ll have a functioning Codex Agent that:

* Speaks in your voice.
* Uses your real docs.
* Has modes/personality.
* Shows AI-collaboration artifacts.
* And is simple to run.

---

**AI Follow-up:**
If you want, I can hand you a **ready-to-run starter project** (already zipped with Streamlit app + RAG setup + sample docs) so you just drop in your own CV/blogs and tweak persona.

👉 Do you want me to share that starter so you can just plug your data in and go?

---

**Final Code Snippet:**

```python
import os
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, TextLoader

def load_documents(data_path="data"):
    docs = []
    for f in os.listdir(data_path):
        path = os.path.join(data_path, f)
        if f.endswith(".pdf"):
            docs.extend(PyPDFLoader(path).load())
        elif f.endswith(".txt") or f.endswith(".md"):
            docs.extend(TextLoader(path).load())
    return docs

def build_vectorstore(docs):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore

def create_agent(vectorstore):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.5)

    def agent(question: str) -> str:
        context_docs = retriever.get_relevant_documents(question)
        context = "\n\n".join([d.page_content for d in context_docs])
        prompt = f"""You are Grant's personal Codex Agent. 
Answer questions in his voice, grounded in the following context:

{context}

Question: {question}
Answer:"""
        return llm.predict(prompt)

    return agent
```

```python
import streamlit as st
from rag_agent import load_documents, build_vectorstore, create_agent

st.set_page_config(page_title="Grant's Codex Agent", page_icon="📖")

@st.cache_resource
def init_agent():
    docs = load_documents("data")
    vs = build_vectorstore(docs)
    return create_agent(vs)

st.title("📖 Grant's Codex Agent")
st.write("Ask me anything about my experience, skills, and values!")

agent = init_agent()

query = st.text_input("Your question:")
if query:
    answer = agent(query)
    st.markdown("### Answer")
    st.write(answer)
```





## 2. How to add resources
**Prompt:**
I want to know how to do the following: Make radio buttons on the frontend to configure modes. Put in additional resources that can be fun facts about me like a link to my spotify playlist, chess.com username, and a newspaper article about me as a child. How do I integrate it into the app? Give me a clear step by step process on how to add features - explain it as simply as possible. Then provide me example code as to how to implement the features

---

**AI Response:**

Love this—let’s bolt on modes + “fun links” in a way that’s simple, free, and tidy. I’ll give you a clear checklist first, then drop copy-pasteable code.

---

# Step-by-step: add Modes (radio buttons) + Fun Resources

## A) Modes (Interview, Story, Fast Facts, Humble Brag, Reflect)

1. **Define your modes**
   Decide the set and what each should do (short + punchy vs narrative, etc.).

2. **Add radio buttons in the UI**
   In Streamlit, put a `st.radio` in the sidebar. The selected value becomes `mode`.

3. **Wire mode → prompt**
   Map each mode to a short instruction string and include it in your prompt to the LLM (or your heuristic fallback).

4. **(Optional) Mode-specific extras**
   For example, if `FastFacts`, also display bullet-point facts in the UI.

---

## B) Fun resources (Spotify, chess.com, childhood article)

You want them:

* **Visible in the UI** (clickable links)
* **Searchable by the agent** (so they can be referenced in answers)

We’ll do both:

1. **Add them to `config.yaml`**
   Put links + username in a structured `resources` block.

2. **Create a tiny `data/resources.md`**
   A simple markdown file listing those links and “fast facts”—so your RAG index ingests them like any other doc.

3. **Render them in the app**
   An expander called “Fun links about me”, with clickable links.

4. **(Optional) Use them in answers**
   If `FastFacts` mode is on, prepend some bullets from config to the answer.

---

# Example config + code

Below is *minimal* example code that you can drop into your existing Streamlit app (free stack; no paid APIs required). It assumes you already have a simple RAG setup and a `config.yaml`.

## 1) `config.yaml` — add resources + modes stay the same

```yaml
# ... your existing config ...
resources:
  spotify_playlist: "https://open.spotify.com/playlist/EXAMPLE"
  chess_username: "grant_booysen"
  chess_profile: "https://www.chess.com/member/grant_booysen"
  childhood_article: "https://news.example.com/article-about-grant"
  fast_facts:
    - "I’m happiest when shipping measurable things."
    - "Comfort stack: Python, SQL, PyTorch; sprinkle of GNNs."
    - "I debug with a thin end-to-end slice, then binary-search the stack."
    - "Former chess club troublemaker. Rapid ~1500."
```

## 2) `data/resources.md` — make it indexable by RAG

Create this file (ingestion will pick it up just like your CV/blogs):

```markdown
---
title: "Resources — Links & Fast Facts"
tags: ["links", "fun-facts"]
---

# Links
- Spotify playlist: https://open.spotify.com/playlist/EXAMPLE
- Chess.com profile: https://www.chess.com/member/grant_booysen
- Childhood newspaper article: https://news.example.com/article-about-grant

# Fast Facts
- I’m happiest when shipping measurable things.
- Comfort stack: Python, SQL, PyTorch; sprinkle of GNNs.
- I debug with a thin end-to-end slice, then binary-search the stack.
- Former chess club troublemaker. Rapid ~1500.
```

## 3) Streamlit UI — radio buttons for modes + resources expander

Add this to your `app.py` (or integrate with your existing app):

```python
import streamlit as st
import yaml
from pathlib import Path

# --------- Load config ---------
cfg = yaml.safe_load(open("config.yaml", "r"))

def mode_instructions(mode: str) -> str:
    return {
        "Interview":  "Answer concisely and professionally. Prioritize skills, impact, clarity.",
        "Story":      "Answer with a short narrative and an example grounded in the docs.",
        "FastFacts":  "Answer in tight bullet points (max 5).",
        "HumbleBrag": "Be confident and specific about achievements; avoid exaggeration.",
        "Reflect":    "Self-assess energy, collaboration style, and growth areas with candor.",
    }.get(mode, "")

st.set_page_config(page_title="Personal Codex Agent", page_icon="🗂️")

# --------- Sidebar: Modes ---------
with st.sidebar:
    st.header("Settings")
    mode = st.radio(
        "Mode",
        ["Interview", "Story", "FastFacts", "HumbleBrag", "Reflect"],
        index=0
    )
    st.caption("Mode tweaks tone & structure of answers.")

# --------- Main: Title + Resources ---------
st.title("🗂️ Personal Codex Agent")
st.caption("Ask about my work, skills, and values — grounded in my own docs.")

with st.expander("🔗 Fun links about me"):
    res = cfg.get("resources", {})
    if res:
        if res.get("spotify_playlist"):
            st.markdown(f"- **Spotify playlist:** [{res['spotify_playlist']}]({res['spotify_playlist']})")
        if res.get("chess_profile"):
            label = res.get("chess_username", res["chess_profile"])
            st.markdown(f"- **Chess.com:** [{label}]({res['chess_profile']})")
        if res.get("childhood_article"):
            st.markdown(f"- **Childhood article:** [{res['childhood_article']}]({res['childhood_article']})")
    else:
        st.info("Add links in `config.yaml → resources` to show them here.")

# --------- Query box ---------
st.write("Try questions like:")
st.code(
    "What kind of engineer are you?\n"
    "What are your strongest technical skills?\n"
    "What projects are you most proud of?\n"
    "What do you value in a team culture?\n"
    "How do you learn or debug something new?",
    language="text"
)

q = st.text_input("Your question")
go = st.button("Ask")

# --------- Replace these with your actual RAG calls ---------
def retrieve_context(question: str) -> str:
    # TODO: call your vector index and join top chunks here.
    # For now, return a placeholder.
    return "(context from your docs goes here)"

def answer_with_llm(system_prompt: str, user_prompt: str) -> str:
    # TODO: call your local LLM (e.g., Ollama) or your fallback.
    # For now, just echo how it would work.
    return f"🧠 (LLM would answer here)\n\n{user_prompt[:600]}..."

def build_system_prompt(mode: str) -> str:
    base = (
        "You are Grant Booysen's Personal Codex Agent.\n"
        "- Voice: concise, pragmatic, dry humor.\n"
        "- Speak in first person ('I', 'my').\n"
        "- Ground answers in the provided context from my documents.\n"
        "- If context is insufficient, say what you need.\n"
    )
    return base + f"\nCurrent Mode: {mode}\n"

# --------- Handle ask ---------
if go and q.strip():
    with st.spinner("Thinking..."):
        context = retrieve_context(q)
        instructions = mode_instructions(mode)
        user_prompt = (
            f"{instructions}\n\n"
            f"Question: {q}\n\n"
            f"Context:\n{context}\n\n"
            "If context is insufficient, say what you need."
        )
        system_prompt = build_system_prompt(mode)
        answer = answer_with_llm(system_prompt, user_prompt)

        # Mode-specific extras (e.g., FastFacts: prepend quick bullets)
        if mode == "FastFacts":
            facts = cfg.get("resources", {}).get("fast_facts", [])
            if facts:
                st.markdown("**Fast facts**")
                st.markdown("- " + "\n- ".join(facts))

        st.markdown(answer)
```

> Where to plug in RAG:
> Replace `retrieve_context()` with: search your FAISS/Chroma index for top-k chunks and join them.
> Replace `answer_with_llm()` with your **free** local model call (e.g., using **Ollama**).

---

## 4) (Optional) Free local LLM call with Ollama

If you’re on the free stack:

```python
from ollama import Client
ollama_client = Client()  # assumes `ollama` is running locally

def answer_with_llm(system_prompt: str, user_prompt: str) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    resp = ollama_client.chat(model="llama3", messages=messages)
    return resp["message"]["content"]
```

---

## 5) Make the links searchable by your agent (RAG)

You already created `data/resources.md`. Just **re-run your ingest step** so it’s in the index:

```bash
python scripts/ingest.py
```

Now if someone asks, “Do you have a playlist?” the retriever can surface those URLs.

---

# TL;DR (you asked for simple!)

1. Add a `resources` section into `config.yaml` with your links + facts.
2. Create `data/resources.md` (so the agent can “see” them).
3. Add a radio button for modes in Streamlit and pass the mode as instructions into the prompt.
4. Show the links in a Streamlit expander.
5. (Optional) In `FastFacts` mode, print bullet points from `config.yaml`.
6. Re-ingest docs so your resources are retrievable.

If you want the **ready-to-run starter zip** with these features already wired up (Streamlit app + RAG skeleton + resources block + FastFacts injection), say the word and I’ll hand you the download link.

---

**Prompt → AI Response → Final Code Snippet**

---










## 3. How do I deploy this project?

**Prompt:**
I am trying to deploy my app to streamlit but I am having import issues. from sentence_transformers import SentenceTransformer import faiss from ollama import Client import os  # Embeddings embedder = SentenceTransformer("all-MiniLM-L6-v2")  def build_vectorstore(docs):     texts = [d.page_content for d in docs]     embeddings = embedder.encode(texts)     dim = embeddings.shape[1]     index = faiss.IndexFlatL2(dim)     index.add(embeddings)     return index, texts  # Local Ollama client ollama_client = Client()  def create_agent(index, texts):     def agent(question: str) -> str:         q_emb = embedder.encode([question])         D, I = index.search(q_emb, k=3)         context = "\n\n".join([texts[i] for i in I[0]])         prompt = f"""You are Grant's personal Codex Agent.  Answer based on the context below, in his own voice:  {context}  Question: {question} Answer:"""         response = ollama_client.chat(model="llama3", messages=[{"role": "user", "content": prompt}])         return response['message']['content']     return agent and another file: import streamlit as st
from rag_agent import load_documents, build_vectorstore, create_agent

st.set_page_config(page_title="Grant's Codex Agent", page_icon="📖")

@st.cache_resource
def init_agent():
    docs = load_documents("data")
    vs = build_vectorstore(docs)
    return create_agent(vs)

st.title("📖 Grant's Codex Agent")
st.write("Ask me anything about my experience, skills, and values!")

agent = init_agent()

query = st.text_input("Your question:")
if query:
    answer = agent(query)
    st.markdown("### Answer")
    st.write(answer)

---

**AI Response:**
Looking at your error logs, the main issue is that your `ui.py` file is trying to import `load_documents` from `rag_agent.py`, but that function doesn't exist in your `rag_agent.py` file.

Here are the issues I can see:

## Missing Function
Your `rag_agent.py` file is missing the `load_documents` function that `ui.py` is trying to import.

## HuggingFace Rate Limiting
The 429 errors from HuggingFace suggest you're hitting rate limits when trying to download the sentence transformer model.

Here's how to fix these issues:## Additional Steps to Fix Deployment

1. **Create a `data` directory** in your project root and add your documents (`.txt` or `.md` files)

2. **Update your `requirements.txt`** to handle the dependencies better:3. **Consider adding a fallback** for when Ollama isn't available in the cloud environment. Streamlit Cloud won't have Ollama running, so you might want to:## Key Changes Made:

1. **Added `load_documents` function** that was missing
2. **Added retry logic** for HuggingFace downloads to handle 429 errors
3. **Better error handling** throughout the code
4. **Added fallback options** for when services aren't available
5. **Made the code more robust** for cloud deployment

## Deployment Recommendations:

1. **For local development**: Use the first version with Ollama
2. **For Streamlit Cloud**: Use the OpenAI version and set `OPENAI_API_KEY` in your Streamlit secrets
3. **Create sample documents**: Add some `.txt` or `.md` files to a `data/` directory
4. **Test locally first** before deploying to make sure everything works

The main issue was the missing `load_documents` function and the HuggingFace rate limiting. These fixes should resolve your import errors and make your app more resilient to deployment issues.
---

4. How do I integrate a free tool like Groq into my service?





## 4. How to switch to free tools
**Prompt:**
can I not use llama for cloud deployment? Integrate my current setup with groq instead of Llama. Only change the necessary lines of code. 

---

**AI Response:**

You're right to ask about this - using Llama (via Ollama) for cloud deployment on Streamlit Cloud has some significant challenges:

## Why Ollama/Llama is problematic for Streamlit Cloud:

1. **No persistent compute**: Streamlit Cloud runs in containers that don't have Ollama installed and don't allow you to install system-level services
2. **Resource limitations**: Running local LLMs requires significant CPU/RAM that exceeds Streamlit Cloud's free tier limits
3. **Cold starts**: Even if you could run it, model loading would make your app extremely slow to start

## Better alternatives for cloud deployment:

### Option 1: Use Ollama via API (if you have a server)
If you have your own server running Ollama, you can connect to it remotely:### Option 2: Use Hugging Face Inference API (Free tier available)
This gives you access to Llama models without running them locally:### Option 3: Use Groq (Very fast Llama inference, free tier)
Groq offers incredibly fast Llama inference with a generous free tier:## My Recommendations:

1. **For development**: Keep using Ollama locally
2. **For cloud deployment**: Use **Groq** - it's free, fast, and gives you access to Llama 3 models
3. **Fallback option**: Hugging Face Inference API
4. **If you have your own server**: Remote Ollama

## Updated requirements.txt for Groq:
```
streamlit
sentence-transformers>=2.2.0
faiss-cpu
groq
requests
urllib3
```

**Groq is probably your best bet** - they offer free access to Llama 3 models with very fast inference speeds, perfect for a Streamlit app. You just need to:

1. Sign up at [console.groq.com](https://console.groq.com)
2. Get a free API key
3. Add it to your Streamlit secrets as `GROQ_API_KEY`

Would you like me to help you set up the Groq version?
---

**Final code snippet:**

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







5. Load data from different file types:
**Prompt**
How do I parse data from the following document types into the embedder: docx, pdf, md, txt. Write me the code to load these resources from ./data. and collectively add it to the context of the prompt. Give me just the code to do so.

**final code snippet** 
from pathlib import Path
import PyPDF2
from docx import Document as DocxDocument

def load_documents(data_dir):
    """Load documents from a directory - supports txt, md, pdf, docx"""
    docs = []
    data_path = Path(data_dir)
    
    if not data_path.exists():
        print(f"Warning: Data directory {data_dir} does not exist")
        return []
    
    # Text files
    for file_path in data_path.glob("**/*.txt"):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                doc = type('Document', (), {'page_content': content, 'metadata': {'source': str(file_path)}})()
                docs.append(doc)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    # Markdown files
    for file_path in data_path.glob("**/*.md"):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                doc = type('Document', (), {'page_content': content, 'metadata': {'source': str(file_path)}})()
                docs.append(doc)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")

    # PDF files
    for file_path in data_path.glob("**/*.pdf"):
        try:
            with open(file_path, 'rb') as f:
                pdf_reader = PyPDF2.PdfReader(f)
                content = ""
                for page in pdf_reader.pages:
                    content += page.extract_text() + "\n"
                doc = type('Document', (), {'page_content': content, 'metadata': {'source': str(file_path)}})()
                docs.append(doc)
        except Exception as e:
            print(f"Error loading PDF {file_path}: {e}")

    # DOCX files
    for file_path in data_path.glob("**/*.docx"):
        try:
            docx_doc = DocxDocument(file_path)
            content = ""
            for paragraph in docx_doc.paragraphs:
                content += paragraph.text + "\n"
            doc = type('Document', (), {'page_content': content, 'metadata': {'source': str(file_path)}})()
            docs.append(doc)
        except Exception as e:
            print(f"Error loading DOCX {file_path}: {e}")
    
    return docs







6. Live session management
**Prompt**
I want to build a live session management for storing and retraining my chatbot. I want the user to be able to drop files into the 
session storage from the side bar, refresh the question and see the result. Explain to me how the session management would work in this context.
Provide source code as to how to build it.

**Response**
I'll create an upload feature in the sidebar that allows you to add more documents to your corpus. This will require updating both the frontend and the main app to handle file uploads and rebuild the vector store.I've created a comprehensive document upload feature for your RAG system. Here's what I've added:

## New Features:

### 1. **Upload Section in Sidebar**
- File uploader supporting TXT, PDF, DOCX, and Markdown files
- Multiple file upload capability
- Progress indicators and success/error messages
- File validation (size limits, supported formats)

### 2. **Data Management**
- Current document counter
- "Show Current Files" button to list all documents with sizes
- "Refresh Knowledge Base" button to reload the vector store
- File statistics display

### 3. **Smart Cache Management**
- Uses session state to track when new documents are uploaded
- Automatically refreshes the agent when new files are added
- Cache invalidation system to ensure new documents are processed

### 4. **Enhanced User Experience**
- Quick question buttons for common queries
- Current mode display
- Feedback buttons
- Tips section at the bottom
- Better error handling and user guidance

## Key Technical Changes:

1. **Frontend (`design.py`)**:
   - Added file uploader with validation
   - Document management interface
   - Returns both mode and upload status

2. **Main App**:
   - Cache key system for agent reloading
   - Better error messages and guidance
   - Enhanced UI with quick buttons and tips

3. **Helper Functions**:
   - Support for multiple file formats (TXT, PDF, DOCX, MD)
   - File validation and statistics
   - Better error handling for document loading

## Usage:

1. **Upload Documents**: Use the sidebar to select and upload files
2. **Automatic Processing**: The system automatically saves files and rebuilds the knowledge base
3. **Immediate Availability**: New documents are immediately available for queries
4. **Management**: View current files, refresh the system, or clear cache as needed

## Required Dependencies:
You may need to install additional packages:
```bash
pip install PyPDF2 python-docx
```

This creates a dynamic system where users can continuously expand Grant's knowledge base by uploading new documents through the web interface!

**final code snippet**
from groq import Groq
import os
from dotenv import load_dotenv
from pathlib import Path
from sentence_transformers import SentenceTransformer
import faiss
import streamlit as st

# Load environment variables
loaded_env = False
api_key = None

# Try Streamlit secrets first
try:
    api_key = st.secrets["GROQ_API_KEY"]
    loaded_env = True
    print("Loaded API key from Streamlit secrets")
except:
    print("Could not load API key from Streamlit secrets")

# Fallback to .env file
if not loaded_env:
    env_path = Path(__file__).parent.parent / '.env'
    if env_path.exists():
        load_dotenv(env_path)
        api_key = os.getenv('GROQ_API_KEY')
        print(f"Loaded API key from .env: {api_key[:10]}..." if api_key else "No API key found in .env")
    else:
        print("No .env file found")

# Initialize embedder
embedder = None
try:
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    print("Successfully loaded embedder")
except Exception as e:
    print(f"Failed to load embedder: {e}")

def create_agent(vectorstore_data):
    """Create agent using Groq's Llama with in-memory document support"""
    if vectorstore_data is None or vectorstore_data[0] is None:
        return lambda question, mode="Interview": "Sorry, the knowledge base is not available. Please upload some documents first."
    
    index, texts = vectorstore_data
    print(f"Agent created with {len(texts)} documents in knowledge base")
    
    def agent(question: str, mode: str = "Interview") -> str:
        # Check API key
        groq_api_key = api_key or os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            return "❌ Please configure GROQ_API_KEY in Streamlit secrets or environment variables"
        
        if embedder is None:
            return "❌ Text embedder not available. Please check your setup."
        
        try:
            # Initialize Groq client
            groq_client = Groq(api_key=groq_api_key)
            
            # Search for relevant documents
            q_emb = embedder.encode([question])
            D, I = index.search(q_emb.astype('float32'), k=5)  # Get top 5 most relevant chunks
            
            # Get context from relevant documents
            relevant_texts = []
            for i in I[0]:
                if i < len(texts) and D[0][len(relevant_texts)] < 1.5:  # Distance threshold
                    relevant_texts.append(texts[i])
                if len(relevant_texts) >= 3:  # Limit context size
                    break
            
            context = "\n\n".join(relevant_texts) if relevant_texts else "No relevant context found."
            
            # Build prompt with mode-specific instructions
            system_prompt = build_system_prompt(mode)
            mode_instruction = get_mode_instruction(mode)
            
            prompt = f"""{system_prompt}

Mode Instructions: {mode_instruction}

Context from Grant's documents:
{context}

Question: {question}

Answer as Grant in first person, based on the context provided:"""
            
            # Use Groq's Llama model
            response = groq_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are Grant Booysen's AI assistant. Respond in first person as Grant, based on the provided context."},
                    {"role": "user", "content": prompt}
                ],
                model="llama3-8b-8192",
                max_tokens=600,
                temperature=0.7
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"Agent error: {str(e)}")
            return f"❌ Sorry, I encountered an error: {str(e)}"
    
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
        "You are Grant Booysen's Personal AI Assistant.\n"
        "Key instructions:\n"
        "- Respond in first person as Grant ('I', 'my', 'me')\n"
        "- Be conversational and authentic\n"
        "- Base answers on the provided context from Grant's documents\n"
        "- If context is insufficient, acknowledge what you don't know\n"
        "- Stay true to Grant's voice and experiences\n"
    )
    return base + f"\nCurrent Response Mode: {mode}\n"

def get_mode_instruction(mode: str) -> str:
    """Get specific instructions for each response mode"""
    instructions = {
        "Interview": "Answer concisely and professionally. Focus on skills, achievements, and impact. Be direct and clear.",
        "Story": "Tell engaging stories with specific examples. Use narrative structure with context, action, and results.",
        "FastFacts": "Respond with concise bullet points (maximum 5). Each point should be specific and actionable.",
        "HumbleBrag": "Be confident about achievements while staying authentic. Highlight specific accomplishments and their impact.",
        "Reflect": "Be introspective and honest. Discuss growth areas, learning experiences, and self-awareness candidly.",
    }
    return instructions.get(mode, "Answer in a helpful and authentic manner.")

# Helper function for testing
def test_agent_setup():
    """Test if the agent can be properly initialized"""
    try:
        # Test API key
        if not (api_key or os.getenv("GROQ_API_KEY")):
            return False, "GROQ_API_KEY not configured"
        
        # Test embedder
        if embedder is None:
            return False, "Embedder not loaded"
        
        # Test Groq client
        groq_client = Groq(api_key=api_key or os.getenv("GROQ_API_KEY"))
        
        return True, "Agent setup successful"
        
    except Exception as e:
        return False, f"Setup error: {str(e)}"

    import streamlit as st
import os
import yaml
from pathlib import Path

def load_config():
    config_path = os.path.join(os.path.dirname(__file__), "../config.yaml")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def process_uploaded_file(uploaded_file):
    """Process uploaded file and extract text content"""
    try:
        file_extension = uploaded_file.name.split('.')[-1].lower()
        
        if file_extension in ['txt', 'md']:
            # Text files
            content = str(uploaded_file.read(), "utf-8")
            
        elif file_extension == 'pdf':
            # PDF files - you'll need PyPDF2
            try:
                import PyPDF2
                import io
                
                pdf_reader = PyPDF2.PdfReader(io.BytesIO(uploaded_file.read()))
                content = ""
                for page in pdf_reader.pages:
                    content += page.extract_text() + "\n"
                    
            except ImportError:
                st.error("PyPDF2 not available. PDF upload disabled.")
                return None
                
        elif file_extension == 'docx':
            # DOCX files - you'll need python-docx
            try:
                from docx import Document
                import io
                
                doc = Document(io.BytesIO(uploaded_file.read()))
                content = "\n".join([paragraph.text for paragraph in doc.paragraphs])
                
            except ImportError:
                st.error("python-docx not available. DOCX upload disabled.")
                return None
        else:
            st.error(f"Unsupported file type: {file_extension}")
            return None
            
        # Create document object
        class UploadedDocument:
            def __init__(self, content, filename):
                self.page_content = content
                self.metadata = {"source": f"uploaded_{filename}", "type": "uploaded"}
        
        return UploadedDocument(content, uploaded_file.name)
        
    except Exception as e:
        st.error(f"Error processing {uploaded_file.name}: {str(e)}")
        return None

def load_frontend():
    cfg = load_config() if os.path.exists("config.yaml") else {}
    
    # Initialize session state for uploaded documents
    if 'uploaded_docs' not in st.session_state:
        st.session_state.uploaded_docs = []
    if 'upload_counter' not in st.session_state:
        st.session_state.upload_counter = 0
    
    # --------- Sidebar: Settings and Upload ---------
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Mode selection
        mode = st.radio(
            "Response Mode",
            ["Interview", "Story", "FastFacts", "HumbleBrag", "Reflect"],
            index=0
        )
        st.caption("Mode tweaks tone & structure of answers.")
        
        st.divider()
        
        # Document upload section
        st.header("📁 Add Training Data")
        st.caption("Upload documents to expand Grant's knowledge base")
        st.info("💡 Files are processed in-memory for this session")
        
        # File uploader
        uploaded_files = st.file_uploader(
            "Choose files",
            type=['txt', 'pdf', 'docx', 'md'],
            accept_multiple_files=True,
            help="Supported formats: TXT, PDF, DOCX, Markdown",
            key=f"file_uploader_{st.session_state.upload_counter}"
        )
        
        # Process uploaded files
        if uploaded_files:
            new_docs = []
            
            for uploaded_file in uploaded_files:
                # Check if already processed
                already_uploaded = any(
                    doc.metadata.get("source") == f"uploaded_{uploaded_file.name}" 
                    for doc in st.session_state.uploaded_docs
                )
                
                if not already_uploaded:
                    doc = process_uploaded_file(uploaded_file)
                    if doc:
                        new_docs.append(doc)
                        st.success(f"✅ Processed: {uploaded_file.name}")
                else:
                    st.info(f"📄 Already uploaded: {uploaded_file.name}")
            
            if new_docs:
                st.session_state.uploaded_docs.extend(new_docs)
                # Increment counter to force re-initialization
                if 'docs_changed' not in st.session_state:
                    st.session_state.docs_changed = 0
                st.session_state.docs_changed += 1
                
                st.success(f"🎉 Added {len(new_docs)} new documents!")
        
        # Show document statistics
        total_original = len([f for f in os.listdir("data") if f.lower().endswith(('.txt', '.pdf', '.docx', '.md'))]) if os.path.exists("data") else 0
        total_uploaded = len(st.session_state.uploaded_docs)
        total_docs = total_original + total_uploaded
        
        st.metric("📊 Total Documents", total_docs, delta=total_uploaded if total_uploaded > 0 else None)
        
        if total_uploaded > 0:
            st.write("**Uploaded this session:**")
            for doc in st.session_state.uploaded_docs:
                filename = doc.metadata["source"].replace("uploaded_", "")
                content_length = len(doc.page_content)
                st.write(f"• {filename} ({content_length:,} chars)")
        
        st.divider()
        
        # Session management
        st.header("🔄 Session Management")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Clear Uploads"):
                st.session_state.uploaded_docs = []
                st.session_state.docs_changed = st.session_state.get('docs_changed', 0) + 1
                st.success("Cleared uploaded documents!")
                st.rerun()
        
        with col2:
            if st.button("🔄 Refresh"):
                st.session_state.docs_changed = st.session_state.get('docs_changed', 0) + 1
                st.success("Refreshing knowledge base!")
                st.rerun()
        
        # Download processed content (for backup)
        if st.session_state.uploaded_docs:
            if st.button("💾 Download Session Data"):
                combined_content = "\n\n---\n\n".join([
                    f"FILE: {doc.metadata['source']}\n\n{doc.page_content}"
                    for doc in st.session_state.uploaded_docs
                ])
                
                st.download_button(
                    label="📄 Download All Uploaded Content",
                    data=combined_content,
                    file_name="uploaded_session_data.txt",
                    mime="text/plain"
                )
    
    # Return mode and uploaded documents
    return mode, st.session_state.uploaded_docs