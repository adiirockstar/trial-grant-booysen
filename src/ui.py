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
