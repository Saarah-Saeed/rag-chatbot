"""
RAG Chatbot — Streamlit App
----------------------------
Loads a pre-built FAISS index (run build_index.ipynb first), then lets the user
chat with the document using Gemini, with conversation memory and multilingual
query support.
"""

import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langdetect import detect

# ── Page config ──────────────────────────────────────────────
st.set_page_config(page_title="RAG Chatbot", page_icon="🤖", layout="centered")

# ── Config ───────────────────────────────────────────────────
EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
LLM_MODEL = "models/gemini-2.5-flash"
INDEX_PATH = "faiss_index"


# ── Cached resources (loaded once) ──────────────────────────
@st.cache_resource(show_spinner="Loading embedding model...")
def load_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBED_MODEL)


@st.cache_resource(show_spinner="Loading vector store...")
def load_vectorstore(_embeddings):
    return FAISS.load_local(
        INDEX_PATH, _embeddings, allow_dangerous_deserialization=True
    )


@st.cache_resource(show_spinner="Connecting to Gemini...")
def load_llm(api_key: str):
    return ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        temperature=0.7,
        google_api_key=api_key,
    )


# ── Sidebar: API key ─────────────────────────────────────────
st.sidebar.title("⚙️ Settings")
api_key = st.sidebar.text_input("Gemini API Key", type="password")
top_k = st.sidebar.slider("Chunks to retrieve (k)", 1, 6, 3)

if st.sidebar.button("🗑️ Clear chat memory"):
    st.session_state.chat_history = []
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.caption(
    "Get a free Gemini API key at [aistudio.google.com](https://aistudio.google.com)"
)

# ── Main UI ──────────────────────────────────────────────────
st.title("🤖 RAG Chatbot")
st.caption("Ask questions about your document — answers are grounded in its content.")

if not api_key:
    st.info("👈 Enter your Gemini API key in the sidebar to get started.")
    st.stop()

# Load resources
embeddings = load_embeddings()
vectorstore = load_vectorstore(embeddings)
llm = load_llm(api_key)
retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})

# ── Chat memory ──────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list of (role, message)

# Render chat history
for role, msg in st.session_state.chat_history:
    with st.chat_message(role):
        st.markdown(msg)

# ── Core RAG function ───────────────────────────────────────
def ask_rag(question: str) -> str:
    # Detect language so the bot replies in the same language as the question
    try:
        lang = detect(question)
    except Exception:
        lang = "en"

    # Retrieve relevant chunks
    docs = retriever.invoke(question)
    context = "\n\n".join(doc.page_content for doc in docs)

    # Build short conversation history for context-aware follow-ups
    history_text = "\n".join(
        f"User: {q}\nBot: {a}" for q, a in st.session_state.chat_history[-6:]
        if isinstance(a, str)
    )

    prompt = f"""You are a helpful assistant. Answer the question using ONLY the
provided context. If the answer isn't in the context, say you don't know —
do not make things up.

Reply in this language code: {lang}

Previous conversation:
{history_text}

Context:
{context}

Question:
{question}

Answer:"""

    response = llm.invoke(prompt)
    return response.content


# ── Chat input ───────────────────────────────────────────────
user_input = st.chat_input("Ask something about the document...")

if user_input:
    st.session_state.chat_history.append(("user", user_input))
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer = ask_rag(user_input)
        st.markdown(answer)

    st.session_state.chat_history.append(("assistant", answer))