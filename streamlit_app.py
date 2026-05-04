import os
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from embedding_utils import build_embeddings

load_dotenv()

INDEX_DIR = Path("faiss_index")
DEFAULT_GROQ_MODEL = "llama-3.1-8b-instant"

# ── Page config ──
st.set_page_config(page_title="DocuMind AI", page_icon="🧠", layout="wide")

# ── Premium CSS ──
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ── Root vars ── */
:root {
    --bg-primary: #0f0f1a;
    --bg-card: rgba(255,255,255,0.03);
    --border: rgba(255,255,255,0.06);
    --accent-1: #7c3aed;
    --accent-2: #06b6d4;
    --accent-3: #ec4899;
    --text-primary: #f1f5f9;
    --text-secondary: #94a3b8;
}

.stApp {
    font-family: 'Inter', sans-serif !important;
}

/* ── Hide default header/footer ── */
header[data-testid="stHeader"] { background: transparent; }
#MainMenu, footer { display: none; }

/* ── Animated gradient background ── */
.main-bg {
    position: fixed; top: 0; left: 0; right: 0; bottom: 0; z-index: -1;
    background: linear-gradient(135deg, #0f0f1a 0%, #1a1025 30%, #0f1a2e 60%, #0f0f1a 100%);
}
.main-bg::before {
    content: ''; position: absolute; top: -50%; left: -50%; width: 200%; height: 200%;
    background: radial-gradient(circle at 20% 50%, rgba(124,58,237,0.08) 0%, transparent 50%),
                radial-gradient(circle at 80% 20%, rgba(6,182,212,0.06) 0%, transparent 50%),
                radial-gradient(circle at 50% 80%, rgba(236,72,153,0.05) 0%, transparent 50%);
    animation: floatGlow 20s ease-in-out infinite;
}
@keyframes floatGlow {
    0%, 100% { transform: translate(0, 0) rotate(0deg); }
    33% { transform: translate(30px, -30px) rotate(2deg); }
    66% { transform: translate(-20px, 20px) rotate(-1deg); }
}

/* ── Hero ── */
.hero {
    text-align: center; padding: 1.5rem 0 1rem;
}
.hero-icon {
    width: 64px; height: 64px; border-radius: 20px; margin: 0 auto 1rem;
    background: linear-gradient(135deg, var(--accent-1), var(--accent-2));
    display: flex; align-items: center; justify-content: center;
    font-size: 1.8rem; box-shadow: 0 8px 32px rgba(124,58,237,0.3);
    animation: pulse-glow 3s ease-in-out infinite;
}
@keyframes pulse-glow {
    0%, 100% { box-shadow: 0 8px 32px rgba(124,58,237,0.3); }
    50% { box-shadow: 0 8px 48px rgba(124,58,237,0.5); }
}
.hero h1 {
    font-size: 2rem; font-weight: 800; margin: 0;
    background: linear-gradient(135deg, #c084fc, #22d3ee, #f472b6);
    background-size: 200% 200%;
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    animation: gradient-shift 4s ease infinite;
}
@keyframes gradient-shift {
    0%, 100% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
}
.hero p { color: var(--text-secondary); font-size: 0.95rem; margin-top: 0.3rem; }

/* ── Stats row ── */
.stats-row { display: flex; gap: 12px; justify-content: center; margin: 0.8rem 0 1.2rem; flex-wrap: wrap; }
.stat-chip {
    display: inline-flex; align-items: center; gap: 6px;
    padding: 6px 16px; border-radius: 999px; font-size: 0.78rem; font-weight: 500;
    backdrop-filter: blur(12px);
}
.chip-ready { background: rgba(16,185,129,0.1); color: #34d399; border: 1px solid rgba(16,185,129,0.2); }
.chip-waiting { background: rgba(251,191,36,0.1); color: #fbbf24; border: 1px solid rgba(251,191,36,0.2); }
.chip-info { background: rgba(99,102,241,0.1); color: #a5b4fc; border: 1px solid rgba(99,102,241,0.15); }

/* ── Chat container ── */
.chat-area {
    max-width: 800px; margin: 0 auto;
    min-height: 350px; max-height: 55vh; overflow-y: auto;
    padding: 1rem; border-radius: 16px;
    background: var(--bg-card); border: 1px solid var(--border);
    backdrop-filter: blur(16px);
    scrollbar-width: thin; scrollbar-color: rgba(255,255,255,0.1) transparent;
}

/* ── Messages ── */
.msg-row { display: flex; gap: 10px; margin-bottom: 14px; animation: fadeIn 0.4s ease; }
.msg-row.user { flex-direction: row-reverse; }
@keyframes fadeIn { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: none; } }

.avatar {
    width: 36px; height: 36px; border-radius: 12px; flex-shrink: 0;
    display: flex; align-items: center; justify-content: center;
    font-size: 1rem; font-weight: 600;
}
.avatar-user { background: linear-gradient(135deg, var(--accent-1), var(--accent-3)); color: #fff; }
.avatar-bot { background: linear-gradient(135deg, var(--accent-2), #3b82f6); color: #fff; }

.bubble {
    padding: 12px 16px; border-radius: 16px; font-size: 0.9rem; line-height: 1.6;
    max-width: 75%; word-wrap: break-word;
}
.bubble-user {
    background: linear-gradient(135deg, var(--accent-1), #5b21b6);
    color: #fff; border-bottom-right-radius: 4px;
    box-shadow: 0 4px 20px rgba(124,58,237,0.2);
}
.bubble-bot {
    background: rgba(255,255,255,0.05); color: var(--text-primary);
    border: 1px solid var(--border); border-bottom-left-radius: 4px;
    backdrop-filter: blur(8px);
}

/* ── Empty state ── */
.empty-state {
    text-align: center; padding: 3rem 1rem; color: var(--text-secondary);
}
.empty-state .icon { font-size: 3rem; margin-bottom: 1rem; opacity: 0.4; }
.empty-state h3 { color: var(--text-primary); font-weight: 600; margin: 0 0 0.5rem; font-size: 1.1rem; }
.empty-state p { font-size: 0.85rem; max-width: 350px; margin: 0 auto; }

.suggestion-chips { display: flex; gap: 8px; justify-content: center; margin-top: 1.2rem; flex-wrap: wrap; }
.suggestion {
    padding: 8px 16px; border-radius: 999px; font-size: 0.78rem;
    background: rgba(124,58,237,0.08); border: 1px solid rgba(124,58,237,0.15);
    color: #c084fc; cursor: default; transition: all 0.2s;
}
.suggestion:hover { background: rgba(124,58,237,0.15); border-color: rgba(124,58,237,0.3); }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: rgba(15,15,26,0.95) !important; border-right: 1px solid var(--border);
}
.sidebar-title {
    font-size: 0.85rem; font-weight: 600; color: var(--text-secondary);
    text-transform: uppercase; letter-spacing: 1.5px; margin-bottom: 0.8rem;
}
.file-info-card {
    background: linear-gradient(135deg, rgba(124,58,237,0.08), rgba(6,182,212,0.08));
    border: 1px solid rgba(124,58,237,0.15); border-radius: 12px; padding: 14px; margin-top: 12px;
}
.file-info-card .fname { font-weight: 600; color: var(--text-primary); font-size: 0.9rem; }
.file-info-card .fmeta { color: var(--text-secondary); font-size: 0.8rem; margin-top: 4px; }

/* ── Watermark ── */
.watermark {
    position: fixed; bottom: 12px; right: 20px; z-index: 999;
    font-size: 0.7rem; font-weight: 500; letter-spacing: 0.5px;
    color: rgba(148,163,184,0.35);
    font-family: 'Inter', sans-serif;
    pointer-events: none; user-select: none;
}
</style>
<div class="main-bg"></div>
""", unsafe_allow_html=True)


# ── Helpers ──
@st.cache_resource(show_spinner="Loading embedding model...")
def get_embeddings():
    return build_embeddings()

def ingest_pdf_bytes(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name
    loader = PyPDFLoader(tmp_path)
    documents = loader.load()
    os.unlink(tmp_path)
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)
    embeddings = get_embeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(str(INDEX_DIR))
    return vectorstore, len(chunks), len(documents)

def load_existing_index():
    if not INDEX_DIR.exists():
        return None
    embeddings = get_embeddings()
    return FAISS.load_local(str(INDEX_DIR), embeddings, allow_dangerous_deserialization=True)

def build_chain(vectorstore):
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        st.error("`.env` file mein `GROQ_API_KEY` missing hai!")
        st.stop()
    groq_model = os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL)
    llm = ChatGroq(model=groq_model, api_key=api_key)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful PDF chatbot. Answer questions based ONLY on the provided context. If the answer is not found in the context, clearly state that it was not found in the PDF."),
        ("human", "Context:\n{context}\n\nQuestion: {input}"),
    ])
    document_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        combine_docs_chain=document_chain,
    )


# ── Session state ──
for key, val in [("messages", []), ("vectorstore", None), ("chain", None),
                 ("pdf_name", None), ("chunk_count", 0), ("page_count", 0)]:
    if key not in st.session_state:
        st.session_state[key] = val


# ── Sidebar ──
with st.sidebar:
    st.markdown('<p class="sidebar-title">📁 Document</p>', unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload PDF", type=["pdf"], label_visibility="collapsed")

    if uploaded is not None and uploaded.name != st.session_state.pdf_name:
        with st.spinner("⚡ Processing PDF..."):
            vs, chunks, pages = ingest_pdf_bytes(uploaded)
            st.session_state.vectorstore = vs
            st.session_state.chain = build_chain(vs)
            st.session_state.pdf_name = uploaded.name
            st.session_state.chunk_count = chunks
            st.session_state.page_count = pages
            st.session_state.messages = []
            st.rerun()

    if st.session_state.pdf_name:
        st.markdown(f"""
        <div class="file-info-card">
            <div class="fname">📄 {st.session_state.pdf_name}</div>
            <div class="fmeta">📦 {st.session_state.chunk_count} chunks &nbsp;·&nbsp; 📃 {st.session_state.page_count} pages</div>
        </div>""", unsafe_allow_html=True)

    if st.session_state.vectorstore is None and INDEX_DIR.exists():
        st.markdown("---")
        if st.button("📂 Load existing index", use_container_width=True):
            with st.spinner("Loading..."):
                vs = load_existing_index()
                if vs:
                    st.session_state.vectorstore = vs
                    st.session_state.chain = build_chain(vs)
                    st.session_state.pdf_name = "sample.pdf (pre-indexed)"
                    st.session_state.chunk_count = "—"
                    st.session_state.page_count = "—"
                    st.rerun()

    if st.session_state.messages:
        st.markdown("---")
        if st.button("🗑️ Clear chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()




# ── Hero ──
st.markdown("""
<div class="hero">
    <div class="hero-icon">🧠</div>
    <h1>DocuMind AI</h1>
    <p>Upload a PDF and ask anything — AI-powered instant answers</p>
</div>""", unsafe_allow_html=True)

# ── Status chips ──
if st.session_state.chain:
    chips = f'<span class="stat-chip chip-ready">● Ready</span>'
    if st.session_state.pdf_name:
        chips += f'<span class="stat-chip chip-info">📄 {st.session_state.pdf_name}</span>'
    st.markdown(f'<div class="stats-row">{chips}</div>', unsafe_allow_html=True)
else:
    st.markdown('<div class="stats-row"><span class="stat-chip chip-waiting">⏳ Upload a PDF to get started</span></div>', unsafe_allow_html=True)


# ── Chat area ──
chat_html = '<div class="chat-area">'

if not st.session_state.messages:
    chat_html += """
    <div class="empty-state">
        <div class="icon">💬</div>
        <h3>Start a conversation</h3>
        <p>Upload a PDF and ask questions about its content. I'll find answers for you instantly.</p>
        <div class="suggestion-chips">
            <span class="suggestion">📖 Summarize the document</span>
            <span class="suggestion">🔍 Key concepts</span>
            <span class="suggestion">❓ Main topics</span>
        </div>
    </div>"""
else:
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            chat_html += f'''
            <div class="msg-row user">
                <div class="avatar avatar-user">U</div>
                <div class="bubble bubble-user">{msg["content"]}</div>
            </div>'''
        else:
            chat_html += f'''
            <div class="msg-row">
                <div class="avatar avatar-bot">🤖</div>
                <div class="bubble bubble-bot">{msg["content"]}</div>
            </div>'''

chat_html += '</div>'
st.markdown(chat_html, unsafe_allow_html=True)


# ── Chat input ──
if question := st.chat_input("Ask a question about your PDF..."):
    if not st.session_state.chain:
        st.warning("⚠️ Please upload a PDF first or load the existing index.")
    else:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.spinner("🤔 Thinking..."):
            result = st.session_state.chain.invoke({"input": question})
            answer = result["answer"]
        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.rerun()


# ── Watermark ──
st.markdown('<div class="watermark">Made by Devansh</div>', unsafe_allow_html=True)
