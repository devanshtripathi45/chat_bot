# 🧠 DocuMind AI — PDF Chatbot

An AI-powered PDF chatbot that lets you upload any PDF document and ask questions about its content. Built with **LangChain**, **FAISS** vector database, **Groq LLM**, and a beautiful **Streamlit** dark-themed UI.

![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.57-red?logo=streamlit)
![LangChain](https://img.shields.io/badge/LangChain-Powered-green)
![Groq](https://img.shields.io/badge/Groq-LLM-orange)

## ✨ Features

- 📄 **PDF Upload** — Upload any PDF and get instant answers
- 🤖 **AI-Powered Q&A** — Uses Groq's Llama 3.1 for intelligent answers
- 🔍 **Semantic Search** — FAISS vector search for accurate context retrieval
- 🎨 **Premium Dark UI** — Glassmorphism, animated gradients, chat bubbles
- ⚡ **Fast** — Groq's inference speed for near-instant responses

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| **Python** | Backend logic |
| **Streamlit** | Web UI framework |
| **LangChain** | LLM orchestration & RAG pipeline |
| **FAISS** | Vector similarity search |
| **Groq API** | LLM inference (Llama 3.1) |
| **HuggingFace** | Sentence embeddings (all-MiniLM-L6-v2) |

## 🚀 Setup & Run

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/documind-ai.git
cd documind-ai
```

### 2. Create virtual environment
```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate  # Mac/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up environment variables
Create a `.env` file:
```
GROQ_API_KEY=your_groq_api_key_here
```
Get your free API key from [console.groq.com](https://console.groq.com)

### 5. Run the app
```bash
streamlit run streamlit_app.py
```

Open `http://localhost:8501` in your browser 🎉

## 📁 Project Structure

```
documind-ai/
├── .env                  # API keys (not in git)
├── .gitignore
├── .streamlit/
│   └── config.toml       # Streamlit dark theme config
├── app.py                # CLI chatbot version
├── embedding_utils.py    # Text normalization & embeddings
├── ingest.py             # PDF ingestion pipeline
├── streamlit_app.py      # Main Streamlit UI
├── requirements.txt
└── sample.pdf            # Sample PDF for testing
```

## 🔧 How It Works

1. **PDF Upload** → PyPDF extracts text from PDF
2. **Chunking** → Text split into 500-char chunks with overlap
3. **Embedding** → Each chunk converted to 384-dim vector (MiniLM)
4. **Storage** → Vectors stored in FAISS index
5. **Query** → User question embedded → top-3 similar chunks retrieved
6. **Answer** → Groq LLM generates answer from retrieved context

## 📸 Screenshots

> Add your screenshots here after deployment

## 👤 Author

**Devansh**

---

⭐ Star this repo if you found it useful!
