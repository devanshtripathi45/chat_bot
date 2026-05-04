from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

from embedding_utils import build_embeddings

INDEX_DIR = Path("faiss_index")

def ingest_pdf(pdf_path):
    pdf_file = Path(pdf_path)
    if not pdf_file.exists():
        raise FileNotFoundError(f"PDF file nahi mili: {pdf_file}")

    # 1. PDF Load karo
    loader = PyPDFLoader(str(pdf_file))
    documents = loader.load()

    # 2. Chunks mein todo
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(documents)

    # 3. Embeddings banao
    embeddings = build_embeddings()

    # 4. FAISS mein save karo
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(str(INDEX_DIR))
    
    print(f"Done! {len(chunks)} chunks saved.")

if __name__ == "__main__":
    ingest_pdf("sample.pdf")
