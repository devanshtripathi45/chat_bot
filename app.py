import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

from embedding_utils import build_embeddings

load_dotenv()

INDEX_DIR = Path("faiss_index")
DEFAULT_GROQ_MODEL = "llama-3.1-8b-instant"


def load_qa_chain():
    if not INDEX_DIR.exists():
        raise FileNotFoundError(
            "`faiss_index` nahi mila. Pehle `python ingest.py` chala kar PDF ingest karo."
        )

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError("`.env` me `GROQ_API_KEY` missing hai.")

    groq_model = os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL)

    # 1. FAISS load karo
    embeddings = build_embeddings()
    vectorstore = FAISS.load_local(
        str(INDEX_DIR),
        embeddings,
        allow_dangerous_deserialization=True,
    )

    # 2. Groq LLM setup karo
    llm = ChatGroq(
        model=groq_model,
        api_key=api_key,
    )

    # 3. QA Chain banao
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful PDF chatbot. Answer questions based ONLY on the provided context. If the answer is not found in the context, clearly state that it was not found in the PDF.",
            ),
            ("human", "Context:\n{context}\n\nQuestion: {input}"),
        ]
    )
    document_chain = create_stuff_documents_chain(llm, prompt)
    chain = create_retrieval_chain(
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        combine_docs_chain=document_chain,
    )
    return chain


if __name__ == "__main__":
    chain = load_qa_chain()
    print("PDF Chatbot Ready! (type 'exit' to quit)")

    while True:
        question = input("\nTu: ")
        if question == "exit":
            break
        answer = chain.invoke({"input": question})
        print(f"Bot: {answer['answer']}")
