import os
import shutil
import streamlit as st
from PyPDF2 import PdfReader

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate


# ----------------- Config (local & free) -----------------
INDEX_DIR = "faiss_index"
# Fast CPU-friendly embedding model; you can swap to "BAAI/bge-small-en-v1.5" for higher quality (then rebuild index once).
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
# Local chat model served by Ollama (make sure you pulled it once: `ollama run llama3.1:8b`)
LOCAL_CHAT_MODEL = "llama3.1:8b"


# ----------------- Helpers -----------------
def get_pdf_text(pdf_docs) -> str:
    """Extract text from uploaded PDFs."""
    text_parts = []
    for pdf in pdf_docs or []:
        reader = PdfReader(pdf)
        for page in reader.pages:
            page_text = page.extract_text() or ""
            text_parts.append(page_text)
    return "\n".join(text_parts)


def get_text_chunks(text: str):
    """
    Split text into overlapping chunks for embedding.
    Smaller chunks help the local model stay within context length and speed up inference.
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=300)
    return splitter.split_text(text)


def build_or_load_vector_store(text_chunks=None, rebuild: bool = False) -> FAISS:
    """
    Create the FAISS store from text chunks if needed, otherwise load it.
    100% local: HuggingFace embeddings + FAISS.
    """
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)

    if rebuild and os.path.exists(INDEX_DIR):
        shutil.rmtree(INDEX_DIR)

    if not os.path.exists(INDEX_DIR):
        if not text_chunks:
            raise ValueError(
                "No existing FAISS index found and no text to build one from. "
                "Upload PDFs and click 'Submit & Process' first."
            )
        vs = FAISS.from_texts(text_chunks, embedding=embeddings)
        vs.save_local(INDEX_DIR)
        return vs

    # NOTE: This uses pickle for metadata; safe for your own local index.
    return FAISS.load_local(
        INDEX_DIR,
        embeddings,
        allow_dangerous_deserialization=True,
    )


def get_qa_chain():
    """
    Build a simple RAG chain: retrieve -> stuff context into prompt -> local LLM answers.
    """
    prompt_template = """
Answer the question as completely as possible using ONLY the provided context.
If the answer is not in the context, reply exactly: "Answer is not available in the context".

Context:
{context}

Question:
{question}

Answer:
"""
    llm = ChatOllama(
        model=LOCAL_CHAT_MODEL,
        temperature=0.3,
    )
    prompt = PromptTemplate(
        template=prompt_template.strip(),
        input_variables=["context", "question"],
    )
    return load_qa_chain(llm, chain_type="stuff", prompt=prompt)


def shrink_docs(docs, max_chars=12000):
    """
    Keep combined retrieved text under a character cap (helps speed & fits local model context).
    """
    kept, total = [], 0
    for d in docs:
        t = d.page_content or ""
        if not t:
            continue
        if total + len(t) <= max_chars:
            kept.append(d)
            total += len(t)
        else:
            remain = max_chars - total
            if remain > 0:
                d.page_content = t[:remain]
                kept.append(d)
            break
    return kept


def answer_question(user_question: str, max_docs_k=3, max_chars=12000):
    """
    Retrieve relevant chunks from FAISS and ask the local LLM via Ollama.
    """
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)
    vs = FAISS.load_local(
        INDEX_DIR,
        embeddings,
        allow_dangerous_deserialization=True,
    )

    docs = vs.similarity_search(user_question, k=max_docs_k)
    docs = shrink_docs(docs, max_chars=max_chars)

    chain = get_qa_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    return response.get("output_text", "")


# ----------------- Streamlit App -----------------
def main():
    st.set_page_config(page_title="Chat with PDF • Local (Free)")
    st.header("Chat with PDF (100% Local, Free) 💻")

    st.caption(
        "This version uses a local LLM via Ollama and local embeddings via Sentence Transformers. "
        "No API keys, no quotas, no usage fees."
    )

    user_question = st.text_input("Ask a question about your uploaded PDFs")

    if user_question:
        try:
            reply = answer_question(user_question)
            st.write("**Reply:** ", reply)
        except Exception as e:
            st.error(f"Error while answering: {e}")

    with st.sidebar:
        st.title("Menu")
        pdf_docs = st.file_uploader(
            "Upload your PDF files, then click Submit & Process",
            accept_multiple_files=True,
            type=["pdf"],
        )

        rebuild_index = st.checkbox(
            "Rebuild FAISS Index (check after changing embedding model)",
            value=False
        )

        if st.button("Submit & Process"):
            if not pdf_docs:
                st.warning("Please upload at least one PDF.")
            else:
                with st.spinner("Processing..."):
                    try:
                        raw_text = get_pdf_text(pdf_docs)
                        if not raw_text.strip():
                            st.error("No extractable text found in the uploaded PDFs.")
                            return
                        chunks = get_text_chunks(raw_text)
                        build_or_load_vector_store(chunks, rebuild=rebuild_index)
                        st.success("Done! You can now ask questions.")
                    except Exception as e:
                        st.error(f"Indexing failed: {e}")


if __name__ == "__main__":
    main()
