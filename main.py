import os
import time
import shutil
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import google.generativeai as genai
import google.api_core.exceptions as google_exceptions
import asyncio


# ------------ Utilities ------------
def ensure_event_loop():
    """
    Ensure the current thread has an asyncio event loop.
    Streamlit runs callbacks in a worker thread without a loop.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)


def shrink_docs(docs, max_chars=12000):
    """
    Keep only as much combined document text as fits within max_chars.
    This reduces input tokens to avoid hitting free-tier limits.
    """
    kept = []
    total = 0
    for d in docs:
        t = d.page_content or ""
        if not t:
            continue
        if total + len(t) <= max_chars:
            kept.append(d)
            total += len(t)
        else:
            # take a slice of the remainder if it helps
            remain = max_chars - total
            if remain > 0:
                d.page_content = t[:remain]
                kept.append(d)
            break
    return kept


# ------------- Config & Keys -------------
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError(
        "GOOGLE_API_KEY environment variable not found. "
        "Please add it to your .env or environment."
    )
genai.configure(api_key=api_key)

INDEX_DIR = "faiss_index"

# Prefer a lighter model to reduce quota usage; you can switch to "gemini-1.5-pro" if you have quota.
CHAT_MODEL = "gemini-1.5-flash"
EMBED_MODEL = "models/text-embedding-004"  # current embeddings model


# ------------- Helpers -------------
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
    We keep chunks smaller to reduce tokens passed into the model later.
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=300)
    return splitter.split_text(text)


def build_or_load_vector_store(text_chunks=None, rebuild: bool = False) -> FAISS:
    """
    Create the FAISS store from text chunks if needed, otherwise load it.
    We explicitly allow pickle deserialization for loading (only safe if you trust the index).
    """
    ensure_event_loop()
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBED_MODEL)

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

    return FAISS.load_local(
        INDEX_DIR,
        embeddings,
        allow_dangerous_deserialization=True,  # safe only for your own index
    )


def get_qa_chain():
    prompt_template = """
Answer the question as completely as possible using ONLY the provided context.
If the answer is not in the context, reply exactly: "Answer is not available in the context".

Context:
{context}

Question:
{question}

Answer:
"""
    ensure_event_loop()
    model = ChatGoogleGenerativeAI(model=CHAT_MODEL, temperature=0.3)
    prompt = PromptTemplate(template=prompt_template.strip(), input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)


def call_chain_with_retries(chain, docs, user_question, max_retries=4, initial_delay=2.0):
    """
    Retry on 429 rate-limit errors with exponential backoff.
    """
    delay = initial_delay
    for attempt in range(1, max_retries + 1):
        try:
            return chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        except google_exceptions.ResourceExhausted as e:
            # 429 from google.api_core
            if attempt == max_retries:
                raise
            time.sleep(delay)
            delay *= 2
        except Exception:
            # Other errors: surface immediately
            raise


def answer_question(user_question: str, max_docs_k=3, max_chars=12000):
    ensure_event_loop()
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBED_MODEL)

    vs = FAISS.load_local(
        INDEX_DIR,
        embeddings,
        allow_dangerous_deserialization=True,
    )

    # Fewer docs to keep token usage lower
    docs = vs.similarity_search(user_question, k=max_docs_k)
    docs = shrink_docs(docs, max_chars=max_chars)

    chain = get_qa_chain()

    # Retry on 429
    response = call_chain_with_retries(chain, docs, user_question)
    return response.get("output_text", "")


# ------------- Streamlit App -------------
def main():
    st.set_page_config(page_title="Chat with PDF • Gemini")
    st.header("Chat with PDF 💁")

    st.caption(
        "Tip: if you run into 429 rate limits, try fewer/lighter questions, "
        "use the default fast model, or enable billing to raise quotas."
    )

    user_question = st.text_input("Ask a question about your uploaded PDFs")

    if user_question:
        try:
            reply = answer_question(user_question)
            st.write("**Reply:** ", reply)
        except google_exceptions.ResourceExhausted as e:
            st.error(
                "Rate limit hit (429). I tried automatic retries with backoff but still exceeded your quota. "
                "Try again later, use fewer/shorter questions, or enable billing to raise limits."
            )
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
            "Rebuild FAISS Index (required after changing embedding model)",
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
