# build_index.py
import os
import glob
from dotenv import load_dotenv
from PyPDF2 import PdfReader

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("Missing GOOGLE_API_KEY in environment/.env")
genai.configure(api_key=GOOGLE_API_KEY)

EMBED_MODEL = os.getenv("EMBED_MODEL", "models/text-embedding-004")
INDEX_DIR   = os.getenv("INDEX_DIR", "faiss_index")
PDF_DIR     = os.getenv("PDF_DIR", "pdfs")  # put your PDFs in ./pdfs

def get_pdf_text(paths):
    parts = []
    for p in paths:
        with open(p, "rb") as f:
            reader = PdfReader(f)
            for page in reader.pages:
                parts.append(page.extract_text() or "")
    return "\n".join(parts)

def main():
    pdf_paths = glob.glob(os.path.join(PDF_DIR, "*.pdf"))
    if not pdf_paths:
        raise RuntimeError(f"No PDFs found in '{PDF_DIR}'. Put your files there first.")

    print(f"📄 Found {len(pdf_paths)} PDFs. Extracting text…")
    text = get_pdf_text(pdf_paths)
    if not text.strip():
        raise RuntimeError("No extractable text found in PDFs.")

    splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=300)
    chunks = splitter.split_text(text)
    print(f"✂️  Created {len(chunks)} chunks. Building embeddings…")

    embeddings = GoogleGenerativeAIEmbeddings(model=EMBED_MODEL)
    vs = FAISS.from_texts(chunks, embedding=embeddings)

    if os.path.exists(INDEX_DIR):
        print(f"🧹 Removing existing index at '{INDEX_DIR}'…")
        import shutil
        shutil.rmtree(INDEX_DIR)

    vs.save_local(INDEX_DIR)
    print(f"✅ Index saved to '{INDEX_DIR}'.")

if __name__ == "__main__":
    main()
