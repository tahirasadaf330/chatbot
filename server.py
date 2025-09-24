# server.py — Upload a PDF, then ask questions (Gemini + FAISS, per-doc in-memory)
import os
import io
import uuid
from typing import Dict, List, Optional
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel
from dotenv import load_dotenv

from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import google.generativeai as genai
import google.api_core.exceptions as google_exceptions

# ------------------ Config ------------------
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("Missing GOOGLE_API_KEY in environment/.env")
genai.configure(api_key=GOOGLE_API_KEY)

CHAT_MODEL  = os.getenv("CHAT_MODEL", "gemini-1.5-flash")
EMBED_MODEL = os.getenv("EMBED_MODEL", "models/text-embedding-004")

ALLOW_ORIGINS = os.getenv(
    "ALLOW_ORIGINS",
    "http://localhost:9000,http://127.0.0.1:9000,http://localhost:8000,http://127.0.0.1:8000"
)
ALLOW_ORIGINS_LIST = [o.strip() for o in ALLOW_ORIGINS.split(",") if o.strip()]

# ------------------ App ------------------
app = FastAPI(title="PDF RAG Chat (Upload → Ask)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOW_ORIGINS_LIST or ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# static folder is optional (put your demo.html here if you want)
STATIC_DIR = Path("public")
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

@app.get("/", response_class=HTMLResponse)
def root():
    # serve demo if it exists, else a small landing page
    demo = STATIC_DIR / "demo.html"
    if demo.exists():
        return FileResponse(str(demo))
    return HTMLResponse("<h1>PDF RAG Chat</h1><p>Use /docs to try the API.</p>")

# ------------------ RAG bits ------------------
def get_qa_chain():
    prompt_template = """
Answer the question as completely as possible using ONLY the provided context.
If the answer is not in the context, reply exactly: "Answer is not available in the context".

Context:
{context}

Question:
{question}

Answer:
""".strip()
    model = ChatGoogleGenerativeAI(model=CHAT_MODEL, temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=150)
embeddings = GoogleGenerativeAIEmbeddings(model=EMBED_MODEL)
qa_chain = get_qa_chain()

# In-memory store of FAISS indexes per uploaded document
# doc_id -> FAISS vector store
DOC_STORES: Dict[str, FAISS] = {}

# ------------------ Helpers ------------------
def extract_pdf_text(file_bytes: bytes) -> str:
    try:
        reader = PdfReader(io.BytesIO(file_bytes))
        parts: List[str] = []
        for page in reader.pages:
            try:
                parts.append(page.extract_text() or "")
            except Exception:
                parts.append("")
        return "\n\n".join(parts).strip()
    except Exception:
        return ""

def build_faiss_for_text(text: str) -> FAISS:
    if not text.strip():
        raise ValueError("Empty text extracted from PDF.")
    chunks = text_splitter.split_text(text)
    if not chunks:
        raise ValueError("No chunks produced from PDF text.")
    vs = FAISS.from_texts(chunks, embeddings)
    return vs

def shrink_docs(docs, max_chars=12000):
    kept, total = [], 0
    for d in docs:
        t = (d.page_content or "")
        if not t:
            continue
        if total + len(t) <= max_chars:
            kept.append(d); total += len(t)
        else:
            remain = max_chars - total
            if remain > 0:
                d.page_content = t[:remain]
                kept.append(d)
            break
    return kept

# ------------------ Schemas ------------------
class UploadOut(BaseModel):
    doc_id: str
    chunks: int

class AskIn(BaseModel):
    doc_id: str
    question: str
    k: Optional[int] = 3
    max_chars: Optional[int] = 12000

class AskOut(BaseModel):
    answer: str

# ------------------ Routes ------------------
@app.get("/health")
def health():
    return {"status": "ok", "docs": len(DOC_STORES), "model": CHAT_MODEL}

@app.post("/upload", response_model=UploadOut)
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")
    data = await file.read()
    text = extract_pdf_text(data)
    if not text:
        raise HTTPException(status_code=400, detail="Could not extract text from PDF (is it scanned?).")

    try:
        vs = build_faiss_for_text(text)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Indexing failed: {e}")

    doc_id = str(uuid.uuid4())
    DOC_STORES[doc_id] = vs
    # we can estimate chunk count from the FAISS store (metadata not preserved by default)
    chunks = len(vs.docstore._dict) if hasattr(vs, "docstore") else 0
    return UploadOut(doc_id=doc_id, chunks=chunks)

@app.post("/ask", response_model=AskOut)
def ask(req: AskIn):
    if not req.doc_id or req.doc_id not in DOC_STORES:
        raise HTTPException(status_code=404, detail="Unknown doc_id. Upload a PDF first.")
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    vs = DOC_STORES[req.doc_id]
    try:
        docs = vs.similarity_search(req.question, k=req.k or 3)
        docs = shrink_docs(docs, max_chars=req.max_chars or 12000)
        out = qa_chain({"input_documents": docs, "question": req.question}, return_only_outputs=True)
        return AskOut(answer=(out.get("output_text") or "").strip())
    except google_exceptions.ResourceExhausted:
        return AskOut(answer="We’re experiencing high load. Please try again soon.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
