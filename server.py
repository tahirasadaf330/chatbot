# server.py
import os
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

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

INDEX_DIR   = os.getenv("INDEX_DIR", "faiss_index")
CHAT_MODEL  = os.getenv("CHAT_MODEL", "gemini-1.5-flash")
EMBED_MODEL = os.getenv("EMBED_MODEL", "models/text-embedding-004")

ALLOW_ORIGINS = os.getenv("ALLOW_ORIGINS", "http://localhost:5500,http://127.0.0.1:5500,https://yourdomain.com")
ALLOW_ORIGINS_LIST = [o.strip() for o in ALLOW_ORIGINS.split(",") if o.strip()]

# ------------------ App ------------------
app = FastAPI(title="RAG Chat API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOW_ORIGINS_LIST,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

def shrink_docs(docs, max_chars=12000):
    kept, total = [], 0
    for d in docs:
        t = (d.page_content or "")
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

class AskIn(BaseModel):
    question: str
    k: Optional[int] = 3
    max_chars: Optional[int] = 12000

class AskOut(BaseModel):
    answer: str

# Globals loaded at startup
_vs: Optional[FAISS] = None
_chain = None

@app.on_event("startup")
def on_startup():
    global _vs, _chain
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBED_MODEL)
    if not os.path.exists(INDEX_DIR):
        raise RuntimeError(
            f"Index folder '{INDEX_DIR}' not found. "
            "Run build_index.py first to create the FAISS index."
        )
    _vs = FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
    _chain = get_qa_chain()
    print("✅ Loaded FAISS index and QA chain.")

@app.get("/health")
def health():
    return {"status": "ok", "index_dir": INDEX_DIR, "model": CHAT_MODEL}

@app.post("/ask", response_model=AskOut)
def ask(req: AskIn, request: Request):
    try:
        docs = _vs.similarity_search(req.question, k=req.k or 3)
        docs = shrink_docs(docs, max_chars=req.max_chars or 12000)
        out = _chain(
            {"input_documents": docs, "question": req.question},
            return_only_outputs=True
        )
        return AskOut(answer=out.get("output_text", "").strip())
    except google_exceptions.ResourceExhausted:
        # 429 quota/rate limit
        return AskOut(answer="We’re experiencing high load. Please try again soon.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
