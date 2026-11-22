"""HTTP API.   uvicorn finreg.api:app --port 8000

POST /search {"question", "k"}  -> retrieved chunks with citations (no LLM, no cost)
POST /ask    {"question", "k"}  -> answer with [n] citations mapped to legal references
GET  /health

The index, embedder and LLM are created on first use and can be replaced in tests via
`app.state`. /ask returns 503 when no LLM key is configured; /search keeps working.
"""
import os
import time

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from finreg import answer as ans
from finreg import index

app = FastAPI(title="eu-finreg-rag")
CHUNK_CHARS = int(os.environ.get("FINREG_CHUNK_CHARS", "512"))
INDEX_DIR = os.environ.get("FINREG_INDEX_DIR", "data/index")


class Query(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    k: int = Field(default=5, ge=1, le=20)


def _deps(request: Request):
    st = request.app.state
    if not hasattr(st, "client"):
        from finreg.embed import FastEmbedder
        st.client, st.embedder = index.open_client(INDEX_DIR), FastEmbedder()
        st.collection = index.collection_name(CHUNK_CHARS)
    return st


def _hit_json(h: index.Hit) -> dict:
    return {"rank": h.rank, "score": round(h.score, 4), "citation": h.citation, "heading": h.heading, "text": h.text}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/search")
def search(q: Query, request: Request):
    st = _deps(request)
    start = time.perf_counter()
    hits = index.search(st.client, st.embedder, st.collection, q.question, q.k)
    return {"hits": [_hit_json(h) for h in hits], "latency_s": round(time.perf_counter() - start, 3)}


@app.post("/ask")
def ask(q: Query, request: Request):
    st = _deps(request)
    if not hasattr(st, "generate"):
        try:
            st.generate = ans.claude_generate()
        except RuntimeError as e:
            raise HTTPException(503, f"LLM not configured: {e}") from e
    start = time.perf_counter()
    hits = index.search(st.client, st.embedder, st.collection, q.question, q.k)
    a = ans.answer(q.question, hits, st.generate)
    return {"answer": a.text, "not_found": a.not_found, "citations": a.citations,
            "sources": [_hit_json(h) for h in hits], "cost_usd": round(a.cost_usd, 6),
            "latency_s": round(time.perf_counter() - start, 3)}
