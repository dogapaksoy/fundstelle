"""HTTP service: a search endpoint, a cited-answer endpoint, a small web page, metrics.

Configuration (environment variables):

- ``FUNDSTELLE_DATA``: data directory with ``corpus.jsonl`` (default ``data``)
- ``FUNDSTELLE_RETRIEVER``: e.g. ``bm25+dense:models/e5-small-gerlayqa`` (default ``bm25``)
- ``FUNDSTELLE_LLM``: ``llama:qwen3-4b`` for a local GGUF model, or
  ``openai:<model>@<base url>`` for any OpenAI-compatible server (key in
  ``OPENAI_API_KEY``). Unset: ``/ask`` is disabled and only search works.
- ``FUNDSTELLE_NORMS``: show the model the top N retrieved norms with all their Absätze
  (default 4); ``0`` shows only the retrieved Absätze
- ``FUNDSTELLE_NLI``: ``0`` to skip the per-sentence check (not recommended)

Every request is logged as one JSON line with its timings, so latency and abstention
rates can be followed in any log tool; ``/metrics`` exposes the same as Prometheus text.
"""

from __future__ import annotations

import json
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, PlainTextResponse
from pydantic import BaseModel, Field

from fundstelle.answer import Answerer, NLISupport, cite, whole_norms
from fundstelle.corpus import load
from fundstelle.llm import LlamaCppBackend, OpenAICompatibleBackend

log = logging.getLogger("fundstelle")
STATE: dict = {}
COUNTS = {"search": 0, "ask": 0, "ask_abstained": 0, "errors": 0}
SECONDS = {"search": 0.0, "ask": 0.0}


class Query(BaseModel):
    question: str = Field(min_length=10, max_length=4000)
    k: int = Field(default=8, ge=1, le=20)


def _llm(spec: str | None):
    if not spec:
        return None
    kind, _, rest = spec.partition(":")
    if kind == "llama":
        return LlamaCppBackend(rest or "qwen3-4b")
    if kind == "openai":
        model, _, base = rest.partition("@")
        return OpenAICompatibleBackend(model, base or "https://api.openai.com/v1")
    raise ValueError(f"unknown FUNDSTELLE_LLM {spec!r}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    from fundstelle.cli import build

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    data = Path(os.environ.get("FUNDSTELLE_DATA", "data"))
    passages = load(data / "corpus.jsonl")
    STATE["passages"] = passages
    STATE["by_norm"] = {}
    for p in passages:
        STATE["by_norm"].setdefault(p.norm_id, []).append(p)
    STATE["retriever"] = build(os.environ.get("FUNDSTELLE_RETRIEVER", "bm25"), passages, data)
    llm = _llm(os.environ.get("FUNDSTELLE_LLM"))
    support = NLISupport() if os.environ.get("FUNDSTELLE_NLI", "1") != "0" else None
    STATE["answerer"] = Answerer(llm, support) if llm else None
    yield


app = FastAPI(title="fundstelle", lifespan=lifespan)


def _search(question: str, k: int):
    passages = STATE["passages"]
    (ranking,) = STATE["retriever"].search([question], k)
    return [(passages[i], score) for i, score in ranking]


@app.get("/health")
def health() -> dict:
    return {"ok": True, "passages": len(STATE.get("passages", [])),
            "answers": STATE.get("answerer") is not None}  # fmt: skip


@app.post("/search")
def search(q: Query) -> dict:
    t0 = time.perf_counter()
    hits = _search(q.question, q.k)
    dt = time.perf_counter() - t0
    COUNTS["search"] += 1
    SECONDS["search"] += dt
    log.info(json.dumps({"event": "search", "seconds": round(dt, 3), "k": q.k}))
    return {"passages": [{"id": p.id, "cite": cite(p.id), "title": p.title, "text": p.text,
                          "score": round(s, 4)} for p, s in hits]}  # fmt: skip


@app.post("/ask")
def ask(q: Query) -> dict:
    if STATE.get("answerer") is None:
        raise HTTPException(503, "no language model configured (set FUNDSTELLE_LLM)")
    t0 = time.perf_counter()
    try:
        hits = _search(q.question, q.k)
        t1 = time.perf_counter()
        shown = [p for p, _ in hits]
        norms = int(os.environ.get("FUNDSTELLE_NORMS", "4"))
        if norms:  # show the retrieved norms whole, not only the retrieved Absätze
            shown = whole_norms(shown, STATE["by_norm"], norms)
        a = STATE["answerer"](q.question, shown)
    except Exception:
        COUNTS["errors"] += 1
        raise
    t2 = time.perf_counter()
    COUNTS["ask"] += 1
    COUNTS["ask_abstained"] += a.abstained
    SECONDS["ask"] += t2 - t0
    trace = {"event": "ask", "retrieval_s": round(t1 - t0, 3), "llm_s": round(a.seconds, 3),
             "check_s": round(t2 - t1 - a.seconds, 3), "drafted": len(a.draft),
             "kept": len(a.kept), "abstained": a.abstained}  # fmt: skip
    log.info(json.dumps(trace))
    return {
        "answer": a.text(),
        "abstained": a.abstained,
        "points": [
            {
                "cite": cite(p.cite),
                "rule": p.rule,
                "relevance": p.relevance,
                "support": p.support,
                "missing": p.missing,
                "kept": p in a.kept,
            }
            for p in a.draft
        ],
        "passages": [
            {"id": p.id, "cite": cite(p.id), "title": p.title, "text": p.text} for p in shown
        ],
        "trace": trace,
    }


@app.get("/metrics", response_class=PlainTextResponse)
def metrics() -> str:
    lines = []
    for k, v in COUNTS.items():
        lines.append(f'fundstelle_requests_total{{kind="{k}"}} {v}')
    for k, v in SECONDS.items():
        lines.append(f'fundstelle_seconds_total{{endpoint="{k}"}} {v:.3f}')
    return "\n".join(lines) + "\n"


@app.get("/", response_class=HTMLResponse)
def page() -> str:
    return (Path(__file__).parent / "page.html").read_text(encoding="utf-8")
