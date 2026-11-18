"""REST API for phishing URL scoring.

Run:  uvicorn phishdetect.api:app --host 0.0.0.0 --port 8000

Environment variables
  PHISHDETECT_MODELS_DIR   directory with lexical.joblib / full.joblib (default: ./models)
  PHISHDETECT_API_KEY      if set, every /v1 request needs header X-API-Key with this value
  PHISHDETECT_RATE_LIMIT   requests per minute per client IP for /v1 (default 120; 0 = off)
  PHISHDETECT_LOOKUP_TIMEOUT  seconds per WHOIS/DNS lookup (default 4)
  PHISHDETECT_DOCS         set to 0 to disable /docs and /openapi.json

Security notes
  * The service never fetches the submitted URL. It only performs DNS and RDAP/WHOIS
    queries about the domain, so it cannot be used as an SSRF proxy.
  * Input is length-limited and restricted to http(s) URLs.
"""
from __future__ import annotations

import concurrent.futures as cf
import hmac
import logging
import os
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from .config import ROOT
from .scorer import Scorer
from .url_features import MAX_URL_LENGTH, InvalidURLError, normalize_url

log = logging.getLogger("phishdetect.api")
MAX_BATCH = 50


class ScoreRequest(BaseModel):
    url: str = Field(..., min_length=1, max_length=MAX_URL_LENGTH, examples=["http://paypal-login-verify.xyz/signin"])
    enrich: bool = Field(True, description="Run live WHOIS + DNS lookups and use the full model")


class BatchRequest(BaseModel):
    urls: list[str] = Field(..., min_length=1, max_length=MAX_BATCH)
    enrich: bool = True


class RateLimiter:
    """Fixed-window per-client limiter. In-memory, so per process; put a gateway
    in front for multi-replica deployments."""

    def __init__(self, per_minute: int):
        self.per_minute = per_minute
        self._hits: dict[str, tuple[int, int]] = {}
        self._lock = threading.Lock()

    def allow(self, key: str) -> bool:
        if self.per_minute <= 0:
            return True
        window = int(time.time() // 60)
        with self._lock:
            w, n = self._hits.get(key, (window, 0))
            n = n + 1 if w == window else 1
            self._hits[key] = (window, n)
            if len(self._hits) > 50_000:  # bound memory
                self._hits = {k: v for k, v in self._hits.items() if v[0] == window}
            return n <= self.per_minute


@asynccontextmanager
async def lifespan(app: FastAPI):
    if getattr(app.state, "scorer", None) is None:  # tests may inject one
        models_dir = Path(os.environ.get("PHISHDETECT_MODELS_DIR", ROOT / "models"))
        app.state.scorer = Scorer(models_dir, timeout=float(os.environ.get("PHISHDETECT_LOOKUP_TIMEOUT", "4")))
    app.state.batch_pool = cf.ThreadPoolExecutor(max_workers=8, thread_name_prefix="batch")
    yield
    app.state.batch_pool.shutdown(wait=False, cancel_futures=True)
    app.state.scorer.close()


_docs = os.environ.get("PHISHDETECT_DOCS", "1") != "0"
app = FastAPI(
    title="Phishing URL Detection API",
    version="1.0.0",
    description="Scores URLs with a gradient-boosting model over domain-lexical, WHOIS (RDAP) and DNS features.",
    lifespan=lifespan,
    docs_url="/docs" if _docs else None,
    redoc_url=None,
    openapi_url="/openapi.json" if _docs else None,
)
app.state.scorer = None
limiter = RateLimiter(int(os.environ.get("PHISHDETECT_RATE_LIMIT", "120")))


@app.middleware("http")
async def security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Cache-Control"] = "no-store"
    response.headers["Cross-Origin-Resource-Policy"] = "same-origin"
    response.headers["Permissions-Policy"] = "geolocation=(), camera=(), microphone=()"
    if request.url.path.startswith("/docs"):
        # Swagger UI loads its assets from a CDN.
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
            "style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; img-src 'self' data: https://fastapi.tiangolo.com; "
            "frame-ancestors 'none'")
    else:
        response.headers["Content-Security-Policy"] = "default-src 'none'; frame-ancestors 'none'"
    return response


def guard(request: Request) -> None:
    expected = os.environ.get("PHISHDETECT_API_KEY")
    if expected:
        supplied = request.headers.get("x-api-key", "")
        if not hmac.compare_digest(supplied.encode(), expected.encode()):
            raise HTTPException(status_code=401, detail="invalid or missing API key")
    client = request.client.host if request.client else "unknown"
    if not limiter.allow(client):
        raise HTTPException(status_code=429, detail="rate limit exceeded")


def _score_one(scorer: Scorer, url: str, enrich: bool) -> dict:
    try:
        clean = normalize_url(url)
    except InvalidURLError as exc:
        return {"url": url, "error": str(exc)}
    return scorer.score(clean, enrich_lookups=enrich)


@app.get("/health")
def health(request: Request):
    s: Scorer = request.app.state.scorer
    return {"status": "ok", "models": {"lexical": True, "full": s.full is not None}}


@app.get("/v1/model", dependencies=[Depends(guard)])
def model_info(request: Request):
    s: Scorer = request.app.state.scorer
    out = {}
    for m in (s.lexical, s.full):
        if m is not None:
            out[m.name] = {"version": m.version, "trained_at": m.trained_at, "threshold": m.threshold,
                           "n_features": len(m.features), "test_metrics": m.metrics}
    return out


@app.post("/v1/score", dependencies=[Depends(guard)])
def score(req: ScoreRequest, request: Request):
    result = _score_one(request.app.state.scorer, req.url, req.enrich)
    if "error" in result:
        raise HTTPException(status_code=422, detail=result["error"])
    return result


@app.post("/v1/score/batch", dependencies=[Depends(guard)])
def score_batch(req: BatchRequest, request: Request):
    for u in req.urls:
        if len(u) > MAX_URL_LENGTH:
            raise HTTPException(status_code=422, detail=f"URL longer than {MAX_URL_LENGTH} characters")
    pool: cf.ThreadPoolExecutor = request.app.state.batch_pool
    results = list(pool.map(lambda u: _score_one(request.app.state.scorer, u, req.enrich), req.urls))
    return {"results": results}


@app.exception_handler(Exception)
async def unhandled(request: Request, exc: Exception):
    log.exception("unhandled error on %s", request.url.path)
    return JSONResponse(status_code=500, content={"detail": "internal error"})
