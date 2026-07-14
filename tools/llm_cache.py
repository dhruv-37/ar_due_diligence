"""tools/llm_cache.py — sqlite cache for LLM .invoke() calls (key = sha256(model+prompt))."""
import hashlib
import sqlite3
import time
from pathlib import Path
from types import SimpleNamespace

_DB = Path(__file__).resolve().parent.parent / "data" / "llm_cache.sqlite3"
_DB.parent.mkdir(parents=True, exist_ok=True)

# Retry only on transient server-overload errors (503 UNAVAILABLE — "high
# demand, try again later"). Quota errors (429 RESOURCE_EXHAUSTED) are NOT
# retried here: those are already handled by the llm object's own
# max_retries=1, and hammering them further just burns quota for no gain.
_OVERLOAD_MARKERS = ("503", "UNAVAILABLE", "overloaded")
_MAX_ATTEMPTS = 4
_BASE_DELAY_SEC = 5


def _is_overload_error(exc: Exception) -> bool:
    msg = str(exc)
    return any(marker in msg for marker in _OVERLOAD_MARKERS)


def _invoke_with_retry(llm, prompt: str):
    """llm.invoke(prompt) with exponential backoff, retried only for
    transient 503/overload errors. Other errors (quota, auth, etc.) raise
    immediately."""
    last_exc = None
    for attempt in range(_MAX_ATTEMPTS):
        try:
            return llm.invoke(prompt)
        except Exception as exc:
            if not _is_overload_error(exc) or attempt == _MAX_ATTEMPTS - 1:
                raise
            last_exc = exc
            delay = _BASE_DELAY_SEC * (2 ** attempt)
            print(f"  ⚠️  Gemini overloaded (503) — retrying in {delay}s "
                  f"(attempt {attempt + 1}/{_MAX_ATTEMPTS})...")
            time.sleep(delay)
    raise last_exc  # pragma: no cover — loop always returns or raises above


def _conn():
    c = sqlite3.connect(_DB)
    c.execute("CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, content TEXT)")
    return c

def _key(model: str, prompt: str) -> str:
    return hashlib.sha256(f"{model}::{prompt}".encode("utf-8")).hexdigest()


def cached_invoke(llm, prompt: str, model: str):
    """Drop-in replacement for llm.invoke(prompt) with sqlite caching and
    retry-on-overload."""
    k = _key(model, prompt)
    conn = _conn()
    row = conn.execute("SELECT content FROM cache WHERE key=?", (k,)).fetchone()
    if row:
        conn.close()
        return SimpleNamespace(content=row[0])

    response = _invoke_with_retry(llm, prompt)
    content = response.content if hasattr(response, "content") else str(response)
    conn.execute("INSERT OR REPLACE INTO cache (key, content) VALUES (?, ?)", (k, content))
    conn.commit()
    conn.close()
    return SimpleNamespace(content=content)