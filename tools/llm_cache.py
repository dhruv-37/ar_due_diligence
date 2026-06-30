"""tools/llm_cache.py — sqlite cache for LLM .invoke() calls (key = sha256(model+prompt))."""
import hashlib
import sqlite3
from pathlib import Path
from types import SimpleNamespace

_DB = Path(__file__).resolve().parent.parent / "data" / "llm_cache.sqlite3"
_DB.parent.mkdir(parents=True, exist_ok=True)


def _conn():
    c = sqlite3.connect(_DB)
    c.execute("CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, content TEXT)")
    return c

def _key(model: str, prompt: str) -> str:
    return hashlib.sha256(f"{model}::{prompt}".encode("utf-8")).hexdigest()


def cached_invoke(llm, prompt: str, model: str):
    """Drop-in replacement for llm.invoke(prompt) with sqlite caching."""
    k = _key(model, prompt)
    conn = _conn()
    row = conn.execute("SELECT content FROM cache WHERE key=?", (k,)).fetchone()
    if row:
        conn.close()
        return SimpleNamespace(content=row[0])

    response = llm.invoke(prompt)
    content = response.content if hasattr(response, "content") else str(response)
    conn.execute("INSERT OR REPLACE INTO cache (key, content) VALUES (?, ?)", (k, content))
    conn.commit()
    conn.close()
    return SimpleNamespace(content=content)