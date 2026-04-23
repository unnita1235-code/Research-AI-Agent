"""Optional SQLite persistence for job metadata (in-memory is still the hot path in ``main``)."""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

_lock = threading.Lock()
_conn: sqlite3.Connection | None = None
_DB = Path(os.environ.get("JOB_DB_PATH", "")).resolve() if os.environ.get("JOB_DB_PATH") else None
if not _DB:
    _DB = Path(__file__).resolve().parent / "data" / "jobs.sqlite"


def _get_conn() -> sqlite3.Connection:
    global _conn
    with _lock:
        if _conn is not None:
            return _conn
        _DB.parent.mkdir(parents=True, exist_ok=True)
        c = sqlite3.connect(str(_DB), check_same_thread=False)
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY,
                topic TEXT NOT NULL,
                status TEXT NOT NULL,
                report TEXT,
                error TEXT,
                created REAL NOT NULL,
                events_json TEXT NOT NULL,
                options_json TEXT,
                latest_json TEXT,
                facts_json TEXT
            )
            """
        )
        for col in ("facts_json",):
            try:
                c.execute(f"ALTER TABLE jobs ADD COLUMN {col} TEXT")
            except sqlite3.OperationalError:
                pass
        c.commit()
        _conn = c
        return c


def enabled() -> bool:
    v = (os.environ.get("ENABLE_JOB_DB") or "1").strip().lower()
    if v in ("0", "false", "no"):
        return False
    return True


def upsert_row(row: dict[str, Any]) -> None:
    if not enabled():
        return
    c = _get_conn()
    ejson = json.dumps(row.get("events") or [], ensure_ascii=False)
    ojson = json.dumps(row.get("options") or {}, ensure_ascii=False) if row.get("options") is not None else None
    ljson = json.dumps(row.get("latest"), ensure_ascii=False) if row.get("latest") is not None else "null"
    fjson = json.dumps(row.get("extracted_facts") or [], ensure_ascii=False)
    with _lock:
        c.execute(
            """
            INSERT INTO jobs (id, topic, status, report, error, created, events_json, options_json, latest_json, facts_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                topic=excluded.topic,
                status=excluded.status,
                report=excluded.report,
                error=excluded.error,
                events_json=excluded.events_json,
                options_json=excluded.options_json,
                latest_json=excluded.latest_json,
                facts_json=excluded.facts_json
            """,
            (
                row["id"],
                row.get("topic", ""),
                row.get("status", "pending"),
                row.get("report"),
                row.get("error"),
                float(row.get("created") or time.time()),
                ejson,
                ojson,
                ljson,
                fjson,
            ),
        )
        c.commit()


def fetch_job(job_id: str) -> dict[str, Any] | None:
    c = _get_conn()
    with _lock:
        cur = c.execute("SELECT * FROM jobs WHERE id = ?", (job_id,))
        r = cur.fetchone()
        cols = [d[0] for d in cur.description]  # type: ignore[union-attr]
    if not r:
        return None
    d = dict(zip(cols, r, strict=True))
    return _row_to_job(d)


def _row_to_job(d: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "id": d["id"],
        "topic": d.get("topic") or "",
        "status": d.get("status") or "pending",
        "report": d.get("report"),
        "error": d.get("error"),
        "events": json.loads(d.get("events_json") or "[]"),
        "created": d.get("created") or 0.0,
    }
    oj = d.get("options_json")
    if oj:
        try:
            out["options"] = json.loads(oj)
        except json.JSONDecodeError:
            out["options"] = {}
    fj = d.get("facts_json")
    if fj:
        try:
            out["extracted_facts"] = json.loads(fj)
        except json.JSONDecodeError:
            out["extracted_facts"] = []
    if d.get("latest_json"):
        try:
            out["latest"] = json.loads(d["latest_json"])
        except json.JSONDecodeError:
            out["latest"] = None
    return out


def list_jobs(limit: int = 20) -> list[dict[str, Any]]:
    if not enabled():
        return []
    c = _get_conn()
    with _lock:
        cur = c.execute(
            "SELECT * FROM jobs ORDER BY created DESC LIMIT ?",
            (max(1, min(limit, 200)),),
        )
        rows = cur.fetchall()
        cols = [d[0] for d in cur.description]  # type: ignore[union-attr]
    out: list[dict[str, Any]] = []
    for r in rows:
        d = dict(zip(cols, r, strict=True))
        o = _row_to_job(d)
        out.append(
            {
                "id": o["id"],
                "topic": o.get("topic"),
                "status": o.get("status"),
                "created": o.get("created"),
            }
        )
    return out
