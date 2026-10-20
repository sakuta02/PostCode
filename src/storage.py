from __future__ import annotations

import json
import sqlite3
import statistics
import threading
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS estimates (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    shipment_id    TEXT    NOT NULL,
    load_pct       REAL    NOT NULL,
    interval_low   REAL    NOT NULL,
    interval_high  REAL    NOT NULL,
    confidence     TEXT    NOT NULL,
    occupied_ratio REAL,
    mode           TEXT    NOT NULL,
    model_version  TEXT    NOT NULL,
    n_models       INTEGER NOT NULL,
    image_sha256   TEXT,
    image_path     TEXT,
    image_format   TEXT,
    processing_ms  INTEGER NOT NULL,
    created_at     TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_estimates_shipment ON estimates (shipment_id, id);
CREATE INDEX IF NOT EXISTS ix_estimates_sha ON estimates (shipment_id, image_sha256);

CREATE TABLE IF NOT EXISTS feedback (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    shipment_id        TEXT    NOT NULL,
    estimate_id        INTEGER NOT NULL REFERENCES estimates(id),
    confirmed_load_pct REAL    NOT NULL,
    comment            TEXT,
    image_path         TEXT,
    used_in_version    TEXT,
    created_at         TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_feedback_shipment ON feedback (shipment_id, id);

CREATE TABLE IF NOT EXISTS models (
    version    TEXT PRIMARY KEY,
    path       TEXT NOT NULL,
    kind       TEXT NOT NULL,
    is_active  INTEGER NOT NULL DEFAULT 0,
    meta       TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS jobs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    kind        TEXT NOT NULL,
    status      TEXT NOT NULL,
    result      TEXT,
    log         TEXT,
    created_at  TEXT NOT NULL,
    finished_at TEXT
);
"""


def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class Storage:
    def __init__(self, path: Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._db = sqlite3.connect(str(path), check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA foreign_keys=ON")
        self._db.executescript(SCHEMA)
        self._db.commit()

    def _q(self, sql: str, args=()) -> list[dict]:
        with self._lock:
            return [dict(r) for r in self._db.execute(sql, args).fetchall()]

    def _x(self, sql: str, args=()) -> int:
        with self._lock:
            cur = self._db.execute(sql, args)
            self._db.commit()
            return cur.lastrowid

    def add_estimate(self, shipment_id: str, pred, info: dict, image_path: str | None,
                     processing_ms: int) -> dict:
        eid = self._x(
            """INSERT INTO estimates (shipment_id, load_pct, interval_low, interval_high,
               confidence, occupied_ratio, mode, model_version, n_models, image_sha256,
               image_path, image_format, processing_ms, created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (shipment_id, pred.load_pct, pred.low, pred.high, pred.confidence,
             pred.occupied_ratio, pred.mode, pred.model_version, pred.n_models,
             info.get("sha256"), image_path, info.get("format"), processing_ms, now()))
        return self.get_estimate(eid)

    def get_estimate(self, estimate_id: int) -> dict | None:
        rows = self._q("SELECT * FROM estimates WHERE id = ?", (estimate_id,))
        return rows[0] if rows else None

    def find_duplicate(self, shipment_id: str, sha256: str, mode: str,
                       model_version: str) -> dict | None:
        rows = self._q(
            """SELECT * FROM estimates WHERE shipment_id = ? AND image_sha256 = ?
               AND mode = ? AND model_version = ? ORDER BY id DESC LIMIT 1""",
            (shipment_id, sha256, mode, model_version))
        return rows[0] if rows else None

    def history(self, shipment_id: str) -> list[dict]:
        return self._q("SELECT * FROM estimates WHERE shipment_id = ? ORDER BY id",
                       (shipment_id,))

    def summary(self, shipment_id: str) -> dict:
        hist = self.history(shipment_id)
        latest: dict[str, dict] = {}
        for e in hist:
            latest[e["image_sha256"] or f"id:{e['id']}"] = e
        loads = [e["load_pct"] for e in latest.values()]
        return {"count": len(hist), "photos": len(latest), "method": "median",
                "aggregate": statistics.median(loads) if loads else 0.0}

    def shipment(self, shipment_id: str) -> dict | None:
        hist = self.history(shipment_id)
        if not hist:
            return None
        fb = self._q("SELECT * FROM feedback WHERE shipment_id = ? ORDER BY id DESC LIMIT 1",
                     (shipment_id,))
        return {"shipment_id": shipment_id, "history": hist,
                "summary": self.summary(shipment_id), "feedback": fb[0] if fb else None}

    def add_feedback(self, shipment_id: str, estimate: dict, confirmed: float,
                     comment: str | None) -> dict:
        fid = self._x(
            """INSERT INTO feedback (shipment_id, estimate_id, confirmed_load_pct, comment,
               image_path, created_at) VALUES (?,?,?,?,?,?)""",
            (shipment_id, estimate["id"], confirmed, comment, estimate.get("image_path"), now()))
        return self._q("SELECT * FROM feedback WHERE id = ?", (fid,))[0]

    _LATEST_PER_PHOTO = """
        SELECT f.* FROM feedback f
        JOIN (SELECT MAX(id) AS id FROM feedback
              WHERE image_path IS NOT NULL GROUP BY image_path) last ON last.id = f.id"""

    def confirmed_feedback(self) -> list[dict]:
        return self._q(self._LATEST_PER_PHOTO + " ORDER BY f.id")

    def pending_feedback(self) -> list[dict]:
        return self._q(self._LATEST_PER_PHOTO +
                       " WHERE f.used_in_version IS NULL ORDER BY f.id")

    def mark_feedback_used(self, ids: list[int], version: str) -> None:
        if ids:
            marks = ",".join("?" * len(ids))
            self._x(f"UPDATE feedback SET used_in_version = ? WHERE id IN ({marks})",
                    (version, *ids))

    def accuracy(self) -> dict:
        rows = self._q(
            """SELECT e.load_pct AS predicted, f.confirmed_load_pct AS confirmed,
                      e.model_version, e.mode
               FROM feedback f
               JOIN (SELECT MAX(id) AS id FROM feedback GROUP BY estimate_id) last
                    ON last.id = f.id
               JOIN estimates e ON e.id = f.estimate_id""")

        def stats(items: list[dict]) -> dict:
            err = [r["predicted"] - r["confirmed"] for r in items]
            if not err:
                return {"n": 0}
            return {"n": len(err),
                    "mae": sum(abs(x) for x in err) / len(err),
                    "within10": 100 * sum(abs(x) <= 10 for x in err) / len(err),
                    "bias": sum(err) / len(err)}

        groups: dict[tuple, list] = {}
        for r in rows:
            groups.setdefault((r["model_version"], r["mode"]), []).append(r)
        return {"overall": stats(rows),
                "by_model": [{"version": v, "mode": m, **stats(g)}
                             for (v, m), g in sorted(groups.items())]}

    def register_model(self, version: str, path: str, kind: str, meta: dict | None = None,
                       activate: bool = False) -> None:
        with self._lock:
            self._db.execute(
                """INSERT OR IGNORE INTO models (version, path, kind, meta, created_at)
                   VALUES (?,?,?,?,?)""",
                (version, path, kind, json.dumps(meta or {}, ensure_ascii=False), now()))
            self._db.commit()
        if activate:
            self.activate_model(version)

    def activate_model(self, version: str) -> dict | None:
        with self._lock:
            row = self._db.execute("SELECT * FROM models WHERE version = ?",
                                   (version,)).fetchone()
            if row is None:
                return None
            self._db.execute("UPDATE models SET is_active = (version = ?)", (version,))
            self._db.commit()
            return dict(row)

    def get_model(self, version: str) -> dict | None:
        rows = self._q("SELECT * FROM models WHERE version = ?", (version,))
        return rows[0] if rows else None

    def active_model(self) -> dict | None:
        rows = self._q("SELECT * FROM models WHERE is_active = 1 LIMIT 1")
        return rows[0] if rows else None

    def models(self) -> list[dict]:
        return self._q("SELECT * FROM models ORDER BY created_at, version")

    def create_job(self, kind: str) -> dict:
        jid = self._x("INSERT INTO jobs (kind, status, created_at) VALUES (?,?,?)",
                      (kind, "queued", now()))
        return self.job(jid)

    def update_job(self, job_id: int, status: str, result: dict | None = None,
                   log: str | None = None) -> None:
        finished = now() if status in ("done", "failed") else None
        self._x("""UPDATE jobs SET status = ?, result = COALESCE(?, result),
                   log = COALESCE(?, log), finished_at = COALESCE(?, finished_at)
                   WHERE id = ?""",
                (status, json.dumps(result, ensure_ascii=False) if result else None,
                 log, finished, job_id))

    def job(self, job_id: int) -> dict | None:
        rows = self._q("SELECT * FROM jobs WHERE id = ?", (job_id,))
        return rows[0] if rows else None

    def running_job(self, kind: str) -> dict | None:
        rows = self._q("SELECT * FROM jobs WHERE kind = ? AND status IN ('queued','running')",
                       (kind,))
        return rows[0] if rows else None
