"""SQLite 기반 메타 저장소 (프로젝트 · 잡).

단일 프로세스 + 워커 스레드 구성이라 스레드 안전을 위해 하나의 커넥션을
Lock으로 보호한다. 대용량 아티팩트는 파일시스템(storage.py)에 두고 여기엔
메타만 저장한다.
"""
import json
import sqlite3
import threading
import time
import uuid
from datetime import datetime, timezone

from .config import CPU_JOB_TYPES, DB_PATH, GPU_JOB_TYPES, ensure_storage

_lock = threading.RLock()
_conn: sqlite3.Connection | None = None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:20]}"


def init_db() -> None:
    global _conn
    ensure_storage()
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    _conn = sqlite3.connect(DB_PATH, check_same_thread=False, timeout=30)
    _conn.row_factory = sqlite3.Row
    with _lock:
        # WAL + busy_timeout: 여러 프로세스(API 호스트 + 원격 GPU 워커)가 공유
        # 스토리지의 같은 DB를 큐로 쓸 수 있게 한다 (docs/remote_unsloth.md §3①).
        _conn.execute("PRAGMA journal_mode=WAL")
        _conn.execute("PRAGMA busy_timeout=30000")
        _conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS projects (
                id TEXT PRIMARY KEY, name TEXT, base_model TEXT,
                chat_template TEXT, created_at TEXT
            );
            CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY, project_id TEXT, type TEXT, stage TEXT,
                status TEXT, params TEXT, progress TEXT, result TEXT, error TEXT,
                callback_url TEXT, idempotency_key TEXT, worker TEXT,
                created_at TEXT, started_at TEXT, finished_at TEXT
            );
            """
        )
        # 기존 DB 호환: worker 컬럼이 없으면 추가
        cols = {r["name"] for r in _conn.execute("PRAGMA table_info(jobs)")}
        if "worker" not in cols:
            _conn.execute("ALTER TABLE jobs ADD COLUMN worker TEXT")
        _conn.commit()


# ---------- 프로젝트 ----------

def create_project(name: str, base_model: str, chat_template: str) -> dict:
    pid = _new_id("proj")
    with _lock:
        _conn.execute(
            "INSERT INTO projects VALUES (?,?,?,?,?)",
            (pid, name, base_model, chat_template, _now()))
        _conn.commit()
    return get_project(pid)


def get_project(pid: str) -> dict | None:
    with _lock:
        row = _conn.execute("SELECT * FROM projects WHERE id=?", (pid,)).fetchone()
    return dict(row) if row else None


def list_projects() -> list[dict]:
    with _lock:
        rows = _conn.execute(
            "SELECT * FROM projects ORDER BY created_at DESC").fetchall()
    return [dict(r) for r in rows]


def delete_project(pid: str) -> None:
    with _lock:
        _conn.execute("DELETE FROM projects WHERE id=?", (pid,))
        _conn.execute("DELETE FROM jobs WHERE project_id=?", (pid,))
        _conn.commit()


# ---------- 잡 ----------

def _job_row_to_dict(row: sqlite3.Row) -> dict:
    d = dict(row)
    for k in ("params", "progress", "result"):
        d[k] = json.loads(d[k]) if d[k] else None
    d["job_id"] = d["id"]          # 설계 문서와 일치하는 별칭
    d["logs_url"] = f"/v1/jobs/{d['id']}/logs"
    return d


def create_job(project_id: str, jtype: str, params: dict,
               stage: str | None = None, callback_url: str | None = None,
               idempotency_key: str | None = None) -> dict:
    # 멱등성: 같은 키의 잡이 있으면 재사용
    if idempotency_key:
        with _lock:
            row = _conn.execute(
                "SELECT * FROM jobs WHERE project_id=? AND idempotency_key=?",
                (project_id, idempotency_key)).fetchone()
        if row:
            return _job_row_to_dict(row)

    jid = _new_id("job")
    with _lock:
        _conn.execute(
            "INSERT INTO jobs (id,project_id,type,stage,status,params,progress,"
            "result,error,callback_url,idempotency_key,created_at,started_at,"
            "finished_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (jid, project_id, jtype, stage, "queued", json.dumps(params),
             None, None, None, callback_url, idempotency_key, _now(), None, None))
        _conn.commit()
    return get_job(jid)


def get_job(jid: str) -> dict | None:
    with _lock:
        row = _conn.execute("SELECT * FROM jobs WHERE id=?", (jid,)).fetchone()
    return _job_row_to_dict(row) if row else None


def list_jobs(project_id: str | None = None, status: str | None = None,
              jtype: str | None = None, limit: int = 50) -> list[dict]:
    q = "SELECT * FROM jobs WHERE 1=1"
    args: list = []
    if project_id:
        q += " AND project_id=?"; args.append(project_id)
    if status:
        q += " AND status=?"; args.append(status)
    if jtype:
        q += " AND type=?"; args.append(jtype)
    q += " ORDER BY created_at DESC LIMIT ?"; args.append(limit)
    with _lock:
        rows = _conn.execute(q, args).fetchall()
    return [_job_row_to_dict(r) for r in rows]


def update_job(jid: str, **fields) -> None:
    if not fields:
        return
    sets, args = [], []
    for k, v in fields.items():
        if k in ("params", "progress", "result") and v is not None:
            v = json.dumps(v)
        sets.append(f"{k}=?"); args.append(v)
    args.append(jid)
    with _lock:
        _conn.execute(f"UPDATE jobs SET {', '.join(sets)} WHERE id=?", args)
        _conn.commit()


def claim_next_job(gpu: bool, worker_id: str | None = None) -> dict | None:
    """워커가 실행할 다음 queued 잡을 원자적으로 running으로 전환.

    gpu=True면 GPU 잡, False면 CPU 잡만 대상으로 한다. 멀티프로세스(원격 GPU
    워커 포함)에서도 안전하도록 `UPDATE ... WHERE status='queued'`의 rowcount로
    실제 클레임 여부를 판정한다 — 경쟁에서 진 프로세스는 rowcount 0 → 다음 후보로.
    """
    types = tuple(GPU_JOB_TYPES if gpu else CPU_JOB_TYPES)
    placeholders = ",".join("?" * len(types))
    with _lock:
        rows = _conn.execute(
            f"SELECT id FROM jobs WHERE status='queued' AND type IN "
            f"({placeholders}) ORDER BY created_at LIMIT 5", types).fetchall()
        for r in rows:
            cur = _conn.execute(
                "UPDATE jobs SET status='running', started_at=?, worker=? "
                "WHERE id=? AND status='queued'", (_now(), worker_id, r["id"]))
            _conn.commit()
            if cur.rowcount == 1:      # 이 프로세스가 실제로 잡았을 때만
                claimed = _conn.execute("SELECT * FROM jobs WHERE id=?",
                                        (r["id"],)).fetchone()
                return _job_row_to_dict(claimed)
        return None


def now() -> str:
    return _now()
