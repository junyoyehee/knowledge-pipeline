"""잡 조회 · 로그(SSE) · 취소 라우터."""
import asyncio
import json

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from .. import runner, storage, store
from ..auth import require_api_key
from ._common import err, get_project_or_404

router = APIRouter(dependencies=[Depends(require_api_key)])


@router.get("/jobs/{jid}")
def get_job(jid: str):
    job = store.get_job(jid)
    if not job:
        raise err(404, "not_found", f"잡 없음: {jid}")
    return job


@router.get("/projects/{pid}/jobs")
def list_jobs(pid: str, status: str | None = None, type: str | None = None,
              limit: int = 50):
    get_project_or_404(pid)
    return {"jobs": store.list_jobs(project_id=pid, status=status, jtype=type,
                                    limit=limit)}


@router.post("/jobs/{jid}:cancel")
def cancel_job(jid: str):
    job = store.get_job(jid)
    if not job:
        raise err(404, "not_found", f"잡 없음: {jid}")
    ok = runner.cancel(jid)
    if not ok:
        raise err(409, "conflict", "이미 종료되었거나 취소할 수 없는 잡입니다.")
    return store.get_job(jid)


@router.get("/jobs/{jid}/logs")
async def job_logs(jid: str, follow: bool = False):
    job = store.get_job(jid)
    if not job:
        raise err(404, "not_found", f"잡 없음: {jid}")
    log_path = storage.job_dir(job["project_id"], jid) / "job.log"

    if not follow:
        text = log_path.read_text(encoding="utf-8") if log_path.exists() else ""
        return StreamingResponse(iter([text]), media_type="text/plain")

    async def stream():
        pos = 0
        last_progress = None
        while True:
            if log_path.exists():
                with open(log_path, "r", encoding="utf-8") as fh:
                    fh.seek(pos)
                    for line in fh:
                        yield f"event: log\ndata: {json.dumps({'line': line.rstrip()})}\n\n"
                    pos = fh.tell()
            cur = store.get_job(jid)
            # 진행률이 갱신되면 progress 이벤트로 흘려보낸다 (step/loss/pct 등).
            prog = (cur or {}).get("progress")
            if prog and prog != last_progress:
                last_progress = prog
                yield f"event: progress\ndata: {json.dumps(prog, ensure_ascii=False)}\n\n"
            if cur and cur["status"] in ("succeeded", "failed", "canceled"):
                yield f"event: status\ndata: {json.dumps({'status': cur['status']})}\n\n"
                return
            await asyncio.sleep(1.0)

    return StreamingResponse(stream(), media_type="text/event-stream")
