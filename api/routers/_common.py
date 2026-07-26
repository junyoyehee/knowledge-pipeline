"""라우터 공용 헬퍼."""
from fastapi import HTTPException

from .. import store


def get_project_or_404(pid: str) -> dict:
    project = store.get_project(pid)
    if not project:
        raise HTTPException(status_code=404, detail={
            "error": {"code": "not_found", "message": f"프로젝트 없음: {pid}"}})
    return project


def err(status: int, code: str, message: str, **details):
    return HTTPException(status_code=status, detail={
        "error": {"code": code, "message": message, "details": details or None}})
