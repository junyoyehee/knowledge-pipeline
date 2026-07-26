"""API 키 인증 (docs/api_design.md §9)."""
from fastapi import Header, HTTPException

from .config import API_KEYS


def require_api_key(authorization: str = Header(default="")) -> None:
    """Authorization: Bearer <key> 검증."""
    token = authorization.removeprefix("Bearer ").strip()
    if token not in API_KEYS:
        raise HTTPException(status_code=401, detail={
            "error": {"code": "unauthenticated", "message": "유효한 API 키가 필요합니다."}})
