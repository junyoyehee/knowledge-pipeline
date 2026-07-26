"""프로젝트별 스토리지 레이아웃 (docs/api_design.md §3.3).

{STORAGE_ROOT}/{project_id}/
├── raw/          업로드 원문·jsonl
├── processed/    준비된 데이터셋
├── outputs/      어댑터·병합모델
├── jobs/{jid}/   잡별 config·로그
└── secrets.enc   암호화된 시크릿
"""
from pathlib import Path

from .config import STORAGE_ROOT

# raw 파일 kind → 저장 파일명 접두 규칙
KIND_PREFIX = {
    "doc": "",              # 원문은 확장자로 구분(.md/.txt), 접두어 없음
    "qa": "qa_",
    "tools": "tools_",
    "plans": "plans_",
    "react": "react_",
    "planact": "planact_",
}


def project_dir(pid: str) -> Path:
    return STORAGE_ROOT / pid


def raw_dir(pid: str) -> Path:
    return project_dir(pid) / "raw"


def processed_dir(pid: str) -> Path:
    return project_dir(pid) / "processed"


def outputs_dir(pid: str) -> Path:
    return project_dir(pid) / "outputs"


def jobs_dir(pid: str) -> Path:
    return project_dir(pid) / "jobs"


def job_dir(pid: str, jid: str) -> Path:
    return jobs_dir(pid) / jid


def secrets_path(pid: str) -> Path:
    return project_dir(pid) / "secrets.enc"


def init_project_dirs(pid: str) -> None:
    for d in (raw_dir(pid), processed_dir(pid), outputs_dir(pid), jobs_dir(pid)):
        d.mkdir(parents=True, exist_ok=True)


def raw_filename(kind: str, original: str) -> str:
    """kind와 원본 파일명으로 저장 파일명을 만든다 (경로 요소 제거)."""
    base = Path(original).name  # 경로 탈출 방지
    prefix = KIND_PREFIX.get(kind, "")
    if prefix and not base.startswith(prefix):
        # tools/qa/... 는 반드시 .jsonl
        stem = Path(base).stem
        return f"{prefix}{stem}.jsonl"
    return base
