"""API 런타임 설정 (환경변수로 오버라이드)."""
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
TEMPLATE_CONFIG = REPO_ROOT / "configs" / "config.yaml"

# 모든 프로젝트 데이터/산출물이 저장되는 루트
STORAGE_ROOT = Path(os.environ.get("KP_STORAGE_ROOT", REPO_ROOT / "storage"))
DB_PATH = Path(os.environ.get("KP_DB_PATH", STORAGE_ROOT / "kp.db"))

# 인증: 쉼표로 구분된 API 키 목록 (기본값은 개발용)
API_KEYS = {k for k in os.environ.get("KP_API_KEYS", "dev-key").split(",") if k}

# 잡 워커 동시성 — GPU 잡은 반드시 직렬(1) 권장
GPU_CONCURRENCY = int(os.environ.get("KP_GPU_CONCURRENCY", "1"))
CPU_CONCURRENCY = int(os.environ.get("KP_CPU_CONCURRENCY", "2"))

# 서브프로세스로 스크립트를 실행할 파이썬 인터프리터
PYTHON = os.environ.get("KP_PYTHON", sys.executable)

# 업로드 제한
MAX_UPLOAD_BYTES = int(os.environ.get("KP_MAX_UPLOAD_BYTES", str(200 * 1024 * 1024)))

# 시크릿 암호화 키 (Fernet). 없으면 secrets.py가 스토리지에 자동 생성.
SECRET_KEY = os.environ.get("KP_SECRET_KEY")

# 잡 타입 → 워커 클래스
GPU_JOB_TYPES = {"train", "export", "infer"}
CPU_JOB_TYPES = {"prepare", "generate"}


def ensure_storage() -> None:
    STORAGE_ROOT.mkdir(parents=True, exist_ok=True)
