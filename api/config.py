"""API 런타임 설정 (환경변수로 오버라이드)."""
import os
import socket
import sys
from pathlib import Path


def _env_bool(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.lower() in ("1", "true", "yes", "on")

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

# ---------- 워커 역할 게이팅 (docs/remote_unsloth.md §3 ①) ----------
# 이 프로세스가 어떤 워커를 돌릴지. 원격 GPU 워커 구성에서는 API 호스트가
# KP_GPU_WORKER_ENABLED=false로 GPU 잡을 원격 워커에 넘기고, GPU 호스트는
# `python -m api.worker --role gpu`로 GPU 워커만 돌린다(공유 스토리지의 큐 공유).
GPU_WORKER_ENABLED = _env_bool("KP_GPU_WORKER_ENABLED", True)
CPU_WORKER_ENABLED = _env_bool("KP_CPU_WORKER_ENABLED", True)
# 잡을 클레임한 워커 식별자(추적용)
WORKER_ID = os.environ.get("KP_WORKER_ID", socket.gethostname())

# 서브프로세스로 스크립트를 실행할 파이썬 인터프리터
PYTHON = os.environ.get("KP_PYTHON", sys.executable)

# 업로드 제한
MAX_UPLOAD_BYTES = int(os.environ.get("KP_MAX_UPLOAD_BYTES", str(200 * 1024 * 1024)))

# 시크릿 암호화 키 (Fernet). 없으면 secrets.py가 스토리지에 자동 생성.
SECRET_KEY = os.environ.get("KP_SECRET_KEY")

# 잡 타입 → 워커 클래스
GPU_JOB_TYPES = {"train", "export", "infer"}
CPU_JOB_TYPES = {"prepare", "generate"}

# ---------- 원격 unsloth(GPU) 실행 (docs/remote_unsloth.md §3 ②) ----------
# 활성화 시 GPU 잡(train/export/infer)을 원격 GPU 호스트에서 SSH로 실행한다.
# CPU 잡(prepare/generate)은 항상 로컬에서 실행된다.
REMOTE_ENABLED = os.environ.get("KP_REMOTE_ENABLED", "").lower() in ("1", "true", "yes")
REMOTE_HOST = os.environ.get("KP_REMOTE_HOST")            # 예: user@gpu-host
REMOTE_DIR = os.environ.get("KP_REMOTE_DIR")              # 원격 리포 체크아웃 경로
REMOTE_PYTHON = os.environ.get("KP_REMOTE_PYTHON", "python3")
REMOTE_SSH_OPTS = os.environ.get("KP_REMOTE_SSH_OPTS", "")  # 예: "-p 2222 -i ~/key"
# 원격 스토리지 루트는 로컬과 동일 절대경로로 미러링한다(생성된 config가 절대경로를
# 담으므로 원격에서도 같은 경로여야 해석된다). 다른 경로를 쓰려면 이 값을 조정.
REMOTE_STORAGE_ROOT = os.environ.get("KP_REMOTE_STORAGE_ROOT", str(STORAGE_ROOT))

# ---------- 관리형 스케줄러 (docs/remote_unsloth.md §3 ③) ----------
# 활성화 시 GPU 잡(train/export/infer)을 SLURM/Kubernetes 등 클러스터에 제출하고
# API는 제출·상태 폴링·로그 회수만 담당한다. CPU 잡은 항상 로컬 실행.
# 전제: 클러스터 노드와 API가 공유 스토리지를 동일 절대경로로 본다(#8① 동일).
SCHEDULER_ENABLED = _env_bool("KP_SCHEDULER_ENABLED", False)
SCHEDULER_BACKEND = os.environ.get("KP_SCHEDULER_BACKEND", "slurm")  # slurm | k8s
SCHEDULER_POLL_INTERVAL = float(os.environ.get("KP_SCHEDULER_POLL_INTERVAL", "10"))
# SLURM
SLURM_PARTITION = os.environ.get("KP_SLURM_PARTITION")        # 예: gpu
SLURM_GRES = os.environ.get("KP_SLURM_GRES", "gpu:1")          # 예: gpu:a100:1
SLURM_EXTRA_SBATCH = os.environ.get("KP_SLURM_EXTRA_SBATCH", "")  # 임의 #SBATCH 옵션
SLURM_TIME_LIMIT = os.environ.get("KP_SLURM_TIME_LIMIT")      # 예: 24:00:00
# Kubernetes
K8S_NAMESPACE = os.environ.get("KP_K8S_NAMESPACE", "default")
K8S_IMAGE = os.environ.get("KP_K8S_IMAGE")                    # unsloth 포함 학습 이미지
K8S_GPU_RESOURCE = os.environ.get("KP_K8S_GPU_RESOURCE", "nvidia.com/gpu")
K8S_GPU_COUNT = os.environ.get("KP_K8S_GPU_COUNT", "1")
# 공유 스토리지를 파드에 마운트하는 PVC (STORAGE_ROOT 절대경로에 마운트).
K8S_STORAGE_PVC = os.environ.get("KP_K8S_STORAGE_PVC")
# 클러스터 노드가 리포를 체크아웃해 둔 경로(작업 디렉터리). 없으면 이미지 WORKDIR.
SCHEDULER_REMOTE_DIR = os.environ.get("KP_SCHEDULER_REMOTE_DIR")
SCHEDULER_PYTHON = os.environ.get("KP_SCHEDULER_PYTHON", "python3")


def ensure_storage() -> None:
    STORAGE_ROOT.mkdir(parents=True, exist_ok=True)
