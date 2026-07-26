"""원격 unsloth(GPU) 실행 — SSH 러너 PoC (docs/remote_unsloth.md §3 ②).

GPU 잡을 원격 GPU 호스트에서 실행한다:
  1) 프로젝트 스토리지(raw/processed/config/기존 outputs)를 원격에 rsync 푸시
  2) SSH로 `python -m scripts.<group>.<name> --config <절대경로>` 실행 (stdout 스트리밍)
  3) 성공 시 원격 outputs/ 를 로컬로 rsync 회수

전제 (docs/remote_unsloth.md §5):
  - 원격 호스트에 이 리포가 REMOTE_DIR에 체크아웃되어 있고 unsloth가 설치됨
  - 원격 스토리지 경로가 로컬과 동일 절대경로(REMOTE_STORAGE_ROOT)로 접근 가능
  - 로컬에 ssh/rsync 설치, 원격 접속 자격증명(키) 구성

한계(PoC): 취소는 로컬 ssh 프로세스를 종료하며 원격 프로세스가 잠시 잔존할 수 있음.
스토리지는 rsync 전송(공유 마운트 아님) — 대용량/다중 노드는 §3 ①(공유 스토리지)로 승격.
"""
import shlex
import subprocess

from . import config, storage


def enabled_for(jtype: str) -> bool:
    return config.REMOTE_ENABLED and jtype in config.GPU_JOB_TYPES


def _require_settings() -> None:
    missing = [k for k, v in (("KP_REMOTE_HOST", config.REMOTE_HOST),
                              ("KP_REMOTE_DIR", config.REMOTE_DIR)) if not v]
    if missing:
        raise RuntimeError(f"원격 실행 설정 누락: {', '.join(missing)}")


def _ssh_base() -> list[str]:
    return ["ssh", *shlex.split(config.REMOTE_SSH_OPTS), config.REMOTE_HOST]


def _rsync_cmd(src: str, dst: str) -> list[str]:
    cmd = ["rsync", "-az"]
    if config.REMOTE_SSH_OPTS:
        cmd += ["-e", "ssh " + config.REMOTE_SSH_OPTS]
    cmd += [src, dst]
    return cmd


def _remote_project_dir(pid: str) -> str:
    # 로컬 절대경로를 그대로 미러링 (config의 절대경로가 원격에서도 해석되도록)
    return str(storage.project_dir(pid))


def push_project(pid: str) -> None:
    """프로젝트 스토리지를 원격으로 동기화(입력 데이터·config·기존 어댑터)."""
    _require_settings()
    rpath = _remote_project_dir(pid)
    subprocess.run(_ssh_base() + ["mkdir", "-p", rpath], check=True,
                   capture_output=True, text=True)
    subprocess.run(_rsync_cmd(str(storage.project_dir(pid)) + "/",
                              f"{config.REMOTE_HOST}:{rpath}/"),
                   check=True, capture_output=True, text=True)


def pull_outputs(pid: str) -> None:
    """원격 산출물(outputs/)을 로컬로 회수. 원격에 없으면 무시."""
    rout = str(storage.outputs_dir(pid))
    subprocess.run(_ssh_base() + ["mkdir", "-p", rout], check=False,
                   capture_output=True, text=True)
    storage.outputs_dir(pid).mkdir(parents=True, exist_ok=True)
    subprocess.run(_rsync_cmd(f"{config.REMOTE_HOST}:{rout}/",
                              str(storage.outputs_dir(pid)) + "/"),
                   check=False, capture_output=True, text=True)


def pull_file(abs_path) -> None:
    """원격의 단일 파일(동일 절대경로)을 로컬로 회수. 없으면 무시."""
    subprocess.run(_rsync_cmd(f"{config.REMOTE_HOST}:{abs_path}", str(abs_path)),
                   check=False, capture_output=True, text=True)


def build_cmd(module: str, extra: list[str], cfg_path) -> list[str]:
    """원격 실행 SSH 커맨드 구성. config 절대경로는 로컬=원격 미러 전제."""
    inner_parts = [f"cd {shlex.quote(config.REMOTE_DIR)} &&",
                   shlex.quote(config.REMOTE_PYTHON), "-m", module,
                   *[shlex.quote(a) for a in extra],
                   "--config", shlex.quote(str(cfg_path))]
    return _ssh_base() + [" ".join(inner_parts)]
