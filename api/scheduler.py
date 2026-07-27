"""관리형 스케줄러 실행 — SLURM / Kubernetes Job (docs/remote_unsloth.md §3 ③).

기존 로컬(§3 없음)·SSH(②)·원격 워커(①)와 달리, 여기서는 API가 GPU 잡을
**클러스터에 제출(submit)하고 상태를 폴링(poll)**만 한다. 실제 실행은 클러스터
노드가 담당한다.

전제 (docs/remote_unsloth.md §5, #8① 과 동일):
  - 클러스터 노드와 API가 **공유 스토리지를 동일 절대경로**로 본다
    (생성된 config·데이터셋·outputs·report.json·progress.json이 그대로 오간다).
  - 노드에 이 리포와 unsloth가 설치되어 있다(SLURM: SCHEDULER_REMOTE_DIR,
    K8s: K8S_IMAGE).

실행 모델:
  submit() → 외부 잡 ID → poll()로 상태 확인 → 종료 시 로그·리포트는 공유
  스토리지에서 회수. 로그는 잡이 log_path(공유)에 직접 쓰므로 러너가 그대로 tail.

백엔드는 CLI(sbatch/squeue/scancel, kubectl)를 통해 동작하며, 상태 매핑 등 순수
로직은 CLI 없이도 단위 테스트할 수 있도록 함수로 분리했다.
"""
import shlex
import subprocess

from . import config

# 스케줄러 관점의 정규화된 상태
PENDING, RUNNING, SUCCEEDED, FAILED = "pending", "running", "succeeded", "failed"
_TERMINAL = {SUCCEEDED, FAILED}


def enabled_for(jtype: str) -> bool:
    return config.SCHEDULER_ENABLED and jtype in config.GPU_JOB_TYPES


def get_backend():
    """설정된 백엔드 인스턴스를 반환."""
    name = (config.SCHEDULER_BACKEND or "").lower()
    if name == "slurm":
        return SlurmBackend()
    if name in ("k8s", "kubernetes"):
        return K8sBackend()
    raise RuntimeError(f"알 수 없는 스케줄러 백엔드: {config.SCHEDULER_BACKEND} "
                       "(slurm | k8s)")


def _inner_command(module: str, extra: list[str], cfg_path) -> str:
    """클러스터 노드에서 실행할 학습 커맨드 문자열(공용)."""
    parts = [shlex.quote(config.SCHEDULER_PYTHON), "-m", module,
             *[shlex.quote(a) for a in extra],
             "--config", shlex.quote(str(cfg_path))]
    cmd = " ".join(parts)
    if config.SCHEDULER_REMOTE_DIR:
        cmd = f"cd {shlex.quote(config.SCHEDULER_REMOTE_DIR)} && {cmd}"
    return cmd


def _run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


# ---------------------------------------------------------------------------
# SLURM
# ---------------------------------------------------------------------------

def map_slurm_state(state: str) -> str:
    """squeue/sacct 상태 문자열 → 정규화 상태.

    - PENDING/CONFIGURING/... → pending
    - RUNNING/COMPLETING     → running
    - COMPLETED              → succeeded
    - FAILED/CANCELLED/TIMEOUT/OUT_OF_MEMORY/... → failed
    - 알 수 없음/빈 값        → pending (아직 등록 전일 수 있음)
    """
    s = (state or "").strip().upper().split()[0] if state else ""
    # sacct는 "CANCELLED by 1000" 처럼 뒤에 부가어가 붙기도 함 → 첫 토큰만.
    if s in ("PENDING", "CONFIGURING", "REQUEUED", "RESIZING", "SUSPENDED"):
        return PENDING
    if s in ("RUNNING", "COMPLETING", "STAGE_OUT", "SIGNALING"):
        return RUNNING
    if s in ("COMPLETED",):
        return SUCCEEDED
    if s in ("FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL",
             "BOOT_FAIL", "DEADLINE", "PREEMPTED", "REVOKED"):
        return FAILED
    return PENDING


def build_sbatch_script(job: dict, inner_cmd: str, log_path) -> str:
    """제출용 sbatch 스크립트 본문 생성(순수 — 테스트 용이)."""
    name = f"kp-{job['id']}"
    lines = ["#!/bin/bash",
             f"#SBATCH --job-name={name}",
             f"#SBATCH --output={log_path}",
             f"#SBATCH --error={log_path}"]
    if config.SLURM_PARTITION:
        lines.append(f"#SBATCH --partition={config.SLURM_PARTITION}")
    if config.SLURM_GRES:
        lines.append(f"#SBATCH --gres={config.SLURM_GRES}")
    if config.SLURM_TIME_LIMIT:
        lines.append(f"#SBATCH --time={config.SLURM_TIME_LIMIT}")
    for opt in shlex.split(config.SLURM_EXTRA_SBATCH or ""):
        lines.append(f"#SBATCH {opt}")
    lines += ["set -euo pipefail", inner_cmd, ""]
    return "\n".join(lines)


class SlurmBackend:
    name = "slurm"

    def submit(self, job, module, extra, cfg_path, env, log_path) -> str:
        script = build_sbatch_script(job, _inner_command(module, extra, cfg_path),
                                     log_path)
        # 잡의 환경변수(예: generate 시크릿)는 --export=ALL 로 상속 + 명시 주입.
        exports = ",".join(f"{k}={v}" for k, v in (env or {}).items())
        cmd = ["sbatch", "--parsable"]
        if exports:
            cmd += [f"--export=ALL,{exports}"]
        r = _run(cmd, input=script)
        if r.returncode != 0:
            raise RuntimeError(f"sbatch 실패: {r.stderr.strip() or r.stdout.strip()}")
        # --parsable 출력: "<jobid>[;cluster]"
        return r.stdout.strip().split(";")[0]

    def poll(self, extid: str) -> str:
        # 실행 중이면 squeue에 보인다.
        q = _run(["squeue", "-h", "-j", extid, "-o", "%T"])
        if q.returncode == 0 and q.stdout.strip():
            return map_slurm_state(q.stdout.strip().splitlines()[0])
        # 큐에 없으면 종료됨 → sacct로 최종 상태 확인.
        a = _run(["sacct", "-n", "-X", "-j", extid, "-o", "State"])
        if a.returncode == 0 and a.stdout.strip():
            return map_slurm_state(a.stdout.strip().splitlines()[0])
        # 아직 sacct 반영 전일 수 있음 → pending 유지(러너가 계속 폴링).
        return PENDING

    def cancel(self, extid: str) -> None:
        _run(["scancel", extid])

    def stream_logs(self, extid: str, log_path) -> None:
        # SLURM은 --output 으로 log_path(공유 스토리지)에 직접 기록 → 러너가 tail.
        return None


# ---------------------------------------------------------------------------
# Kubernetes Job
# ---------------------------------------------------------------------------

def map_k8s_status(status: dict) -> str:
    """`kubectl get job -o json` 의 .status → 정규화 상태.

    succeeded>=1 → succeeded, failed>=1 → failed, active>=1 → running,
    그 외(스케줄 전) → pending.
    """
    if not isinstance(status, dict):
        return PENDING
    if (status.get("succeeded") or 0) >= 1:
        return SUCCEEDED
    if (status.get("failed") or 0) >= 1:
        return FAILED
    if (status.get("active") or 0) >= 1:
        return RUNNING
    return PENDING


def build_k8s_manifest(job: dict, inner_cmd: str) -> dict:
    """Kubernetes Job 매니페스트(dict) 생성(순수 — 테스트 용이)."""
    name = f"kp-{job['id']}"
    container = {
        "name": "trainer",
        "image": config.K8S_IMAGE,
        "command": ["/bin/bash", "-c", inner_cmd],
        "resources": {"limits": {config.K8S_GPU_RESOURCE: config.K8S_GPU_COUNT}},
    }
    spec_template = {"spec": {"containers": [container],
                              "restartPolicy": "Never"}}
    # 공유 스토리지 PVC를 STORAGE_ROOT 절대경로에 마운트(공유 경로 규약 유지).
    if config.K8S_STORAGE_PVC:
        container["volumeMounts"] = [{"name": "storage",
                                      "mountPath": str(config.STORAGE_ROOT)}]
        spec_template["spec"]["volumes"] = [
            {"name": "storage",
             "persistentVolumeClaim": {"claimName": config.K8S_STORAGE_PVC}}]
    if config.SCHEDULER_REMOTE_DIR:
        container["workingDir"] = config.SCHEDULER_REMOTE_DIR
    return {
        "apiVersion": "batch/v1", "kind": "Job",
        "metadata": {"name": name, "namespace": config.K8S_NAMESPACE,
                     "labels": {"app": "knowledge-pipeline", "kp-job": job["id"]}},
        "spec": {"backoffLimit": 0, "template": spec_template},
    }


class K8sBackend:
    name = "k8s"

    def _job_name(self, extid: str) -> str:
        return extid  # extid == 매니페스트 metadata.name (kp-<jid>)

    def submit(self, job, module, extra, cfg_path, env, log_path) -> str:
        import json
        if not config.K8S_IMAGE:
            raise RuntimeError("KP_K8S_IMAGE 미설정(학습 이미지 필요)")
        # env를 컨테이너 env로 주입.
        manifest = build_k8s_manifest(
            job, _inner_command(module, extra, cfg_path))
        if env:
            manifest["spec"]["template"]["spec"]["containers"][0]["env"] = [
                {"name": k, "value": str(v)} for k, v in env.items()]
        r = _run(["kubectl", "apply", "-n", config.K8S_NAMESPACE, "-f", "-"],
                 input=json.dumps(manifest))
        if r.returncode != 0:
            raise RuntimeError(f"kubectl apply 실패: "
                               f"{r.stderr.strip() or r.stdout.strip()}")
        return manifest["metadata"]["name"]

    def poll(self, extid: str) -> str:
        import json
        r = _run(["kubectl", "get", "job", extid, "-n", config.K8S_NAMESPACE,
                  "-o", "json"])
        if r.returncode != 0:
            return PENDING
        try:
            return map_k8s_status(json.loads(r.stdout).get("status", {}))
        except (ValueError, AttributeError):
            return PENDING

    def cancel(self, extid: str) -> None:
        _run(["kubectl", "delete", "job", extid, "-n", config.K8S_NAMESPACE,
              "--ignore-not-found"])

    def stream_logs(self, extid: str, log_path) -> None:
        """파드 로그를 공유 로그 파일로 스냅샷(공유 볼륨이 없을 때 대비)."""
        r = _run(["kubectl", "logs", f"job/{extid}", "-n", config.K8S_NAMESPACE])
        if r.returncode == 0 and r.stdout:
            try:
                with open(log_path, "w", encoding="utf-8") as fh:
                    fh.write(r.stdout)
            except OSError:
                pass
