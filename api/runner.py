"""잡 → 서브프로세스 실행 (docs/api_design.md §3.1).

요청 파라미터로 잡별 config.yaml을 생성하고 기존 스크립트를 실행한다.
stdout을 로그 파일에 스트리밍하고 `loss:` 등을 파싱해 메트릭을 만든다.
"""
import os
import re
import signal
import subprocess
import threading

from . import config, config_builder, remote, secrets, storage, store

# 실행 중인 잡의 Popen 핸들 (취소용)
_procs: dict[str, subprocess.Popen] = {}
_procs_lock = threading.Lock()

_LOSS_RE = re.compile(r"loss[:=]\s*([0-9]+\.[0-9]+)", re.IGNORECASE)
_STEP_RE = re.compile(r"(\d+)\s*/\s*(\d+)")


def _script_for(job: dict) -> list[str]:
    """잡 타입/스테이지 → 실행할 모듈과 추가 인자.

    반환 [0]은 `python -m` 으로 실행할 모듈 경로, [1:]은 추가 CLI 인자.
    (config는 호출부에서 --config로 덧붙인다.)
    """
    jtype = job["type"]
    params = job.get("params") or {}

    if jtype == "prepare":
        return ["scripts.data.prepare_data"]
    if jtype == "generate":
        kind = params.get("kind")
        module = {
            "qa": "generate_qa", "tool": "generate_tool_calls",
            "plan": "generate_plans", "react": "generate_react",
            "planact": "generate_planact", "preference": "generate_preference",
        }.get(kind)
        if not module:
            raise ValueError(f"알 수 없는 generate kind: {kind}")
        args = [f"scripts.generate.{module}"]
        for flag in ("per_chunk", "max_chunks", "per_sample", "refusals_per_chunk"):
            if params.get(flag) is not None:
                args += [f"--{flag.replace('_', '-')}", str(params[flag])]
        if params.get("overwrite"):
            args.append("--overwrite")
        if params.get("tools_catalog_file"):
            args += ["--tools", str(storage.raw_dir(job["project_id"])
                                    / params["tools_catalog_file"])]
        if params.get("mode"):          # preference 전용
            args += ["--mode", str(params["mode"])]
        return args
    if jtype == "train":
        stage = job["stage"]
        if stage not in ("cpt", "sft", "tool", "plan", "react", "planact",
                         "dpo", "orpo", "kto"):
            raise ValueError(f"알 수 없는 train stage: {stage}")
        return [f"scripts.train.train_{stage}"]
    if jtype == "export":
        args = ["scripts.model.export_model"]
        if job.get("stage"):
            args += ["--stage", job["stage"]]
        return args
    if jtype == "infer":
        args = ["scripts.model.test_model", "--stage", job.get("stage") or "sft"]
        for q in (params.get("questions") or []):
            args += ["-q", q]
        if params.get("max_new_tokens"):
            args += ["--max-new-tokens", str(params["max_new_tokens"])]
        return args
    raise ValueError(f"알 수 없는 잡 타입: {jtype}")


def run_job(job: dict) -> None:
    """잡을 동기로 실행하고 상태/메트릭/결과를 store에 반영한다 (워커 스레드에서 호출)."""
    jid, pid = job["id"], job["project_id"]
    project = store.get_project(pid)
    jdir = storage.job_dir(pid, jid)
    jdir.mkdir(parents=True, exist_ok=True)
    log_path = jdir / "job.log"

    try:
        # 1) config 생성
        cfg, rejected = config_builder.build_config(
            pid, project, (job["params"] or {}).get("overrides"))
        cfg_path = jdir / "config.yaml"
        config_builder.write_config(cfg, cfg_path)
        if rejected:
            store.update_job(jid, progress={"rejected_overrides": rejected})

        # 2) 로컬/원격 실행 결정 (GPU 잡 + 원격 설정 시 SSH 원격 실행)
        module, *extra = _script_for(job)
        remote_mode = remote.enabled_for(job["type"])

        # 3) 환경변수 (generate 잡은 LLM 시크릿 주입; 원격 GPU 잡은 불필요)
        env = os.environ.copy()
        if job["type"] == "generate":
            sec = secrets.load_secrets(pid)
            llm = (job["params"] or {}).get("llm") or {}
            env["QA_GEN_BASE_URL"] = llm.get("base_url", sec.get("QA_GEN_BASE_URL", ""))
            env["QA_GEN_MODEL"] = llm.get("model", sec.get("QA_GEN_MODEL", ""))
            env["QA_GEN_API_KEY"] = llm.get("api_key", sec.get("QA_GEN_API_KEY", "dummy"))

        if remote_mode:
            # 입력 데이터·config·기존 어댑터를 원격으로 푸시 후 SSH 실행 커맨드 구성
            remote.push_project(pid)
            cmd = remote.build_cmd(module, extra, cfg_path)
            cwd = None
        else:
            cmd = [config.PYTHON, "-m", module, *extra, "--config", str(cfg_path)]
            cwd = str(config.REPO_ROOT)

    except Exception as e:  # noqa: BLE001  (구성/원격 준비 단계 실패)
        store.update_job(jid, status="failed", error=f"구성 실패: {e}",
                         finished_at=store.now())
        _notify(job)
        return

    # 4) 서브프로세스 실행 + 로그 스트리밍 + 메트릭 파싱
    last_loss = None
    progress: dict = {}
    try:
        with open(log_path, "w", encoding="utf-8") as logf:
            if remote_mode:
                logf.write(f"[remote] {config.REMOTE_HOST} 에서 실행: "
                           f"{' '.join(module.split())}\n")
                logf.flush()
            proc = subprocess.Popen(
                cmd, cwd=cwd, env=env,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                bufsize=1)
            with _procs_lock:
                _procs[jid] = proc
            for line in proc.stdout:
                logf.write(line)
                logf.flush()
                m = _LOSS_RE.search(line)
                if m:
                    last_loss = float(m.group(1))
                    progress["loss"] = last_loss
                sm = _STEP_RE.search(line)
                if sm:
                    progress["step"] = int(sm.group(1))
                    progress["total"] = int(sm.group(2))
                if progress:
                    store.update_job(jid, progress=progress)
            rc = proc.wait()
    finally:
        with _procs_lock:
            _procs.pop(jid, None)

    # 5) 결과 판정
    cur = store.get_job(jid)
    if cur and cur["status"] == "canceled":
        _notify(cur)
        return
    if rc == 0:
        if remote_mode:                       # 원격 산출물(어댑터/모델)을 로컬로 회수
            try:
                remote.pull_outputs(pid)
            except Exception as e:  # noqa: BLE001
                store.update_job(jid, status="failed",
                                 error=f"원격 산출물 회수 실패: {e}",
                                 finished_at=store.now())
                _notify(store.get_job(jid))
                return
        store.update_job(jid, status="succeeded",
                         result=_collect_result(job, last_loss),
                         finished_at=store.now())
    else:
        tail = _log_tail(log_path)
        store.update_job(jid, status="failed",
                         error=f"스크립트 종료코드 {rc}\n{tail}",
                         finished_at=store.now())
    _notify(store.get_job(jid))


def _collect_result(job: dict, last_loss: float | None) -> dict:
    jtype, pid = job["type"], job["project_id"]
    result: dict = {}
    if last_loss is not None:
        result["metrics"] = {"train_loss": last_loss}
    if jtype == "train":
        result["adapter"] = {"stage": job["stage"],
                             "path": str(storage.outputs_dir(pid) / job["stage"] / "final")}
    elif jtype == "export":
        result["model"] = {"merged_dir": str(storage.outputs_dir(pid) / "final_model")}
    elif jtype == "prepare":
        result["datasets"] = _dataset_summary(pid)
    return result


def _dataset_summary(pid: str) -> list[dict]:
    out = []
    proc = storage.processed_dir(pid)
    for f in sorted(proc.glob("*.jsonl")):
        with open(f, "r", encoding="utf-8") as fh:
            n = sum(1 for line in fh if line.strip())
        out.append({"kind": f.stem.replace("_dataset", ""), "count": n,
                    "file": f.name})
    return out


def _log_tail(path, n: int = 20) -> str:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
        return "\n".join(lines[-n:])
    except OSError:
        return ""


def cancel(jid: str) -> bool:
    """실행 중인 잡의 서브프로세스를 종료. queued면 상태만 취소로 전환."""
    with _procs_lock:
        proc = _procs.get(jid)
    if proc and proc.poll() is None:
        store.update_job(jid, status="canceled", finished_at=store.now())
        try:
            proc.send_signal(signal.SIGTERM)
        except ProcessLookupError:
            pass
        return True
    # 아직 실행 전이면 상태만 변경
    job = store.get_job(jid)
    if job and job["status"] == "queued":
        store.update_job(jid, status="canceled", finished_at=store.now())
        return True
    return False


def _notify(job: dict | None) -> None:
    """callback_url이 있으면 완료 웹훅 전송 (best-effort)."""
    if not job or not job.get("callback_url"):
        return
    try:
        import json
        import urllib.request
        body = json.dumps({"job_id": job["id"], "status": job["status"],
                           "result": job.get("result")}).encode("utf-8")
        req = urllib.request.Request(job["callback_url"], data=body,
                                     headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=10)
    except Exception:  # noqa: BLE001  (웹훅 실패는 무시)
        pass
