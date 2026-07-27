"""잡 → 서브프로세스 실행 (docs/api_design.md §3.1).

요청 파라미터로 잡별 config.yaml을 생성하고 기존 스크립트를 실행한다.
stdout을 로그 파일에 스트리밍하고 `loss:` 등을 파싱해 메트릭을 만든다.
"""
import os
import re
import signal
import subprocess
import threading
import time

from . import (config, config_builder, remote, scheduler, secrets, storage,
               store)

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
        # 구조화 메트릭 리포트 경로 (stdout 파싱 대신 이 파일에서 결과 수집)
        report_path = jdir / "report.json"
        # 실시간 진행률 파일 경로 (학습 스크립트의 TrainerCallback이 주기적으로 기록)
        progress_path = jdir / "progress.json"
        extra = [*extra, "--report-json", str(report_path),
                 "--progress-json", str(progress_path)]
        remote_mode = remote.enabled_for(job["type"])
        sched_mode = scheduler.enabled_for(job["type"])

        # 3) 환경변수 (generate 잡은 LLM 시크릿 주입; 원격 GPU 잡은 불필요)
        env = os.environ.copy()
        if job["type"] == "generate":
            sec = secrets.load_secrets(pid)
            llm = (job["params"] or {}).get("llm") or {}
            env["QA_GEN_BASE_URL"] = llm.get("base_url", sec.get("QA_GEN_BASE_URL", ""))
            env["QA_GEN_MODEL"] = llm.get("model", sec.get("QA_GEN_MODEL", ""))
            env["QA_GEN_API_KEY"] = llm.get("api_key", sec.get("QA_GEN_API_KEY", "dummy"))

        if sched_mode:
            # 클러스터 제출-폴링 경로(§3③). 서브프로세스 스트리밍을 쓰지 않는다.
            cmd = cwd = None
        elif remote_mode:
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

    # 스케줄러 모드: 제출→폴링 경로로 위임하고 종료
    if sched_mode:
        _run_via_scheduler(job, module, extra, cfg_path, log_path,
                           report_path, progress_path)
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
                # 학습 콜백이 남긴 진행률 파일을 우선 반영 (step/loss/lr/epoch/pct).
                # 원격 SSH 실행은 파일이 원격에 있어 폴링 불가(최종 리포트만 회수).
                if not remote_mode:
                    fp = _read_progress(progress_path)
                    if fp:
                        progress.update(fp)
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
        if remote_mode:                       # 원격 산출물·리포트를 로컬로 회수
            try:
                remote.pull_outputs(pid)
                remote.pull_file(report_path)
            except Exception as e:  # noqa: BLE001
                store.update_job(jid, status="failed",
                                 error=f"원격 산출물 회수 실패: {e}",
                                 finished_at=store.now())
                _notify(store.get_job(jid))
                return
        report = _read_report(report_path)
        store.update_job(jid, status="succeeded",
                         result=_collect_result(job, last_loss, report),
                         finished_at=store.now())
    else:
        tail = _log_tail(log_path)
        store.update_job(jid, status="failed",
                         error=f"스크립트 종료코드 {rc}\n{tail}",
                         finished_at=store.now())
    _notify(store.get_job(jid))


def _run_via_scheduler(job, module, extra, cfg_path, log_path,
                       report_path, progress_path) -> None:
    """GPU 잡을 클러스터에 제출하고 상태를 폴링한다 (docs/remote_unsloth.md §3③).

    공유 스토리지 전제(#8① 동일): 잡이 쓰는 log/report/progress 파일을 API가
    같은 절대경로에서 읽는다. 러너 스레드에서 동기로 폴링한다.
    """
    jid, pid = job["id"], job["project_id"]
    try:
        backend = scheduler.get_backend()
        # GPU 잡(train/export/infer)만 스케줄되므로 generate 시크릿은 불필요.
        # 클러스터 노드는 자체 환경을 사용한다(호스트 env 유출 방지).
        extid = backend.submit(job, module, extra, cfg_path, {}, log_path)
    except Exception as e:  # noqa: BLE001
        store.update_job(jid, status="failed",
                         error=f"스케줄러 제출 실패: {e}", finished_at=store.now())
        _notify(store.get_job(jid))
        return

    sched_meta = {"backend": backend.name, "id": extid}
    store.update_job(jid, progress={"scheduler": sched_meta})

    state = scheduler.PENDING
    while True:
        cur = store.get_job(jid)
        if cur and cur["status"] == "canceled":   # 외부에서 취소 요청됨
            try:
                backend.cancel(extid)
            except Exception:  # noqa: BLE001
                pass
            _notify(cur)
            return
        try:
            state = backend.poll(extid)
            backend.stream_logs(extid, log_path)      # best-effort
        except Exception as e:  # noqa: BLE001  (일시적 CLI 오류는 계속 폴링)
            _append_log(log_path, f"[scheduler] 폴링 오류: {e}\n")
        prog = {"scheduler": sched_meta}
        fp = _read_progress(progress_path)            # 공유 스토리지의 진행률
        if fp:
            prog.update(fp)
        store.update_job(jid, progress=prog)
        if state in (scheduler.SUCCEEDED, scheduler.FAILED):
            break
        time.sleep(config.SCHEDULER_POLL_INTERVAL)

    if state == scheduler.SUCCEEDED:
        report = _read_report(report_path)
        store.update_job(jid, status="succeeded",
                         result=_collect_result(job, None, report),
                         finished_at=store.now())
    else:
        tail = _log_tail(log_path)
        store.update_job(jid, status="failed",
                         error=f"스케줄러 잡 실패 ({backend.name}:{extid})\n{tail}",
                         finished_at=store.now())
    _notify(store.get_job(jid))


def _append_log(path, text: str) -> None:
    try:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(text)
    except OSError:
        pass


def _read_report(path) -> dict | None:
    """스크립트가 남긴 구조화 리포트(JSON)를 읽는다. 없거나 손상 시 None."""
    try:
        import json
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _read_progress(path) -> dict | None:
    """학습 콜백이 남긴 진행률 파일을 읽는다. 없거나 기록 중(손상)이면 None.

    write_progress가 tmp→replace로 원자적으로 갈아끼우므로 부분 읽기는 없다.
    """
    try:
        import json
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _collect_result(job: dict, last_loss: float | None,
                    report: dict | None = None) -> dict:
    """결과 조립 — 스크립트 리포트(정확)를 우선하고, 없으면 stdout/경로로 폴백."""
    jtype, pid = job["type"], job["project_id"]

    if report:  # --report-json 으로 받은 구조화 메트릭을 그대로 신뢰
        result = {k: v for k, v in report.items() if k not in ("status", "task")}
        result.setdefault("source", "report-json")
        return result

    # 폴백: 리포트가 없을 때 stdout loss·경로 규약으로 추정
    result: dict = {"source": "stdout-fallback"}
    if last_loss is not None:
        result["metrics"] = {"train_loss": last_loss}
    if jtype == "train":
        result["adapter_dir"] = str(storage.outputs_dir(pid) / job["stage"] / "final")
    elif jtype == "export":
        result["merged_dir"] = str(storage.outputs_dir(pid) / "final_model")
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
    job = store.get_job(jid)
    if not job:
        return False
    # 스케줄러 제출 잡: 로컬 프로세스가 없으므로 외부 잡을 취소한다.
    # (폴링 루프도 canceled 상태를 감지해 재차 취소하지만, 즉시성을 위해 여기서도.)
    sched = (job.get("progress") or {}).get("scheduler")
    if job["status"] == "running" and sched:
        store.update_job(jid, status="canceled", finished_at=store.now())
        try:
            scheduler.get_backend().cancel(sched["id"])
        except Exception:  # noqa: BLE001
            pass
        return True
    # 아직 실행 전이면 상태만 변경
    if job["status"] == "queued":
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
