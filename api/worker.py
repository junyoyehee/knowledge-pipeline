"""백그라운드 잡 워커 (docs/api_design.md §3.2, remote_unsloth.md §3①).

- GPU 워커: 동시성 1(설정). train/export/infer 직렬 실행.
- CPU 워커: 동시성 N. prepare/generate 병렬.

두 가지 배치 모드:
1) 단일 호스트(기본): API 프로세스가 lifespan에서 GPU+CPU 워커 스레드를 모두 띄운다.
2) 원격 GPU 워커: API 호스트는 `KP_GPU_WORKER_ENABLED=false`로 CPU 워커만,
   GPU 호스트는 `python -m api.worker --role gpu`로 GPU 워커만 돌린다. 둘은
   공유 스토리지의 같은 DB(kp.db)를 큐로 공유하며, claim은 멀티프로세스 안전하다.

스케일아웃(다중 GPU 노드·Redis 큐)은 docs/api_design.md §11 Phase 3.
"""
import threading
import time

from . import config, runner, store

_stop = threading.Event()
_threads: list[threading.Thread] = []


def _loop(gpu: bool) -> None:
    while not _stop.is_set():
        job = store.claim_next_job(gpu=gpu, worker_id=config.WORKER_ID)
        if job is None:
            time.sleep(0.5)
            continue
        try:
            runner.run_job(job)
        except Exception as e:  # noqa: BLE001  (워커는 죽지 않아야 함)
            store.update_job(job["id"], status="failed",
                             error=f"워커 오류: {e}", finished_at=store.now())


def start(gpu_enabled: bool | None = None, cpu_enabled: bool | None = None) -> None:
    """워커 스레드 기동. None이면 config의 역할 게이팅을 따른다."""
    if _threads:
        return
    gpu_enabled = config.GPU_WORKER_ENABLED if gpu_enabled is None else gpu_enabled
    cpu_enabled = config.CPU_WORKER_ENABLED if cpu_enabled is None else cpu_enabled
    _stop.clear()
    if gpu_enabled:
        for _ in range(max(1, config.GPU_CONCURRENCY)):
            t = threading.Thread(target=_loop, args=(True,), daemon=True,
                                 name="gpu-worker")
            t.start(); _threads.append(t)
    if cpu_enabled:
        for _ in range(max(1, config.CPU_CONCURRENCY)):
            t = threading.Thread(target=_loop, args=(False,), daemon=True,
                                 name="cpu-worker")
            t.start(); _threads.append(t)
    roles = [r for r, on in (("gpu", gpu_enabled), ("cpu", cpu_enabled)) if on]
    print(f"[worker] 시작: {', '.join(roles) or '(없음)'} "
          f"(worker_id={config.WORKER_ID})")


def stop() -> None:
    _stop.set()


def main() -> None:
    """독립 워커 프로세스 진입점: python -m api.worker --role gpu|cpu|both"""
    import argparse
    import signal

    ap = argparse.ArgumentParser(description="지식 파이프라인 잡 워커")
    ap.add_argument("--role", choices=["gpu", "cpu", "both"], default="gpu",
                    help="이 워커가 처리할 잡 클래스 (기본 gpu)")
    args = ap.parse_args()

    config.ensure_storage()
    store.init_db()
    start(gpu_enabled=args.role in ("gpu", "both"),
          cpu_enabled=args.role in ("cpu", "both"))
    print(f"[worker] storage={config.STORAGE_ROOT} db={config.DB_PATH} "
          "대기 중... (Ctrl-C 종료)")

    done = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: done.set())
    done.wait()
    stop()
    print("[worker] 종료")


if __name__ == "__main__":
    main()
