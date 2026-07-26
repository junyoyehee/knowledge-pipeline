"""백그라운드 잡 워커 (docs/api_design.md §3.2).

- GPU 워커: 동시성 1(설정 가능). train/export/infer 를 직렬 실행.
- CPU 워커: 동시성 N. prepare/generate 를 병렬 실행.

단일 프로세스 내 스레드로 구동한다. 실제 작업은 서브프로세스라 대기 중 GIL을
점유하지 않는다. 스케일아웃이 필요하면 Celery/RQ로 교체(docs/api_design.md §11).
"""
import threading
import time

from . import config, runner, store

_stop = threading.Event()
_threads: list[threading.Thread] = []


def _loop(gpu: bool) -> None:
    while not _stop.is_set():
        job = store.claim_next_job(gpu=gpu)
        if job is None:
            time.sleep(0.5)
            continue
        try:
            runner.run_job(job)
        except Exception as e:  # noqa: BLE001  (워커는 죽지 않아야 함)
            store.update_job(job["id"], status="failed",
                             error=f"워커 오류: {e}", finished_at=store.now())


def start() -> None:
    """워커 스레드 기동 (FastAPI startup에서 호출)."""
    if _threads:
        return
    for _ in range(max(1, config.GPU_CONCURRENCY)):
        t = threading.Thread(target=_loop, args=(True,), daemon=True,
                             name="gpu-worker")
        t.start(); _threads.append(t)
    for _ in range(max(1, config.CPU_CONCURRENCY)):
        t = threading.Thread(target=_loop, args=(False,), daemon=True,
                             name="cpu-worker")
        t.start(); _threads.append(t)


def stop() -> None:
    _stop.set()
