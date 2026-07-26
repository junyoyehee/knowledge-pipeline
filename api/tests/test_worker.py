"""원격 GPU 워커(#8①) 검증 — 멀티프로세스 안전 큐 + 역할 게이팅.

실제 GPU/Redis 없이:
- claim_next_job이 원자적(같은 잡을 두 번 클레임하지 않음)이고 클래스별로 분리됨
- CPU-only 워커 구성에서 GPU 잡은 큐에 남고(=원격 GPU 워커가 가져갈 몫), CPU 잡만 처리됨

실행: PYTHONPATH=. python -m api.tests.test_worker
"""
import os
import tempfile
import time

_TMP = tempfile.mkdtemp(prefix="kp-worker-test-")
os.environ["KP_STORAGE_ROOT"] = _TMP
os.environ["KP_DB_PATH"] = os.path.join(_TMP, "kp.db")

from api import config, runner, store, worker  # noqa: E402


def test_atomic_claim():
    config.ensure_storage(); store.init_db()
    p = store.create_project("t", "m", "qwen-2.5")
    job = store.create_job(p["id"], "export", {}, stage="sft")   # GPU 잡

    first = store.claim_next_job(gpu=True, worker_id="w1")
    second = store.claim_next_job(gpu=True, worker_id="w2")
    assert first and first["id"] == job["id"], first
    assert second is None, "같은 잡이 두 번 클레임됨(이중 클레임 버그)"
    assert store.get_job(job["id"])["worker"] == "w1"

    # 클래스 분리: CPU 워커는 GPU 잡을 가져가지 않음
    cpu_job = store.create_job(p["id"], "prepare", {})
    assert store.claim_next_job(gpu=True) is None          # GPU 워커엔 안 보임
    got = store.claim_next_job(gpu=False)                   # CPU 워커가 가져감
    assert got and got["id"] == cpu_job["id"]
    print("[ok] 원자적 클레임 + 클래스 분리")


def test_role_gating():
    """CPU-only 워커: GPU 잡은 큐에 남고 CPU 잡만 처리된다 (원격 GPU 워커 분리 모델)."""
    config.ensure_storage(); store.init_db()
    p = store.create_project("t2", "m", "qwen-2.5")

    # 서브프로세스 실행 대신 즉시 성공 처리로 대체
    def fake_run(job):
        store.update_job(job["id"], status="succeeded", finished_at=store.now())
    orig = runner.run_job
    runner.run_job = fake_run
    try:
        cpu_job = store.create_job(p["id"], "prepare", {})
        gpu_job = store.create_job(p["id"], "train", {}, stage="sft")
        worker.start(gpu_enabled=False, cpu_enabled=True)      # CPU 워커만
        # CPU 잡이 처리될 때까지 잠시 대기
        for _ in range(40):
            if store.get_job(cpu_job["id"])["status"] == "succeeded":
                break
            time.sleep(0.1)
        worker.stop(); time.sleep(0.3)
        assert store.get_job(cpu_job["id"])["status"] == "succeeded", "CPU 잡 미처리"
        assert store.get_job(gpu_job["id"])["status"] == "queued", \
            "GPU 잡이 CPU-only 호스트에서 처리됨(원격 워커 몫이어야 함)"
        print("[ok] 역할 게이팅: GPU 잡은 큐 대기, CPU 잡만 처리")
    finally:
        runner.run_job = orig


if __name__ == "__main__":
    test_atomic_claim()
    test_role_gating()
    print("WORKER TESTS PASS")
