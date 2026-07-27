"""실시간 학습 진행률(#9) 검증 — GPU/transformers 없이.

- progress_from_state: (step, total, logs) → 진행률 dict (순수 함수)
- write_progress ↔ _read_progress: 원자적 기록·읽기 라운드트립
- make_progress_callback: path 없으면 None (transformers 지연 import라 import 안 됨)
- SSE follow 스트림에 progress 이벤트가 실린다 (store.progress 갱신 → 이벤트)

실행: PYTHONPATH=. python -m api.tests.test_progress
"""
import os
import tempfile

_TMP = tempfile.mkdtemp(prefix="kp-progress-test-")
os.environ["KP_STORAGE_ROOT"] = _TMP
os.environ["KP_DB_PATH"] = os.path.join(_TMP, "kp.db")

from scripts.lib.report import (  # noqa: E402
    make_progress_callback, progress_from_state, write_progress)
from api import runner  # noqa: E402


def test_progress_from_state():
    # 정상: step/total → pct 계산, logs에서 loss/lr/epoch 추출
    d = progress_from_state(5, 20, {"loss": 1.5, "learning_rate": 1e-4,
                                    "epoch": 0.25})
    assert d["step"] == 5 and d["total_steps"] == 20
    assert d["pct"] == 25.0, d
    assert d["loss"] == 1.5 and d["lr"] == 1e-4 and d["epoch"] == 0.25

    # eval_loss 매핑
    assert progress_from_state(1, 2, {"eval_loss": 0.9})["eval_loss"] == 0.9

    # None/누락 값은 제외 (pct는 step/total 둘 다 있어야 계산)
    d2 = progress_from_state(None, None, None)
    assert "step" not in d2 and "pct" not in d2, d2
    d3 = progress_from_state(3, 0, {})   # total=0 → pct 미계산, total_steps 제외
    assert "pct" not in d3 and d3.get("step") == 3
    print("[ok] progress_from_state 순수 계산")


def test_write_read_roundtrip():
    path = os.path.join(_TMP, "sub", "progress.json")
    write_progress(path, {"step": 7, "total_steps": 10, "pct": 70.0})
    # 러너의 폴링 헬퍼로 되읽기
    got = runner._read_progress(__import__("pathlib").Path(path))
    assert got == {"step": 7, "total_steps": 10, "pct": 70.0}, got
    # 없는 파일 → None
    assert runner._read_progress(
        __import__("pathlib").Path(os.path.join(_TMP, "nope.json"))) is None

    # write_progress(None) 무시
    write_progress(None, {"x": 1})   # 예외 없이 통과해야 함
    print("[ok] write/read 라운드트립 + None 무시")


def test_callback_none_without_path():
    assert make_progress_callback(None) is None
    assert make_progress_callback("") is None
    print("[ok] path 없으면 콜백 None (transformers 미의존)")


def test_sse_progress_event():
    """follow 스트림이 store.progress 변화를 progress 이벤트로 방출하는지."""
    import asyncio

    from api import config, storage, store
    from api.routers import jobs

    config.ensure_storage(); store.init_db()
    p = store.create_project("pg", "m", "qwen-2.5")
    job = store.create_job(p["id"], "train", {}, stage="sft")
    jid = job["id"]
    # 로그 파일 생성(스트림이 열 수 있도록)
    jdir = storage.job_dir(p["id"], jid); jdir.mkdir(parents=True, exist_ok=True)
    (jdir / "job.log").write_text("start\n", encoding="utf-8")

    async def drive():
        resp = await jobs.job_logs(jid, follow=True)
        agen = resp.body_iterator
        events = []
        # 1) 초기 로그 + 진행률 세팅 → progress 이벤트
        store.update_job(jid, progress={"step": 2, "total_steps": 4, "pct": 50.0})
        # 스트림에서 몇 개의 청크를 뽑되, 마지막엔 완료 상태로 종료시킨다
        async def pump(n):
            out = []
            for _ in range(n):
                out.append(await agen.__anext__())
            return out
        events += await pump(2)      # log + progress (또는 순서 섞임)
        store.update_job(jid, status="succeeded", finished_at=store.now())
        try:
            while True:
                events.append(await agen.__anext__())
        except StopAsyncIteration:
            pass
        return "".join(events)

    blob = asyncio.get_event_loop().run_until_complete(drive())
    assert "event: progress" in blob, blob
    assert '"pct": 50.0' in blob, blob
    assert "event: status" in blob and "succeeded" in blob, blob
    print("[ok] SSE progress 이벤트 방출")


if __name__ == "__main__":
    test_progress_from_state()
    test_write_read_roundtrip()
    test_callback_none_without_path()
    test_sse_progress_event()
    print("PROGRESS TESTS PASS")
