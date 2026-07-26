"""원격 SSH 러너 검증 (실제 원격 호스트 불필요).

- remote.build_cmd / enabled_for / rsync 커맨드 구성(순수 함수) 검증
- runner의 원격 분기(push → build_cmd → 실행 → pull)를 remote 함수 몽키패치로 검증
  (GPU/unsloth 없이, build_cmd를 무해한 로컬 커맨드로 대체해 실행 경로만 확인)

실행: PYTHONPATH=. python -m api.tests.test_remote
"""
import os
import sys
import tempfile

os.environ.setdefault("KP_API_KEYS", "test-key")
_TMP = tempfile.mkdtemp(prefix="kp-remote-test-")
os.environ["KP_STORAGE_ROOT"] = _TMP
os.environ["KP_DB_PATH"] = os.path.join(_TMP, "kp.db")
# 원격 설정(빌드 검증용)
os.environ["KP_REMOTE_ENABLED"] = "1"
os.environ["KP_REMOTE_HOST"] = "user@gpu-host"
os.environ["KP_REMOTE_DIR"] = "/opt/knowledge-pipeline"
os.environ["KP_REMOTE_SSH_OPTS"] = "-p 2222"

from api import config, remote, runner, storage, store  # noqa: E402


def test_pure_helpers():
    assert remote.enabled_for("train") and remote.enabled_for("export")
    assert not remote.enabled_for("prepare") and not remote.enabled_for("generate")

    cmd = remote.build_cmd("scripts.train.train_sft", [], "/data/kp/p1/jobs/j1/config.yaml")
    assert cmd[0] == "ssh" and "user@gpu-host" in cmd
    assert "-p" in cmd and "2222" in cmd            # SSH_OPTS 반영
    joined = cmd[-1]
    assert "cd /opt/knowledge-pipeline" in joined
    assert "-m scripts.train.train_sft" in joined
    assert "--config /data/kp/p1/jobs/j1/config.yaml" in joined

    rs = remote._rsync_cmd("src/", "user@gpu-host:dst/")
    assert rs[0] == "rsync" and rs[-2:] == ["src/", "user@gpu-host:dst/"]
    assert "-e" in rs                                # SSH_OPTS → rsync -e
    print("[ok] remote 순수 헬퍼(enabled_for/build_cmd/rsync) 검증")


def test_runner_remote_branch(monkeypatch):
    config.ensure_storage(); store.init_db()
    p = store.create_project("t", "base-model", "qwen-2.5")
    storage.init_project_dirs(p["id"])
    job = store.create_job(p["id"], "export", {"overrides": {}}, stage="sft")

    calls = {"push": 0, "pull": 0, "pull_file": 0}
    monkeypatch.setattr(remote, "enabled_for", lambda t: True)
    monkeypatch.setattr(remote, "push_project", lambda pid: calls.__setitem__("push", calls["push"] + 1))
    monkeypatch.setattr(remote, "pull_outputs", lambda pid: calls.__setitem__("pull", calls["pull"] + 1))
    monkeypatch.setattr(remote, "pull_file", lambda p: calls.__setitem__("pull_file", calls["pull_file"] + 1))
    # build_cmd를 무해한 로컬 커맨드로 대체(원격 실행을 시뮬레이션)
    monkeypatch.setattr(remote, "build_cmd",
                        lambda module, extra, cfg: [sys.executable, "-c",
                                                    "print('[remote-sim] step 3/3 loss: 0.42')"])

    runner.run_job(store.get_job(job["id"]))

    done = store.get_job(job["id"])
    assert done["status"] == "succeeded", done
    assert calls["push"] == 1 and calls["pull"] == 1, calls          # 푸시·산출물 회수
    assert calls["pull_file"] == 1, calls                            # 리포트 회수 시도
    assert done["result"]["metrics"]["train_loss"] == 0.42           # stdout loss 폴백
    log = (storage.job_dir(p["id"], job["id"]) / "job.log").read_text(encoding="utf-8")
    assert "[remote]" in log and "remote-sim" in log
    print("[ok] runner 원격 분기(push→build_cmd→실행→pull) 검증")


def _run():
    test_pure_helpers()
    # 간이 monkeypatch (pytest 없이 실행 가능하도록)
    class MP:
        def __init__(self): self._u = []
        def setattr(self, obj, name, val):
            self._u.append((obj, name, getattr(obj, name)))
            setattr(obj, name, val)
        def undo(self):
            for obj, name, val in reversed(self._u): setattr(obj, name, val)
    mp = MP()
    try:
        test_runner_remote_branch(mp)
    finally:
        mp.undo()
    print("REMOTE TESTS PASS")


if __name__ == "__main__":
    _run()
