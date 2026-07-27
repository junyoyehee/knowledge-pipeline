"""관리형 스케줄러(#8③) 검증 — 실제 SLURM/K8s 클러스터 없이.

- 순수 상태 매핑(map_slurm_state / map_k8s_status)
- 제출 스크립트/매니페스트 생성(build_sbatch_script / build_k8s_manifest)
- 러너의 제출→폴링→성공 경로(가짜 백엔드 주입, 공유 스토리지의 report/progress 반영)
- 러너의 취소 경로(running 스케줄러 잡 → 외부 취소 호출 + canceled)

실행: PYTHONPATH=. python -m api.tests.test_scheduler
"""
import json
import os
import tempfile

_TMP = tempfile.mkdtemp(prefix="kp-sched-test-")
os.environ["KP_STORAGE_ROOT"] = _TMP
os.environ["KP_DB_PATH"] = os.path.join(_TMP, "kp.db")

from api import config, runner, scheduler, storage, store  # noqa: E402


def test_map_slurm_state():
    assert scheduler.map_slurm_state("PENDING") == scheduler.PENDING
    assert scheduler.map_slurm_state("RUNNING") == scheduler.RUNNING
    assert scheduler.map_slurm_state("COMPLETED") == scheduler.SUCCEEDED
    assert scheduler.map_slurm_state("FAILED") == scheduler.FAILED
    assert scheduler.map_slurm_state("TIMEOUT") == scheduler.FAILED
    # sacct의 "CANCELLED by 1000" 형태 → 첫 토큰만 본다
    assert scheduler.map_slurm_state("CANCELLED by 1000") == scheduler.FAILED
    assert scheduler.map_slurm_state("") == scheduler.PENDING
    assert scheduler.map_slurm_state("SOMETHING_NEW") == scheduler.PENDING
    print("[ok] SLURM 상태 매핑")


def test_map_k8s_status():
    assert scheduler.map_k8s_status({"succeeded": 1}) == scheduler.SUCCEEDED
    assert scheduler.map_k8s_status({"failed": 1}) == scheduler.FAILED
    assert scheduler.map_k8s_status({"active": 1}) == scheduler.RUNNING
    assert scheduler.map_k8s_status({}) == scheduler.PENDING
    # succeeded가 failed보다 우선(재시도 없이 완료된 경우)
    assert scheduler.map_k8s_status({"succeeded": 1, "failed": 1}) == \
        scheduler.SUCCEEDED
    assert scheduler.map_k8s_status(None) == scheduler.PENDING
    print("[ok] K8s 상태 매핑")


def test_build_sbatch_script():
    config.SLURM_PARTITION = "gpu"
    config.SLURM_GRES = "gpu:a100:1"
    config.SLURM_TIME_LIMIT = "12:00:00"
    config.SLURM_EXTRA_SBATCH = "--mem=64G"
    config.SCHEDULER_REMOTE_DIR = "/opt/knowledge-pipeline"
    job = {"id": "job_x"}
    inner = scheduler._inner_command("scripts.train.train_sft",
                                     ["--report-json", "/s/r.json"], "/s/c.yaml")
    script = scheduler.build_sbatch_script(job, inner, "/s/job.log")
    assert script.startswith("#!/bin/bash")
    assert "#SBATCH --job-name=kp-job_x" in script
    assert "#SBATCH --output=/s/job.log" in script
    assert "#SBATCH --partition=gpu" in script
    assert "#SBATCH --gres=gpu:a100:1" in script
    assert "#SBATCH --time=12:00:00" in script
    assert "#SBATCH --mem=64G" in script
    assert "cd /opt/knowledge-pipeline &&" in script
    assert "-m scripts.train.train_sft" in script
    assert "--config /s/c.yaml" in script
    print("[ok] sbatch 스크립트 생성")


def test_build_k8s_manifest():
    config.K8S_IMAGE = "registry/kp:latest"
    config.K8S_NAMESPACE = "ml"
    config.K8S_GPU_RESOURCE = "nvidia.com/gpu"
    config.K8S_GPU_COUNT = "2"
    config.K8S_STORAGE_PVC = "kp-shared"
    config.SCHEDULER_REMOTE_DIR = "/opt/knowledge-pipeline"
    job = {"id": "job_y"}
    inner = scheduler._inner_command("scripts.model.export_model", [], "/s/c.yaml")
    m = scheduler.build_k8s_manifest(job, inner)
    assert m["kind"] == "Job"
    assert m["metadata"]["name"] == "kp-job_y"
    assert m["metadata"]["namespace"] == "ml"
    c = m["spec"]["template"]["spec"]["containers"][0]
    assert c["image"] == "registry/kp:latest"
    assert c["resources"]["limits"]["nvidia.com/gpu"] == "2"
    assert c["command"][0] == "/bin/bash" and c["command"][1] == "-c"
    assert c["workingDir"] == "/opt/knowledge-pipeline"
    # PVC를 STORAGE_ROOT 절대경로에 마운트
    vm = c["volumeMounts"][0]
    assert vm["mountPath"] == str(config.STORAGE_ROOT)
    vol = m["spec"]["template"]["spec"]["volumes"][0]
    assert vol["persistentVolumeClaim"]["claimName"] == "kp-shared"
    assert m["spec"]["backoffLimit"] == 0
    print("[ok] K8s 매니페스트 생성")


class _FakeBackend:
    """스크립트된 상태 시퀀스를 반환하는 가짜 백엔드(클러스터 대체)."""
    name = "fake"

    def __init__(self, states, report_path=None, progress_path=None):
        self._states = list(states)
        self._report_path = report_path
        self._progress_path = progress_path
        self.submitted = False
        self.canceled = False

    def submit(self, job, module, extra, cfg_path, env, log_path):
        self.submitted = True
        # 노드가 실행되는 것처럼 진행률 파일을 미리 남겨 둔다(공유 스토리지 모사)
        if self._progress_path:
            with open(self._progress_path, "w", encoding="utf-8") as f:
                json.dump({"step": 3, "total_steps": 6, "pct": 50.0}, f)
        return "fake-123"

    def poll(self, extid):
        st = self._states.pop(0)
        # 성공 직전에 report.json을 남긴다(노드가 완료 시 기록)
        if st == scheduler.SUCCEEDED and self._report_path:
            with open(self._report_path, "w", encoding="utf-8") as f:
                json.dump({"task": "train", "stage": "sft", "status": "ok",
                           "n_samples": 10,
                           "metrics": {"train_loss": 0.42}}, f)
        return st

    def cancel(self, extid):
        self.canceled = True

    def stream_logs(self, extid, log_path):
        return None


def _make_gpu_job():
    config.ensure_storage(); store.init_db()
    p = store.create_project("sched", "m", "qwen-2.5")
    job = store.create_job(p["id"], "train", {}, stage="sft")
    jdir = storage.job_dir(p["id"], job["id"]); jdir.mkdir(parents=True, exist_ok=True)
    return job, jdir


def test_scheduler_submit_poll_success():
    config.SCHEDULER_POLL_INTERVAL = 0        # 테스트 가속
    job, jdir = _make_gpu_job()
    report_path = jdir / "report.json"
    progress_path = jdir / "progress.json"
    fake = _FakeBackend(
        [scheduler.PENDING, scheduler.RUNNING, scheduler.SUCCEEDED],
        report_path=report_path, progress_path=progress_path)
    orig = scheduler.get_backend
    scheduler.get_backend = lambda: fake
    try:
        runner._run_via_scheduler(
            job, "scripts.train.train_sft",
            ["--report-json", str(report_path)],
            jdir / "config.yaml", jdir / "job.log", report_path, progress_path)
    finally:
        scheduler.get_backend = orig

    cur = store.get_job(job["id"])
    assert fake.submitted, "제출되지 않음"
    assert cur["status"] == "succeeded", cur
    # 리포트 기반 결과 수집
    assert cur["result"]["metrics"]["train_loss"] == 0.42, cur["result"]
    # 폴링 중 공유 진행률 파일 + 스케줄러 메타가 반영됨
    assert cur["progress"]["scheduler"]["id"] == "fake-123", cur["progress"]
    assert cur["progress"].get("pct") == 50.0, cur["progress"]
    print("[ok] 제출→폴링→성공 + 리포트/진행률 반영")


def test_scheduler_cancel_running():
    config.SCHEDULER_POLL_INTERVAL = 0
    job, jdir = _make_gpu_job()
    fake = _FakeBackend([scheduler.RUNNING])
    orig = scheduler.get_backend
    scheduler.get_backend = lambda: fake
    try:
        # 실행 중 + 스케줄러 메타가 있는 상태를 만든다
        store.update_job(job["id"], status="running",
                         progress={"scheduler": {"backend": "fake", "id": "fake-9"}})
        ok = runner.cancel(job["id"])
        assert ok, "취소 실패"
        assert fake.canceled, "외부 스케줄러 취소가 호출되지 않음"
        assert store.get_job(job["id"])["status"] == "canceled"
        print("[ok] running 스케줄러 잡 취소 → 외부 취소 호출")
    finally:
        scheduler.get_backend = orig


def test_scheduler_submit_failure():
    """제출 자체가 실패하면 잡이 failed로 마킹된다."""
    job, jdir = _make_gpu_job()

    class _Boom:
        name = "boom"
        def submit(self, *a, **k):
            raise RuntimeError("sbatch not found")
    orig = scheduler.get_backend
    scheduler.get_backend = lambda: _Boom()
    try:
        runner._run_via_scheduler(
            job, "scripts.train.train_sft", [], jdir / "config.yaml",
            jdir / "job.log", jdir / "report.json", jdir / "progress.json")
    finally:
        scheduler.get_backend = orig
    cur = store.get_job(job["id"])
    assert cur["status"] == "failed" and "제출 실패" in cur["error"], cur
    print("[ok] 제출 실패 → failed")


if __name__ == "__main__":
    test_map_slurm_state()
    test_map_k8s_status()
    test_build_sbatch_script()
    test_build_k8s_manifest()
    test_scheduler_submit_poll_success()
    test_scheduler_cancel_running()
    test_scheduler_submit_failure()
    print("SCHEDULER TESTS PASS")
