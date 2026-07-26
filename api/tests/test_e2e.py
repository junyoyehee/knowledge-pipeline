"""API 엔드투엔드 스모크 테스트 (GPU 불필요).

prepare 잡은 prepare_data.py를 서브프로세스로 실제 실행하므로, 업로드→준비→
데이터셋 조회까지 전체 경로(라우팅·인증·config 생성·잡 큐·워커·서브프로세스)를
GPU 없이 검증한다. train/export/infer는 unsloth/GPU가 필요해 여기서는 제외.

실행: pytest api/tests/test_e2e.py  (또는 python api/tests/test_e2e.py)
"""
import os
import tempfile
import time

os.environ.setdefault("KP_API_KEYS", "test-key")
# 격리된 임시 스토리지
_TMP = tempfile.mkdtemp(prefix="kp-api-test-")
os.environ["KP_STORAGE_ROOT"] = _TMP
os.environ["KP_DB_PATH"] = os.path.join(_TMP, "kp.db")

from fastapi.testclient import TestClient  # noqa: E402

from api import config as _cfg  # noqa: E402
from api import store as _store  # noqa: E402
from api import worker as _worker  # noqa: E402
from api.main import app  # noqa: E402

H = {"Authorization": "Bearer test-key"}


def _client() -> TestClient:
    # lifespan 컨텍스트매니저 대신 수동 초기화(테스트 종료 시 shutdown portal 이슈 회피)
    _cfg.ensure_storage()
    _store.init_db()
    _worker.start()
    return TestClient(app)


def _wait_job(client, jid, timeout=120):
    for _ in range(timeout * 2):
        r = client.get(f"/v1/jobs/{jid}", headers=H)
        st = r.json()["status"]
        if st in ("succeeded", "failed", "canceled"):
            return r.json()
        time.sleep(0.5)
    raise AssertionError("잡 타임아웃")


def test_end_to_end():
    client = _client()
    if True:
        # 인증 없으면 401
        assert client.get("/v1/projects").status_code == 401

        # 헬스
        assert client.get("/v1/healthz").json()["status"] == "ok"

        # 프로젝트 생성
        r = client.post("/v1/projects", headers=H, json={"name": "t"})
        assert r.status_code == 201, r.text
        pid = r.json()["id"]

        # 시크릿 저장(값은 응답에 노출 안 됨)
        client.put(f"/v1/projects/{pid}/secrets", headers=H,
                   json={"values": {"QA_GEN_MODEL": "dummy"}})
        assert client.get(f"/v1/projects/{pid}", headers=H).json()["secrets"] \
            == ["QA_GEN_MODEL"]

        # 원문 업로드
        doc = b"# title\n\nfirst para.\n\nsecond para about corestone."
        r = client.post(f"/v1/projects/{pid}/raw-files", headers=H,
                        files={"file": ("world.md", doc, "text/markdown")},
                        data={"kind": "doc"})
        assert r.status_code == 201, r.text

        # QA 업로드 (jsonl → 라인수 카운트)
        qa = (b'{"messages":[{"role":"user","content":"q?"},'
              b'{"role":"assistant","content":"a."}]}\n')
        r = client.post(f"/v1/projects/{pid}/raw-files", headers=H,
                        files={"file": ("qa.jsonl", qa, "application/x-ndjson")},
                        data={"kind": "qa"})
        assert r.json()["lines"] == 1, r.text
        assert r.json()["name"] == "qa_qa.jsonl"

        # prepare 잡 (override로 chunk_size 조정 — 허용 필드)
        r = client.post(f"/v1/projects/{pid}/datasets:prepare", headers=H,
                        json={"overrides": {"data": {"chunk_size": 500},
                                            # 경로류 override는 거부되어야 함
                                            "cpt": {"output_dir": "/etc"}}})
        assert r.status_code == 202, r.text
        jid = r.json()["job_id"]

        job = _wait_job(client, jid)
        assert job["status"] == "succeeded", job
        # 거부된 경로 override 기록 확인
        assert any("output_dir" in p for p in
                   (job.get("progress") or {}).get("rejected_overrides", []))
        # 결과에 데이터셋 요약
        kinds = {d["kind"] for d in job["result"]["datasets"]}
        assert {"cpt", "sft"} <= kinds, job["result"]

        # 데이터셋 조회
        r = client.get(f"/v1/projects/{pid}/datasets", headers=H)
        assert any(d["kind"] == "sft" for d in r.json()["datasets"])
        r = client.get(f"/v1/projects/{pid}/datasets/sft?sample=1", headers=H)
        assert r.json()["sample"][0]["messages"][0]["role"] == "user"

        # 로그 조회(비스트림)
        r = client.get(f"/v1/jobs/{jid}/logs", headers=H)
        assert "CPT 데이터셋" in r.text or "cpt" in r.text.lower()

        # 어댑터 없음(학습 안 함) → infer는 409
        r = client.post(f"/v1/projects/{pid}/infer", headers=H,
                        json={"stage": "sft", "questions": ["hi"]})
        assert r.status_code == 409, r.text

        # train 선행조건: planact는 sft 어댑터 필요 → 409
        r = client.post(f"/v1/projects/{pid}/train", headers=H,
                        json={"stage": "planact"})
        assert r.status_code == 409, r.text

        print("E2E PASS: project→upload→prepare(job)→datasets→guards")


if __name__ == "__main__":
    test_end_to_end()
