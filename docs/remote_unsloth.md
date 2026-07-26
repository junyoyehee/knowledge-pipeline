# 원격 unsloth(GPU) 배포 설계

unsloth(+CUDA)는 **GPU가 있는 호스트에서만** 실행됩니다. 따라서 "unsloth가 원격에
있다"는 것은 **학습·병합·(간이)추론 실행은 원격 GPU 호스트에서, 오케스트레이션
(API·큐·스토리지)은 그와 분리**된다는 뜻입니다. 이 문서는 그 배포 방식을 정리합니다.

- 상태: **초안(Draft)** — 리뷰 후 확정
- 관련: [api_design.md](api_design.md)(§3 실행모델·§11 로드맵), [api/README.md](../api/README.md)

---

## 1. 전제 · 상황 구분

현재 MVP(`api/runner.py`)는 잡을 **같은 호스트에서 `python -m scripts.*` 서브프로세스**로
실행합니다. 즉 "API 호스트 = unsloth 호스트"를 가정합니다. 원격이면 **실행 위치를
옮겨야** 합니다.

무엇이 원격인지에 따라 대응이 다릅니다.

| 상황 | 무엇이 원격 | 대응 |
|---|---|---|
| **A. 학습 머신이 원격** | unsloth+GPU가 별도 호스트 | 학습 잡을 원격에서 실행 (§3 본론) |
| **B. 추론/서빙만 원격** | 이미 학습된 모델을 원격 vLLM 등으로 서빙 | `base_url`만 원격 지정 (§4) |

> **핵심 분리:** `generate_*`(데이터 생성)와 원격 추론은 unsloth가 **불필요**합니다
> (OpenAI 호환 API만 호출). unsloth+GPU가 필요한 것은 **train_* · export_model ·
> test_model** 뿐입니다. 따라서 원격화 대상은 이 세 가지입니다.

---

## 2. 무엇이 unsloth를 필요로 하나 (원격화 대상 식별)

| 스크립트 | unsloth 필요 | 원격 실행 필요 |
|---|---|---|
| `scripts/data/prepare_data` | ✗ | ✗ (CPU 어디서나) |
| `scripts/generate/*` | ✗ (원격 LLM API 호출) | ✗ |
| `scripts/train/*` | ✅ | ✅ |
| `scripts/model/export_model` | ✅ | ✅ |
| `scripts/model/test_model` | ✅ | ✅ |

→ CPU 잡(prepare/generate)은 API 호스트에서, **GPU 잡(train/export/infer)만 원격**에서.

---

## 3. 상황 A — 학습 머신이 원격일 때: 3가지 방식

### ① 원격 GPU 워커 (권장) — ✅ 구현됨 (공유 스토리지 큐)

```
[API 호스트: CPU 워커만]        (공유 스토리지 위 kp.db 큐)      [GPU 호스트: GPU 워커]
  KP_GPU_WORKER_ENABLED=false  ────────────────────────────►  python -m api.worker --role gpu
  prepare/generate 로컬 처리                                    GPU 잡 claim → python -m scripts.* 로컬 실행
         └───────────────── 공유 스토리지 (NFS 등, 동일 절대경로) ─────────────────┘
```

- **GPU 워커를 GPU 호스트에서 별도 프로세스로 구동**하고 공유 큐에서 GPU 잡을 claim →
  그 호스트 로컬에서 스크립트 실행. API 호스트엔 unsloth 불필요.
- **구현:**
  - `store.claim_next_job` — WAL + `UPDATE ... WHERE status='queued'` rowcount 판정으로
    **멀티프로세스 원자적 클레임**(두 호스트가 같은 잡을 이중 클레임하지 않음).
  - `api/worker.py` — 역할 게이팅(`--role gpu|cpu|both`)과 독립 실행 진입점.
  - `runner.run_job` 재사용 — GPU 호스트에서는 `REMOTE_ENABLED` 없이 **로컬 실행**.
- **실행 방법:**
  ```bash
  # (공유) NFS 등으로 두 호스트가 같은 절대경로의 스토리지를 공유하도록 마운트
  # API 호스트 — CPU 잡만, GPU 잡은 큐에 쌓아 원격 워커에 위임
  KP_STORAGE_ROOT=/mnt/shared/kp KP_GPU_WORKER_ENABLED=false \
    uvicorn api.main:app --port 8000
  # GPU 호스트 — unsloth 설치, 같은 스토리지, GPU 워커만
  KP_STORAGE_ROOT=/mnt/shared/kp python -m api.worker --role gpu
  ```
- 필요한 인프라: **공유 스토리지**(데이터셋·어댑터·`kp.db`를 두 호스트가 동일 절대경로로 접근).
- **확장(Phase 3):** 다중 GPU 노드·고빈도에서는 `store`의 SQLite 큐를 **Redis/Celery**로
  교체(공유 스토리지 SQLite는 단일 파일 락 경합 한계). 워커/러너 구조는 그대로 재사용.
- 장점: MVP 재사용도 높고 API 호스트에 GPU/unsloth 불필요. 단점: 공유 스토리지 필요,
  SQLite 큐는 소~중 규모까지.

> **② SSH 러너와의 차이:** ②는 API 호스트가 매 잡마다 SSH로 원격에 밀어넣고 rsync로
> 데이터를 옮긴다(단일 GPU, 인프라 최소). ①은 GPU 호스트가 **상주 워커**로 큐를
> 직접 소비한다(공유 스토리지 전제, 다중 워커 확장 용이).

### ② SSH 원격 실행 (가장 빠른 시작 · PoC) — ✅ 구현됨

```
API 호스트: 잡별 config + 데이터셋 → rsync 푸시 → SSH로 `python -m scripts.<group>.<name>`
          → stdout 스트리밍 → 산출 outputs/ rsync 회수
```

- **구현:** `api/remote.py`(push/pull/build_cmd) + `api/runner.py`가 GPU 잡일 때 원격 분기.
  config 생성·로그·메트릭 파싱 로직은 로컬과 동일하게 재사용.
- **활성화:** `KP_REMOTE_ENABLED=1`, `KP_REMOTE_HOST`, `KP_REMOTE_DIR`(+`KP_REMOTE_SSH_OPTS`).
  자세한 사용법은 [api/README.md](../api/README.md)의 "원격 unsloth 실행".
- 장점: 인프라 최소, 빠른 도입. 단점: 파일 동기화(rsync)·장애·동시성 관리가 취약, 확장 어려움.
  단일 원격 GPU 1대에 적합.
- **한계(PoC):** 취소는 로컬 ssh 프로세스를 종료하며 원격 프로세스가 잠시 잔존할 수 있음.
  스토리지는 rsync 전송(공유 마운트 아님) — 대용량/다중 노드는 ①(공유 스토리지)로 승격.

### ③ 관리형 스케줄러 (이미 클러스터가 있을 때)

- SLURM / Kubernetes Job / Ray / SkyPilot 등에 학습 잡을 제출하고 API는 제출·상태
  폴링만 담당.
- 장점: 다중 GPU·재시도·스케줄링을 인프라가 담당. 단점: 러닝커브·의존성.

### 비교

| 기준 | ① 원격 워커 ✅ | ② SSH ✅ | ③ 스케줄러 |
|---|---|---|---|
| 도입 속도 | 중 | **빠름** | 느림 |
| 확장성(다중 GPU) | **좋음**(Redis 전환 시) | 나쁨 | **좋음** |
| 인프라 요구 | 공유 스토리지 | SSH만 | 클러스터 |
| MVP 코드 재사용 | 높음 | **매우 높음** | 중 |
| 권장 용도 | 운영·확장 | PoC·단일 GPU | 기존 클러스터 |
| 상태 | 구현됨(SQLite 큐) | 구현됨 | 미구현 |

---

## 4. 상황 B — 추론/서빙만 원격

- **데이터 생성:** 이미 지원됨. `generate_*`는 `QA_GEN_BASE_URL`로 원격 OpenAI 호환
  엔드포인트(vLLM/Ollama/사내 게이트웨이)를 호출합니다. API에선 프로젝트 시크릿 또는
  요청 `llm.base_url`로 지정.
- **학습된 모델 서빙:** 병합 모델을 원격 vLLM으로 올려 저지연 추론 → api_design.md
  §11 **Phase 2(Deployment)** 와 동일. 이 경우 학습만 GPU 호스트에서 하고, 서빙은
  별도 원격 엔드포인트로 분리.

---

## 5. 공통 주의사항 (어느 방식이든)

- **스토리지 공유/전송:** 학습 잡이 읽는 `processed/*`·쓰는 `outputs/*` 경로가 실행
  호스트에서 접근 가능해야 함(NFS/S3 마운트 또는 전송). `config_builder`가 경로를
  스토리지 기준 절대경로로 주입하므로, 원격에서도 동일 경로 규약을 마운트하면 됨.
- **설치 분리:** 원격 GPU 호스트에만 루트 `requirements.txt`(unsloth/torch/…). API
  호스트는 `api/requirements.txt`만.
- **로그·취소:** 원격 실행은 stdout 스트리밍과 프로세스 취소를 원격 채널(SSH/큐 메시지)로
  이어야 함.
- **시크릿·네트워크:** 원격 접속 자격증명, 방화벽/프록시, 큐·스토리지 접근 권한.
- **버전 일치:** 원격 호스트의 `scripts/` 코드·config 규약이 API가 생성하는 config와
  일치해야 함(동일 리비전 배포).

---

## 6. 권장 로드맵

1. **PoC:** ② SSH 러너로 단일 원격 GPU에 학습 위임(빠른 검증).
2. **운영:** ① 원격 GPU 워커 + Redis 큐 + 공유 스토리지로 전환(api_design §11 Phase 3).
3. **서빙 분리:** 필요 시 Phase 2 vLLM Deployment로 추론을 원격 엔드포인트화.

---

## 7. 결정이 필요한 사항

1. 대상 상황: A(학습 원격) / B(추론 원격) / 둘 다
2. 방식: ①원격 워커 / ②SSH / ③스케줄러
3. 공유 스토리지: NFS vs S3(오브젝트) vs 전송(rsync)
4. 큐: 현재 DB 큐 유지(단일 원격) vs Redis 전환(다중)
5. GPU 노드 수(단일 vs 다중) 및 동시성 정책
