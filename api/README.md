# 지식 파이프라인 API (Phase 1 MVP)

CLI 파이프라인(준비·생성·학습·병합·추론)을 감싼 **비동기 Job 기반 HTTP API**입니다.
전체 설계는 [docs/api_design.md](../docs/api_design.md)를 참고하세요. 이 구현은 그중
**Phase 1(MVP)** 범위입니다.

## 동작 방식

요청 → 잡 생성 → 워커가 **요청별 config.yaml을 만들어 기존 스크립트를 서브프로세스로 실행**
→ stdout을 로그로 저장하고 `loss:` 등을 파싱 → 산출물(데이터셋/어댑터/모델) 등록.

- GPU 잡(train/export/infer)은 **동시성 1**로 직렬, CPU 잡(prepare/generate)은 병렬.
- 스크립트는 무수정 재사용(서브프로세스 격리). 경로는 서버가 프로젝트 스토리지로 주입.

## 설치 · 실행

```bash
pip install -r api/requirements.txt          # (학습엔 루트 requirements.txt도 필요)
export KP_API_KEYS="my-secret-key"           # 콤마로 여러 개
export KP_STORAGE_ROOT="/data/kp"            # 산출물 저장 위치(기본: ./storage)
uvicorn api.main:app --host 0.0.0.0 --port 8000
# 대화형 문서: http://localhost:8000/v1/docs
```

> 학습/생성 잡은 unsloth+CUDA GPU가 있는 호스트에서만 성공합니다. prepare/generate는
> GPU가 없어도 동작합니다(생성은 외부 LLM API 필요).

## 환경변수

| 변수 | 기본 | 설명 |
|---|---|---|
| `KP_API_KEYS` | `dev-key` | 허용 API 키(콤마 구분) |
| `KP_STORAGE_ROOT` | `./storage` | 프로젝트 데이터/산출물 루트 |
| `KP_DB_PATH` | `{STORAGE}/kp.db` | 메타 DB(SQLite) |
| `KP_GPU_CONCURRENCY` | `1` | 동시 GPU 잡 수 |
| `KP_CPU_CONCURRENCY` | `2` | 동시 CPU 잡 수 |
| `KP_SECRET_KEY` | (자동생성) | 시크릿 암호화 Fernet 키 |
| `KP_PYTHON` | 현재 인터프리터 | 스크립트 실행 파이썬 |
| `KP_REMOTE_ENABLED` | `false` | GPU 잡(train/export/infer)을 원격 SSH로 실행 |
| `KP_REMOTE_HOST` | — | 원격 GPU 호스트 (예: `user@gpu-host`) |
| `KP_REMOTE_DIR` | — | 원격 리포 체크아웃 경로 |
| `KP_REMOTE_PYTHON` | `python3` | 원격 파이썬 |
| `KP_REMOTE_SSH_OPTS` | — | 추가 ssh 옵션 (예: `-p 2222 -i ~/key`) |

### 원격 unsloth(GPU) 실행 (SSH 러너, PoC)

GPU가 없는 호스트에서 API를 돌리고 **학습·병합·추론만 원격 GPU 호스트**에서
실행할 수 있습니다(CPU 잡 prepare/generate는 항상 로컬). 설계·한계는
[../docs/remote_unsloth.md](../docs/remote_unsloth.md).

```bash
export KP_REMOTE_ENABLED=1
export KP_REMOTE_HOST=user@gpu-host
export KP_REMOTE_DIR=/opt/knowledge-pipeline    # 원격에 이 리포 체크아웃 + unsloth 설치
# (선택) export KP_REMOTE_SSH_OPTS="-p 2222 -i ~/.ssh/key"
uvicorn api.main:app --port 8000
```

동작: GPU 잡 실행 시 ① 프로젝트 스토리지를 원격에 `rsync` 푸시 → ② SSH로
`python -m scripts.<group>.<name> --config <경로>` 실행(로그 스트리밍) → ③ 성공 시
원격 `outputs/`를 로컬로 회수. 전제: 원격에 리포 체크아웃+unsloth, 로컬에 ssh/rsync,
스토리지 절대경로를 원격에서도 동일하게 접근 가능(기본은 동일 경로 미러링).

## 빠른 예시

```bash
KEY="my-secret-key"; H="Authorization: Bearer $KEY"; B=http://localhost:8000/v1

# 1) 프로젝트
PID=$(curl -s -XPOST $B/projects -H "$H" -H 'content-type: application/json' \
  -d '{"name":"asteria"}' | python3 -c 'import sys,json;print(json.load(sys.stdin)["id"])')

# 2) 원문 업로드
curl -s -XPOST $B/projects/$PID/raw-files -H "$H" \
  -F kind=doc -F file=@data/raw/sample_worldbook.md

# 3) 데이터 준비(잡)
JID=$(curl -s -XPOST $B/projects/$PID/datasets:prepare -H "$H" \
  -H 'content-type: application/json' -d '{}' \
  | python3 -c 'import sys,json;print(json.load(sys.stdin)["job_id"])')

# 4) 잡 상태 / 로그
curl -s $B/jobs/$JID -H "$H"
curl -s "$B/jobs/$JID/logs" -H "$H"

# 5) 학습(잡) — GPU 필요
curl -s -XPOST $B/projects/$PID/train -H "$H" \
  -H 'content-type: application/json' -d '{"stage":"sft"}'
```

## 엔드포인트 (요약)

`docs/api_design.md §6`과 동일. 대화형 문서(`/v1/docs`)와 `/v1/openapi.json`에서 전체 스펙 확인.

| 그룹 | 예 |
|---|---|
| 프로젝트/파일/데이터셋 | `POST /v1/projects`, `POST /v1/projects/{pid}/raw-files`, `GET /v1/projects/{pid}/datasets` |
| 파이프라인(잡) | `:prepare`, `:generate`, `/train`, `/export`, `/infer` |
| 잡 | `GET /v1/jobs/{id}`, `GET /v1/jobs/{id}/logs?follow=true`(SSE), `POST /v1/jobs/{id}:cancel` |
| 메타 | `GET /v1/healthz`, `GET /v1/config/schema` |

## 구조

```
api/
├── main.py            # FastAPI 앱 · 라우터 마운트 · 예외 처리
├── config.py          # 환경변수 설정
├── auth.py            # API 키 인증
├── store.py           # SQLite(프로젝트·잡) + 잡 큐 클레임
├── storage.py         # 프로젝트 스토리지 레이아웃
├── secrets.py         # 시크릿 암호화(Fernet, 폴백 포함)
├── config_builder.py  # 요청→config.yaml(경로 주입 + override 화이트리스트)
├── runner.py          # 잡→서브프로세스 실행·로그·메트릭·취소
├── worker.py          # GPU/CPU 워커 스레드
├── schemas.py         # 요청/응답 모델
├── routers/           # projects · pipeline · jobs
└── tests/test_e2e.py  # GPU 불필요 스모크 테스트
```

## 테스트

```bash
PYTHONPATH=. python3 -m api.tests.test_e2e
# 프로젝트→업로드→prepare(실제 서브프로세스)→데이터셋→가드까지 검증(GPU 불필요)
```

## 한계 / 다음 단계 (Phase 2+)

- 상시 저지연 서빙(vLLM Deployment)은 미포함 — 현재 추론은 잡 기반 간이 테스트.
- 스케일아웃(Celery/RQ, 다중 GPU, S3), 멀티테넌시/쿼터, 웹 콘솔은 이후 단계.
- 최종 메트릭은 스크립트가 남기는 `--report-json`(잡별 `report.json`)에서 수집합니다
  (stdout 파싱은 폴백). **실시간 step 단위 진행률**은 아직 stdout 파싱이며, TrainerCallback
  기반 정밀 진행률은 후속 작업입니다.
