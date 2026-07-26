"""프로젝트 시크릿(LLM API 키 등) 저장.

기본은 Fernet(대칭키) 암호화. 키는 KP_SECRET_KEY 환경변수, 없으면 스토리지 루트에
자동 생성해 재사용한다. cryptography가 설치/동작하지 않는 환경에서는 **경고와 함께**
base64 난독화로 폴백한다(보안 아님 — 운영에서는 반드시 cryptography 설치).

시크릿 값은 응답·로그에 절대 노출하지 않는다(라우터는 키 이름만 반환).
"""
import base64
import json
import logging

from . import config, storage

log = logging.getLogger("kp.secrets")
_KEY_FILE = config.STORAGE_ROOT / ".fernet_key"
_warned = False


def _fernet():
    """Fernet 인스턴스. cryptography 사용 불가 시 None (폴백 신호)."""
    global _warned
    try:
        from cryptography.fernet import Fernet
    except BaseException:  # noqa: BLE001  (미설치·rust 바인딩 panic 등 모두 폴백)
        if not _warned:
            log.warning("cryptography 사용 불가 → 시크릿을 base64로만 저장합니다"
                        "(보안 아님). 운영에서는 cryptography를 설치하세요.")
            _warned = True
        return None

    key = config.SECRET_KEY
    if key:
        return Fernet(key.encode() if isinstance(key, str) else key)
    config.ensure_storage()
    if _KEY_FILE.exists():
        key = _KEY_FILE.read_bytes()
    else:
        key = Fernet.generate_key()
        _KEY_FILE.write_bytes(key)
        _KEY_FILE.chmod(0o600)
    return Fernet(key)


def _encrypt(data: bytes) -> bytes:
    f = _fernet()
    if f is not None:
        return b"F:" + f.encrypt(data)
    return b"B:" + base64.b64encode(data)


def _decrypt(blob: bytes) -> bytes:
    if blob.startswith(b"F:"):
        f = _fernet()
        if f is None:
            raise RuntimeError("암호화된 시크릿을 열려면 cryptography가 필요합니다.")
        return f.decrypt(blob[2:])
    if blob.startswith(b"B:"):
        return base64.b64decode(blob[2:])
    raise ValueError("알 수 없는 시크릿 형식")


def save_secrets(pid: str, values: dict) -> None:
    """기존 시크릿에 병합 저장. None 값은 삭제로 취급."""
    current = load_secrets(pid)
    for k, v in values.items():
        if v is None:
            current.pop(k, None)
        else:
            current[k] = str(v)
    blob = _encrypt(json.dumps(current).encode("utf-8"))
    path = storage.secrets_path(pid)
    path.write_bytes(blob)
    path.chmod(0o600)


def load_secrets(pid: str) -> dict:
    path = storage.secrets_path(pid)
    if not path.exists():
        return {}
    try:
        return json.loads(_decrypt(path.read_bytes()).decode("utf-8"))
    except Exception:  # noqa: BLE001  (키 불일치·손상)
        return {}


def secret_names(pid: str) -> list[str]:
    return sorted(load_secrets(pid).keys())
