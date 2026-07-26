"""OpenAI 호환 API 클라이언트 — 데이터 자동 생성 스크립트들이 공용으로 사용.

vLLM / Ollama / OpenAI / 사내 게이트웨이 등 /chat/completions를 제공하는
엔드포인트면 무엇이든 동작합니다.

환경변수:
    QA_GEN_BASE_URL   예) http://localhost:11434/v1
    QA_GEN_API_KEY    (없으면 "dummy")
    QA_GEN_MODEL      예) qwen2.5:14b
"""
import json
import os
import re
import urllib.error
import urllib.request


def resolve_env() -> tuple:
    """환경변수에서 (base_url, api_key, model)을 읽는다. 없으면 안내 후 종료."""
    base_url = os.environ.get("QA_GEN_BASE_URL")
    api_key = os.environ.get("QA_GEN_API_KEY", "dummy")
    model = os.environ.get("QA_GEN_MODEL")
    if not base_url or not model:
        raise SystemExit(
            "QA_GEN_BASE_URL / QA_GEN_MODEL 환경변수를 설정하세요.\n"
            "예) export QA_GEN_BASE_URL=http://localhost:11434/v1\n"
            "    export QA_GEN_MODEL=qwen2.5:14b"
        )
    return base_url, api_key, model


def call_llm(base_url: str, api_key: str, model: str, prompt: str,
             temperature: float = 0.3, timeout: int = 300) -> str:
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
    }).encode("utf-8")
    req = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=body,
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {api_key}"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data["choices"][0]["message"]["content"]


def extract_json_array(text: str) -> list:
    """LLM 응답에서 JSON 배열을 추출. 실패하면 빈 리스트."""
    # ```json ... ``` 코드펜스를 먼저 벗겨낸다
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1)
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if not match:
        return []
    try:
        items = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []
    return items if isinstance(items, list) else []
