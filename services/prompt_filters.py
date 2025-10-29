import re
from typing import Callable

from config import Models, RoutingCfg

try:
    import httpx
except ImportError:  # pragma: no cover - fallback if dependency missing
    httpx = None  # type: ignore


_BRAND_WORDS = [
    r"\bapple\b",
    r"\biphone\b",
    r"\btesla\b",
    r"\bbmw\b",
    r"\bwalmart\b",
    r"\bsamsung\b",
    r"\bdisney\b",
    r"\bmarvel\b",
    r"\bstarbucks\b",
]
# простая эвристика для очевидных случаев
_PERSON_HINTS = [
    r"\bnapoleon\b",
    r"\bbonaparte\b",
    r"\belon\b",
    r"\bmusk\b",
    r"\btaylor swift\b",
    r"\bputin\b",
    r"\bzelensky\b",
]


def scrub_brands_and_persons(text: str) -> str:
    out = text
    for pat in _BRAND_WORDS + _PERSON_HINTS:
        out = re.sub(pat, "generic subject", out, flags=re.I)
    # замены эмоциональных слов, которые часто триггерят
    out = re.sub(r"\bbikini\b", "beachwear", out, flags=re.I)
    out = re.sub(r"\bnude|nudity|topless\b", "covered clothing", out, flags=re.I)
    out = re.sub(r"\bgun|pistol|rifle|knife\b", "metal object", out, flags=re.I)
    out = re.sub(r"\bkill|shoot\b", "defeat", out, flags=re.I)
    return out.strip()


def attach_negative_prompt(user_prompt: str, negative_prompt_base: str) -> str:
    # Если пользователь уже дал negative prompt — не дублируем, просто добавим базовый безопасный хвост.
    if "negative:" in user_prompt.lower():
        return f"{user_prompt}\n\nnegative: {negative_prompt_base}"
    return f"{user_prompt}\n\nnegative: {negative_prompt_base}"


REPHRASE_SYSTEM = (
    "Перефразируй описание сцены так, чтобы оно было безопасным, "
    "нейтральным, семейно-дружелюбным и не нарушало политики, "
    "сохранив художественный смысл. Убери любые намеки на бренды, "
    "реальных людей, наготу, насилие, наркотики, опасные действия."
)


def build_rephrase_prompt(original: str) -> str:
    return f"{REPHRASE_SYSTEM}\n\nОригинал:\n{original}\n\nВерсия для безопасной генерации:"


def _httpx_client(timeout: float = 30.0) -> "httpx.Client":
    if httpx is None:  # pragma: no cover - dependency guard
        raise RuntimeError("httpx is required for rephrase_with_text_model")
    return httpx.Client(timeout=timeout)


def rephrase_with_text_model(prompt: str, api_key: str, model: str | None = None) -> str:
    """Перефразирует текст, используя текстовую модель Gemini (REST v1)."""

    if not api_key:
        return prompt
    target_model = model or Models.TEXT_MODEL
    url = (
        f"https://generativelanguage.googleapis.com/{RoutingCfg.TEXT_API_VERSION}/models/"
        f"{target_model}:generateContent"
    )
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    body = {"contents": [{"parts": [{"text": prompt}]}]}
    try:
        with _httpx_client(timeout=30.0) as session:
            response = session.post(url, headers=headers, json=body)
            response.raise_for_status()
            data = response.json()
    except Exception:  # pragma: no cover - network guard
        return prompt
    try:
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except Exception:
        return prompt


def default_rephraser(api_key: str) -> Callable[[str], str]:
    def _call(text: str) -> str:
        prompt = build_rephrase_prompt(text)
        return rephrase_with_text_model(prompt, api_key)

    return _call


__all__ = [
    "attach_negative_prompt",
    "build_rephrase_prompt",
    "default_rephraser",
    "rephrase_with_text_model",
    "scrub_brands_and_persons",
]
