"""Prompt moderation helpers for Sora bot."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

PolicyScope = str

_SAFE_NOTE = "Оригинальный дизайн, без копирования существующих брендов."


@dataclass(slots=True)
class PreflightResult:
    """Outcome of a lightweight prompt moderation pass."""

    blocked: bool
    sanitized_prompt: str
    auto_sanitized: bool = False
    reason: Optional[str] = None
    scope: Optional[PolicyScope] = None
    user_message: Optional[str] = None
    matched_terms: List[str] = field(default_factory=list)


_BRAND_PATTERN = re.compile(
    r"\b("
    r"adidas|nike|puma|reebok|new balance|louis\s*vuitton|lv|gucci|prada|hermes|hermès|chanel|dior|"
    r"balenciaga|versace|fendi|cartier|supreme|burberry|ralph\s*lauren|armani|apple|samsung|"
    r"coca[-\s]?cola|pepsi|mcdonald'?s|starbucks|playboy|mercedes|bmw|audi|tesla|toyota|honda|"
    r"netflix|disney|pixar|marvel|dc\s*comics|warner|sony|google|yandex|gazprom|"
    r"луи\s*витт?он|адидас|найк|мерседес|тойота|гугл|яндекс"
    r")\b",
    flags=re.IGNORECASE,
)

_STYLE_PATTERN = re.compile(
    r"(?:в\s+стиле|как\s+у|style\s+of|in\s+the\s+style\s+of|like)\s+[^,.!?\n]{2,60}",
    flags=re.IGNORECASE,
)


def _contains_any(text: str, keywords: Iterable[str]) -> Optional[str]:
    for keyword in keywords:
        if keyword and keyword in text:
            return keyword
    return None


_ILLEGAL_KEYWORDS: Sequence[str] = (
    "террор", "теракт", "взорвать", "бомбу", "бомба", "взрывчат", "оружие", "торговля оружием",
    "наркот", "метамфетамин", "героин", "кокаин", "отмывание денег", "убийство", "преступление",
)

_VIOLENCE_KEYWORDS: Sequence[str] = (
    "кровь", "расчлен", "жесток", "убий", "насилие", "казнь", "мучить", "избивать", "gore",
    "blood", "beheading", "torture", "murder", "shooting",
)

_NSFW_KEYWORDS: Sequence[str] = (
    "порно", "эрот", "секс", "голый", "обнажен", "интим", "genitals", "nudity", "porn", "nsfw",
    "эротич", "раздет", "грудь", "соски", "sex",
)

_HATE_KEYWORDS: Sequence[str] = (
    "расизм", "расист", "нацист", "ненависть", "убрать всех", "white power", "supremacy",
    "превосходство", "ненавижу", "оскорблять", "гомофоб", "sexist",
)

_POLITICS_KEYWORDS: Sequence[str] = (
    "полит", "выборы", "кампания", "агитац", "партию", "партий", "депутат", "президент", "голосован",
    "митинг", "флаг", "герб", "политический", "политическая", "party", "election", "campaign",
)

_PERSON_KEYWORDS: Sequence[str] = (
    "elon musk", "илон маск", "tim cook", "bill gates", "steve jobs", "mark zuckerberg",
    "donald trump", "trump", "joe biden", "байден", "obama", "putin", "путин", "medvedev",
    "медведев", "zelensky", "зеленск", "navalny", "навал", "selena gomez", "taylor swift",
    "киану ривз", "keanu reeves", "emma watson", "scarlett johansson", "margot robbie",
)

_SPAM_KEYWORDS: Sequence[str] = (
    "массовая рассылка", "spam", "спам", "фейковые новости", "manipulation", "propaganda",
)

_CATEGORY_MESSAGES: Dict[PolicyScope, str] = {
    "policy.brand": "Запрос содержит упоминание реального бренда — я заменил на нейтральный вариант.",
    "policy.person": "Сервис не изображает реальных людей или селебрити.",
    "policy.politics": "Сервис не участвует в политической агитации.",
    "policy.nsfw": "Сервис не создаёт откровенный контент.",
    "policy.violence": "Сервис не может создавать сцены насилия. Попробуй описать без крови.",
    "policy.illegal": "Нельзя создавать опасный или незаконный контент.",
    "policy.hate": "Нельзя создавать контент ненависти или дискриминации.",
    "policy.spam": "Нельзя использовать сервис для спама или манипуляций.",
    "unknown": "Запрос нарушает правила контента.",
}


def policy_message(scope: PolicyScope) -> str:
    """Return a short, user-facing description for a policy scope."""

    return _CATEGORY_MESSAGES.get(scope, _CATEGORY_MESSAGES["unknown"])


def run_preflight(prompt: str) -> PreflightResult:
    """Analyse *prompt* and decide whether it should be blocked or sanitised."""

    text = (prompt or "").strip()
    lowered = text.lower()
    if not text:
        return PreflightResult(blocked=False, sanitized_prompt=text)

    keyword = _contains_any(lowered, _ILLEGAL_KEYWORDS)
    if keyword:
        return PreflightResult(
            blocked=True,
            sanitized_prompt=text,
            reason=f"illegal:{keyword}",
            scope="policy.illegal",
            user_message=policy_message("policy.illegal"),
            matched_terms=[keyword],
        )

    keyword = _contains_any(lowered, _NSFW_KEYWORDS)
    if keyword:
        return PreflightResult(
            blocked=True,
            sanitized_prompt=text,
            reason=f"nsfw:{keyword}",
            scope="policy.nsfw",
            user_message=policy_message("policy.nsfw"),
            matched_terms=[keyword],
        )

    keyword = _contains_any(lowered, _VIOLENCE_KEYWORDS)
    if keyword:
        return PreflightResult(
            blocked=True,
            sanitized_prompt=text,
            reason=f"violence:{keyword}",
            scope="policy.violence",
            user_message=policy_message("policy.violence"),
            matched_terms=[keyword],
        )

    keyword = _contains_any(lowered, _HATE_KEYWORDS)
    if keyword:
        return PreflightResult(
            blocked=True,
            sanitized_prompt=text,
            reason=f"hate:{keyword}",
            scope="policy.hate",
            user_message=policy_message("policy.hate"),
            matched_terms=[keyword],
        )

    keyword = _contains_any(lowered, _POLITICS_KEYWORDS)
    if keyword:
        return PreflightResult(
            blocked=True,
            sanitized_prompt=text,
            reason=f"politics:{keyword}",
            scope="policy.politics",
            user_message=policy_message("policy.politics"),
            matched_terms=[keyword],
        )

    keyword = _contains_any(lowered, _PERSON_KEYWORDS)
    if keyword:
        return PreflightResult(
            blocked=True,
            sanitized_prompt=text,
            reason=f"person:{keyword}",
            scope="policy.person",
            user_message=policy_message("policy.person"),
            matched_terms=[keyword],
        )

    keyword = _contains_any(lowered, _SPAM_KEYWORDS)
    if keyword:
        return PreflightResult(
            blocked=True,
            sanitized_prompt=text,
            reason=f"spam:{keyword}",
            scope="policy.spam",
            user_message=policy_message("policy.spam"),
            matched_terms=[keyword],
        )

    brand_matches = list({match.group(0) for match in _BRAND_PATTERN.finditer(text)})
    if brand_matches:
        sanitized = _BRAND_PATTERN.sub("нейтральный дизайн", text)
        sanitized = _STYLE_PATTERN.sub("в оригинальном стиле", sanitized)
        if _SAFE_NOTE.lower() not in sanitized.lower():
            if sanitized.endswith(('.', '!', '?')):
                sanitized = sanitized.rstrip('.!? ') + f". {_SAFE_NOTE}"
            else:
                sanitized = f"{sanitized.strip()} {_SAFE_NOTE}".strip()
        sanitized = re.sub(r"\s+", " ", sanitized).strip()
        reason = "brand:" + ",".join(sorted({match.lower() for match in brand_matches}))
        return PreflightResult(
            blocked=False,
            sanitized_prompt=sanitized,
            auto_sanitized=True,
            reason=reason,
            scope="policy.brand",
            user_message=policy_message("policy.brand"),
            matched_terms=[match.lower() for match in brand_matches],
        )

    return PreflightResult(blocked=False, sanitized_prompt=text)


_SAFETY_CATEGORY_MAP: Dict[str, PolicyScope] = {
    "HARM_CATEGORY_SEXUAL": "policy.nsfw",
    "HARM_CATEGORY_SEXUAL_AND_NUDITY": "policy.nsfw",
    "HARM_CATEGORY_SEXUALLY_EXPLICIT": "policy.nsfw",
    "HARM_CATEGORY_HATE_SPEECH": "policy.hate",
    "HARM_CATEGORY_HARASSMENT": "policy.hate",
    "HARM_CATEGORY_VIOLENCE": "policy.violence",
    "HARM_CATEGORY_DANGEROUS": "policy.illegal",
    "HARM_CATEGORY_DANGEROUS_CONTENT": "policy.illegal",
    "HARM_CATEGORY_SELF_HARM": "policy.violence",
    "HARM_CATEGORY_CIVIC_INTEGRITY": "policy.illegal",
}


def _collect_safety_categories(ratings: Iterable[Dict[str, object]]) -> List[str]:
    categories: List[str] = []
    for item in ratings:
        category = item.get("category") if isinstance(item, dict) else None
        if isinstance(category, str):
            categories.append(category)
    return categories


def classify_safety_response(
    payload: Optional[Dict[str, object]],
) -> Tuple[PolicyScope, List[str], Dict[str, object]]:
    """Inspect Gemini safety payload and return policy scope information."""

    if not isinstance(payload, dict):
        return "unknown", [], {}

    block_reason = payload.get("blockReason") or payload.get("block_reason")
    finish_reason = payload.get("finishReason") or payload.get("finish_reason")
    categories: List[str] = []

    prompt_feedback = payload.get("promptFeedback") or payload.get("prompt_feedback")
    if isinstance(prompt_feedback, dict):
        block_reason = prompt_feedback.get("blockReason") or block_reason
        ratings = prompt_feedback.get("safetyRatings") or prompt_feedback.get("safety_ratings")
        if isinstance(ratings, list):
            categories.extend(_collect_safety_categories(ratings))

    candidates = payload.get("candidates")
    if isinstance(candidates, list) and candidates:
        first = candidates[0]
        if isinstance(first, dict):
            finish_reason = first.get("finishReason") or finish_reason
            ratings = first.get("safetyRatings") or first.get("safety_ratings")
            if isinstance(ratings, list):
                categories.extend(_collect_safety_categories(ratings))

    mapped_scopes: List[PolicyScope] = []
    for category in categories:
        scope = _SAFETY_CATEGORY_MAP.get(str(category))
        if scope and scope not in mapped_scopes:
            mapped_scopes.append(scope)

    scope = mapped_scopes[0] if mapped_scopes else "unknown"
    summary: Dict[str, object] = {}
    if block_reason:
        summary["blockReason"] = block_reason
    if finish_reason:
        summary["finishReason"] = finish_reason
    if categories:
        summary["categories"] = categories

    return scope, categories, summary


def render_error_json(summary: Dict[str, object]) -> str:
    """Return a compact JSON string for logging safety info."""

    if not summary:
        return ""
    try:
        data = json.dumps(summary, ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError):
        return ""
    if len(data) > 400:
        return data[:397] + "…"
    return data


__all__ = [
    "PreflightResult",
    "PolicyScope",
    "classify_safety_response",
    "policy_message",
    "render_error_json",
    "run_preflight",
]
