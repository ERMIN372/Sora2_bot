from typing import Any, Dict


def extract_block_info(response_json: Dict[str, Any]) -> dict:
    """Извлекает информацию о блокировке из ответа Gemini/Veo."""

    info = {"reason": None, "categories": [], "raw": response_json}
    if not isinstance(response_json, dict):
        return info
    pf = response_json.get("promptFeedback") or response_json.get("prompt_feedback")
    if pf is None:
        pf = response_json.get("safetyFeedback") or response_json.get("safety_feedback")
    if isinstance(pf, dict):
        info["reason"] = pf.get("blockReason") or pf.get("block_reason")
        cats = pf.get("safetyRatings") or pf.get("safety_ratings") or []
        info["categories"] = [
            {"category": c.get("category"), "prob": c.get("probability")}
            for c in cats
            if isinstance(c, dict)
        ]
    return info


__all__ = ["extract_block_info"]
