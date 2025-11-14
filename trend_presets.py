"""Predefined trend video presets for the Telegram bot."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional


@dataclass(frozen=True, slots=True)
class TrendParam:
    """Single question/parameter collected from the user."""

    key: str
    question: str
    example: Optional[str] = None


@dataclass(frozen=True, slots=True)
class TrendPreset:
    """Description of a pre-configured trend workflow."""

    id: str
    title: str
    description: str
    params: List[TrendParam]
    prompt_templates: Mapping[str, str]


_TREND_PRESETS_DATA: Iterable[TrendPreset] = (
    TrendPreset(
        id="product_launch",
        title="Вирусный запуск продукта",
        description="Динамичный ролик с акцентом на проблему, решением и мгновенным CTA.",
        params=[
            TrendParam(
                key="product",
                question="Что продвигаем? Опиши продукт или услугу в 1 предложении.",
                example="онлайн-курс по дизайну презентаций",
            ),
            TrendParam(
                key="audience",
                question="Кто целевая аудитория и какую боль нужно задеть?",
                example="предприниматели, которым нужно быстро собирать питчи",
            ),
            TrendParam(
                key="hook",
                question="Какой броский хук используем в начале?",
                example="5 ошибок, из-за которых ваши питчи проваливаются",
            ),
            TrendParam(
                key="cta",
                question="Какое действие ожидаем от зрителя в конце?",
                example="перейти по ссылке и записаться на интенсив",
            ),
        ],
        prompt_templates={
            "sora": (
                "Создай вертикальное видео тренда TikTok/Shorts о {product}. "
                "Целевая аудитория: {audience}. Начни с хука: {hook}. "
                "Покажи 3 быстрые сцены с крупными надписями, клиповой монтаж, "
                "динамичные переходы, легкий glitch. Финал — явный CTA: {cta}."
            ),
            "veo": (
                "Сгенерируй вертикальный ролик для Reels/TikTok про {product}. "
                "Зрители: {audience}. Первые 2 секунды — хук '{hook}'. "
                "Добавь движения камеры, jump-cut, всплывающие подписи и эмодзи. "
                "Финальный кадр подчеркивает CTA: {cta}."
            ),
        },
    ),
    TrendPreset(
        id="behind_the_scenes",
        title="За кадром и процесс",
        description="Трендовый BTS-ролик: показываем этапы работы и живые эмоции.",
        params=[
            TrendParam(
                key="project",
                question="Какой проект показываем за кадром?",
                example="съемка брендового клипа для кофейни",
            ),
            TrendParam(
                key="steps",
                question="Назови 3 ключевых шага процесса (через запятую).",
                example="мудборд, подготовка реквизита, ночная съемка",
            ),
            TrendParam(
                key="vibe",
                question="Какое настроение и стиль должны чувствоваться?",
                example="ламповый неон, много смеха и живых реакций",
            ),
            TrendParam(
                key="takeaway",
                question="Что зритель должен запомнить или унести из ролика?",
                example="что команда создает атмосферу и думает о деталях",
            ),
        ],
        prompt_templates={
            "sora": (
                "Вертикальный BTS-ролик про {project}. Покажи этапы: {steps}. "
                "Настроение: {vibe}. Используй handheld-съемку, реальные эмоции, "
                "много крупняков и ambient звук. Закрепи мысль: {takeaway}."
            ),
            "veo": (
                "Сними вертикальный behind-the-scenes про {project}. Этапы: {steps}. "
                "Передай вайб: {vibe}. Добавь текстовые подписи с таймкодами, "
                "ускорения и freeze-frame. Финальный титр с мыслью: {takeaway}."
            ),
        },
    ),
    TrendPreset(
        id="expert_tip",
        title="Эксперт делится инсайтом",
        description="Короткий совет от эксперта с личным опытом и убедительным финалом.",
        params=[
            TrendParam(
                key="expert",
                question="Кто говорит в ролике и почему ему верят?",
                example="основатель стартапа с 10-летним опытом",
            ),
            TrendParam(
                key="insight",
                question="Какой главный инсайт или совет нужно донести?",
                example="как тестировать идеи без больших бюджетов",
            ),
            TrendParam(
                key="story",
                question="Добавь короткий пример или личную историю.",
                example="как первая тестовая реклама принесла 100 заявок",
            ),
            TrendParam(
                key="cta",
                question="Какой призыв или следующий шаг для зрителя?",
                example="скачать чек-лист по ссылке в профиле",
            ),
        ],
        prompt_templates={
            "sora": (
                "Вертикальное видео-совет от {expert}. Вступление — короткий сетап, "
                "затем инсайт: {insight}. Подкрепи примером: {story}. "
                "Добавь титры и субтитры, атмосферу доверия, легкую кинематографичную "
                "подсветку. Заверши четким CTA: {cta}."
            ),
            "veo": (
                "Сделай вертикальный ролик-совет: герой {expert}. "
                "Главная мысль: {insight}. Обязательно вставь пример: {story}. "
                "Темп бодрый, jump-cuts, текстовые подсказки и эмодзи. "
                "Заверши призывом: {cta}."
            ),
        },
    ),
)

TREND_PRESETS: List[TrendPreset] = list(_TREND_PRESETS_DATA)
TREND_PRESETS_BY_ID: Dict[str, TrendPreset] = {preset.id: preset for preset in TREND_PRESETS}


def get_trend_by_id(trend_id: str) -> Optional[TrendPreset]:
    """Return preset by identifier, if present."""

    if not trend_id:
        return None
    return TREND_PRESETS_BY_ID.get(trend_id)
