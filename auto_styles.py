from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass(frozen=True, slots=True)
class AutoStyle:
    """Auto-style preset used to build a fixed prompt for car images."""

    id: str
    title: str
    prompt: str


AUTO_STYLES_PAGE1: List[AutoStyle] = [
    AutoStyle(
        id="winter_drift_dps",
        title="❄️ Зимний дрифт с ДПС",
        prompt=(
            "Create a realistic winter drift scene using the user's car from the reference photo.\n"
            "The car is sliding sideways on fresh snow, with snow spraying from under the tires.\n"
            "A Russian police car is chasing behind.\n"
            "Keep the same body shape, color and the exact license plate as in the reference image.\n"
            "Shot from medium distance with a 55mm Zeiss Otus f/1.4 look, natural daylight, photographic style, aspect ratio 4:2."
        ),
    ),
    AutoStyle(
        id="bandai_scale_model",
        title="🧊 Масштабная модель 1/7 Bandai",
        prompt=(
            "Create a 1/7 scale commercialized figure of the car from the reference photo, in a realistic style and environment.\n"
            "Place the model car on a computer desk, on a round transparent acrylic base without any text.\n"
            "On the computer screen, show the 3D modeling process (wireframe or shaded model) of the same car.\n"
            "Next to the monitor, place a BANDAI-style toy packaging box printed with the original artwork from the reference photo.\n"
            "Keep the same car color and body details as in the reference.\n"
            "Realistic studio lighting, high resolution, product photography style."
        ),
    ),
    AutoStyle(
        id="jdm_catalog",
        title="📓 Каталог JDM 90-х",
        prompt=(
            "Create a clean Nissan JDM 90s catalog style photograph using the user's car from the reference photo.\n"
            "Place the car on an empty parking lot or simple studio-like environment, with a soft neutral background.\n"
            "The car is shot in 3/4 front view, wheels straight, body clean and slightly glossy, no dirt.\n"
            "The style should look like a 1990s Japanese car brochure: simple composition, lots of negative space, soft daylight, minimal shadows, no dramatic effects.\n"
            "Keep the same body, color and license plate as in the reference.\n"
            "High resolution, realistic photography."
        ),
    ),
    AutoStyle(
        id="tokyo_night_drift",
        title="🌃 Ночной дрифт в Токио",
        prompt=(
            "Create a realistic night drift scene in Tokyo using the user's car from the reference photo.\n"
            "The car is sliding through a wet street, with reflections of neon lights on the asphalt and on the car body.\n"
            "Background: blurred city buildings, Japanese signs, colorful neon.\n"
            "Keep the same car shape, color and license plate as in the reference.\n"
            "Cinematic lighting, high contrast, 4K, realistic photography style."
        ),
    ),
]

AUTO_STYLES_PAGE2: List[AutoStyle] = [
    AutoStyle(
        id="initial_d_anime",
        title="🟣 Аниме постер Initial D",
        prompt=(
            "Create an anime style illustration inspired by Initial D, using the user's car from the reference photo.\n"
            "Place the car drifting on a mountain road at night, with motion blur on the wheels and tires, smoke coming from under the tires.\n"
            "Stylize the car in Japanese anime style but keep its shape and color recognizable as in the reference.\n"
            "Add dynamic speed lines and dramatic lighting from street lamps or moonlight.\n"
            "No characters, focus on the car.\n"
            "High detail anime poster, vertical composition."
        ),
    ),
    AutoStyle(
        id="stance_static",
        title="🛞 Stance / Static",
        prompt=(
            "Create a stance / static style photograph using the user's car from the reference image.\n"
            "Lower the car slightly and give it flush fitment wheels and a clean stance look (no extreme camber).\n"
            "Place the car in an urban parking lot or industrial background, shot low from the ground for a dramatic angle.\n"
            "Keep the same body, color and license plate as in the reference.\n"
            "Soft overcast lighting, realistic detailed photography."
        ),
    ),
    AutoStyle(
        id="track_day",
        title="🏁 Track day 4K",
        prompt=(
            "Create a realistic track day scene in 4K using the user's car from the reference photo.\n"
            "Place the car on a racing circuit, shot in motion in a fast corner, with slight body roll and motion blur on the wheels and background.\n"
            "Add curbs, track markings and safety barriers.\n"
            "Keep the same car body shape, color and license plate as in the reference.\n"
            "Bright daylight, clean realistic motorsport photography."
        ),
    ),
]


AUTO_STYLES: List[AutoStyle] = [*AUTO_STYLES_PAGE1, *AUTO_STYLES_PAGE2]
AUTO_STYLES_BY_ID: Dict[str, AutoStyle] = {style.id: style for style in AUTO_STYLES}


def get_auto_style(style_id: Optional[str]) -> Optional[AutoStyle]:
    """Return an auto-style preset by id."""

    if not style_id:
        return None
    return AUTO_STYLES_BY_ID.get(style_id)


__all__ = [
    "AutoStyle",
    "AUTO_STYLES",
    "AUTO_STYLES_PAGE1",
    "AUTO_STYLES_PAGE2",
    "get_auto_style",
]
