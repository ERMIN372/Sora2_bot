"""Handlers package — re-exports everything from the core module.

External code should continue to use ``from handlers import …`` or
``import handlers`` exactly as before; this package transparently wraps
the monolithic ``_core`` module while internal code is gradually split
into domain sub-modules.
"""
from handlers._core import *  # noqa: F401,F403
from handlers._core import (  # explicit re-exports used by other modules
    ADMIN_SORA_DURATION_OPTION,
    DEFAULT_VIDEO_DURATION,
    MAIN_MENU_BUTTONS,
    SESSION_MANAGER,
    STATUS_MESSAGES,
    UserSession,
    VEO_FIXED_DURATION,
    VEO_FIXED_PRICE_RUB,
    VIDEO_DURATION_OPTIONS,
    _available_duration_options,
    _create_order,
    _map_provider_error,
    _select_remote_media_asset,
    _send_job_update,
    balance_command,
    chatgpt_menu,
    chatgpt_respond,
    help_command,
    payment_callback_handler,
    referral_command,
    register_handlers,
    resend_pending_order,
    start_command,
    tarot_menu,
    top_up_menu,
    trend_answers_handler,
    trend_menu,
    trend_model_callback_handler,
    trend_select_callback_handler,
)
# Re-export states so ``from handlers import GenerationStates`` works.
from handlers._states import (  # noqa: F401
    AdminStates,
    AutoStyleStates,
    ChatGPTState,
    GenerationStates,
    TarotStates,
    TrendStates,
)
