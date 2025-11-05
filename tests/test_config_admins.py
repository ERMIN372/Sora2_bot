import pytest

from config import load_config


@pytest.fixture(autouse=True)
def clear_env(monkeypatch: pytest.MonkeyPatch):
    keys = [
        "TELEGRAM_BOT_TOKEN",
        "GOOGLE_SHEET_ID",
        "GOOGLE_SA_JSON_BASE64",
        "ADMIN_IDS",
        "ADMINS",
    ]
    for key in keys:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "token")
    monkeypatch.setenv("GOOGLE_SHEET_ID", "sheet")
    monkeypatch.setenv("GOOGLE_SA_JSON_BASE64", "e30=")
    yield
    for key in keys:
        monkeypatch.delenv(key, raising=False)


def test_load_config_uses_admins_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ADMINS", "1, 2,3 ")

    config = load_config()

    assert config.admin_ids == (1, 2, 3)


def test_load_config_prefers_admin_ids(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ADMINS", "5,6")
    monkeypatch.setenv("ADMIN_IDS", "7,8")

    config = load_config()

    assert config.admin_ids == (7, 8)
