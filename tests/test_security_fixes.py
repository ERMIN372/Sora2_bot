"""Tests for security and stability fixes from the audit."""
from __future__ import annotations

import asyncio
import shutil
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest


# ---------------------------------------------------------------------------
# 1. validate_download_url – SSRF protection
# ---------------------------------------------------------------------------

class TestValidateDownloadUrl:
    def _validate(self, url: str) -> None:
        from providers.base import validate_download_url
        validate_download_url(url)

    def test_https_allowed(self) -> None:
        self._validate("https://api.openai.com/v1/videos/123/content")

    def test_http_blocked(self) -> None:
        with pytest.raises(ValueError, match="non-HTTPS"):
            self._validate("http://api.openai.com/v1/videos/123/content")

    def test_ftp_blocked(self) -> None:
        with pytest.raises(ValueError, match="non-HTTPS"):
            self._validate("ftp://evil.com/payload")

    def test_localhost_blocked(self) -> None:
        with pytest.raises(ValueError, match="internal host"):
            self._validate("https://localhost/secret")

    def test_127_0_0_1_blocked(self) -> None:
        with pytest.raises(ValueError, match="internal host"):
            self._validate("https://127.0.0.1/secret")

    def test_metadata_google_blocked(self) -> None:
        with pytest.raises(ValueError, match="internal host"):
            self._validate("https://metadata.google.internal/computeMetadata")

    def test_internal_suffix_blocked(self) -> None:
        with pytest.raises(ValueError, match="internal host"):
            self._validate("https://some-service.internal/api")


# ---------------------------------------------------------------------------
# 2. SessionManager – TTL eviction & size cap
# ---------------------------------------------------------------------------

class TestSessionManager:
    def test_get_creates_session(self) -> None:
        from handlers._core import SessionManager
        mgr = SessionManager()
        session = mgr.get(1)
        assert session is not None
        assert mgr.get(1) is session  # same object

    def test_ttl_eviction(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import handlers._core as core
        monkeypatch.setattr(core, "_SESSION_TTL_SECONDS", 0.0)
        monkeypatch.setattr(core, "_SESSION_CLEANUP_INTERVAL", 0.0)

        mgr = core.SessionManager()
        mgr.get(1)
        assert 1 in mgr._sessions

        # Force _last_access to be old
        mgr._sessions[1]._last_access = time.monotonic() - 10
        mgr._last_cleanup = 0  # force cleanup to run
        mgr.get(2)  # triggers _maybe_cleanup

        assert 1 not in mgr._sessions
        assert 2 in mgr._sessions

    def test_max_size_eviction(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import handlers._core as core
        monkeypatch.setattr(core, "_SESSION_MAX_SIZE", 3)
        monkeypatch.setattr(core, "_SESSION_CLEANUP_INTERVAL", 0.0)
        monkeypatch.setattr(core, "_SESSION_TTL_SECONDS", 99999.0)

        mgr = core.SessionManager()
        for uid in range(5):
            s = mgr.get(uid)
            s._last_access = time.monotonic() + uid  # stagger access times
        mgr._last_cleanup = 0  # force cleanup

        mgr.get(100)  # trigger cleanup
        assert len(mgr._sessions) <= 4  # max_size=3 + the new one just added


# ---------------------------------------------------------------------------
# 3. Payment lock eviction
# ---------------------------------------------------------------------------

class TestPaymentLockEviction:
    def test_stale_locks_evicted(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import handlers._core as core
        monkeypatch.setattr(core, "_PAYMENT_LOCK_TTL", 0.0)

        core._PENDING_PAYMENT_LOCKS.clear()
        core._PENDING_PAYMENT_LAST_ATTEMPT.clear()

        core._PENDING_PAYMENT_LOCKS[42] = asyncio.Lock()
        core._PENDING_PAYMENT_LAST_ATTEMPT[42] = time.monotonic() - 10

        lock = core._get_payment_lock(99)
        assert lock is not None
        # user 42 should have been evicted
        assert 42 not in core._PENDING_PAYMENT_LOCKS
        assert 42 not in core._PENDING_PAYMENT_LAST_ATTEMPT

        # cleanup
        core._PENDING_PAYMENT_LOCKS.clear()
        core._PENDING_PAYMENT_LAST_ATTEMPT.clear()


# ---------------------------------------------------------------------------
# 4. _TRACE_HISTORY cap
# ---------------------------------------------------------------------------

class TestTraceHistoryCap:
    def test_max_keys_enforced(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from providers.gemini import _TRACE_HISTORY, _register_trace_event
        import providers.gemini as gemini_mod

        monkeypatch.setattr(gemini_mod, "_TRACE_HISTORY_MAX_KEYS", 5)

        _TRACE_HISTORY.clear()
        for i in range(10):
            _register_trace_event(f"corr_{i}", {"event": i})

        assert len(_TRACE_HISTORY) <= 5
        _TRACE_HISTORY.clear()

    def test_per_key_limit(self) -> None:
        from providers.gemini import _TRACE_HISTORY, _register_trace_event

        _TRACE_HISTORY.clear()
        for i in range(30):
            _register_trace_event("same_key", {"event": i})

        assert len(_TRACE_HISTORY["same_key"]) <= 20
        _TRACE_HISTORY.clear()


# ---------------------------------------------------------------------------
# 5. stream_response_to_file – temp dir cleanup on error
# ---------------------------------------------------------------------------

class TestStreamResponseCleanup:
    @pytest.mark.asyncio
    async def test_cleanup_on_error(self, tmp_path: Path) -> None:
        from providers.base import stream_response_to_file

        target_dir = tmp_path / "download-test"
        target_dir.mkdir()
        file_path = target_dir / "video.mp4"

        class FailingResponse:
            async def aiter_bytes(self):
                yield b"data"
                raise IOError("disk full")

        with pytest.raises(IOError):
            await stream_response_to_file(FailingResponse(), file_path)

        assert not target_dir.exists(), "temp dir should be cleaned up on error"

    @pytest.mark.asyncio
    async def test_success_preserves_dir(self, tmp_path: Path) -> None:
        from providers.base import stream_response_to_file

        target_dir = tmp_path / "download-ok"
        target_dir.mkdir()
        file_path = target_dir / "video.mp4"

        class OkResponse:
            async def aiter_bytes(self):
                yield b"hello"
                yield b"world"

        size = await stream_response_to_file(OkResponse(), file_path)

        assert target_dir.exists()
        assert file_path.read_bytes() == b"helloworld"
        assert size == 10


# ---------------------------------------------------------------------------
# 6. iter_nodes / extract_url – shared helpers
# ---------------------------------------------------------------------------

class TestSharedHelpers:
    def test_iter_nodes_flat(self) -> None:
        from providers.base import iter_nodes
        nodes = list(iter_nodes({"a": 1, "b": {"c": 2}}))
        assert len(nodes) == 2

    def test_iter_nodes_list(self) -> None:
        from providers.base import iter_nodes
        nodes = list(iter_nodes([{"a": 1}, {"b": 2}]))
        assert len(nodes) == 2

    def test_extract_url_direct(self) -> None:
        from providers.base import extract_url
        assert extract_url({"url": "https://cdn.example.com/v.mp4"}) == "https://cdn.example.com/v.mp4"

    def test_extract_url_nested(self) -> None:
        from providers.base import extract_url
        assert extract_url({"file": {"download_url": "https://a.com/f"}}) == "https://a.com/f"

    def test_extract_url_none(self) -> None:
        from providers.base import extract_url
        assert extract_url({"unrelated": "data"}) is None


# ---------------------------------------------------------------------------
# 7. handlers package backward compatibility
# ---------------------------------------------------------------------------

class TestHandlersPackageCompat:
    def test_register_handlers_importable(self) -> None:
        from handlers import register_handlers
        assert callable(register_handlers)

    def test_states_importable(self) -> None:
        from handlers import GenerationStates, AdminStates, TarotStates
        assert hasattr(GenerationStates, "video_model")
        assert hasattr(AdminStates, "menu")
        assert hasattr(TarotStates, "choosing_type")

    def test_session_manager_importable(self) -> None:
        from handlers import SESSION_MANAGER
        assert SESSION_MANAGER is not None

    def test_resend_pending_order_importable(self) -> None:
        from handlers import resend_pending_order
        assert callable(resend_pending_order)

    def test_map_provider_error_importable(self) -> None:
        from handlers import _map_provider_error
        assert callable(_map_provider_error)


# ---------------------------------------------------------------------------
# 8. Database healthcheck
# ---------------------------------------------------------------------------

class TestDatabaseHealthcheck:
    @pytest.mark.asyncio
    async def test_interface_default_returns_true(self) -> None:
        from db.interface import DatabaseInterface

        # Test the concrete healthcheck() method directly
        result = await DatabaseInterface.healthcheck(MagicMock())
        assert result is True


# ---------------------------------------------------------------------------
# 9. CI/CD path traversal protection
# ---------------------------------------------------------------------------

class TestSafeResolve:
    """Test the path traversal protection logic (reimplemented here
    to avoid importing execute_task.py which depends on anthropic)."""

    @staticmethod
    def _safe_resolve(filepath: str) -> Path:
        """Mirror of execute_task._safe_resolve for testing."""
        resolved = Path(filepath).resolve()
        allowed = Path.cwd().resolve()
        resolved.relative_to(allowed)  # raises ValueError on traversal
        return resolved

    def test_normal_path_allowed(self) -> None:
        result = self._safe_resolve("main.py")
        assert result.name == "main.py"

    def test_traversal_blocked(self) -> None:
        with pytest.raises(ValueError):
            self._safe_resolve("../../../../etc/passwd")
