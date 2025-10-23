"""Google Sheets-backed data access layer for the video generation bot."""
from __future__ import annotations

import asyncio
import copy
import base64
import binascii
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Awaitable, Callable, Dict, Iterable, List, Optional, Set, Tuple

import gspread
from google.oauth2.service_account import Credentials
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from asyncio import Lock


log = logging.getLogger(__name__)


_MAX_CELL_LENGTH = 50000


def _sanitise_cell_value(value: Any, *, column: Optional[str] = None) -> Any:
    """Normalise a value so it respects the Google Sheets cell limits."""

    if value is None:
        return ""
    if isinstance(value, (int, float, bool)):
        return value

    text = str(value)
    if not text:
        return ""

    lowered = text.lower()
    if lowered.startswith("data:image") or len(text) > 30000:
        log.warning(
            "Replacing inline asset value for column %s (length %d) with placeholder to stay under Google Sheets limits",
            column or "?",
            len(text),
        )
        return "[inline asset truncated]"

    if len(text) <= _MAX_CELL_LENGTH:
        return text

    suffix = f"… [truncated; original length {len(text)} chars]"
    if len(suffix) >= _MAX_CELL_LENGTH:
        truncated = suffix[:_MAX_CELL_LENGTH]
    else:
        truncated = text[: _MAX_CELL_LENGTH - len(suffix)] + suffix
    log.warning(
        "Truncated value for column %s from %d to %d characters", column or "?", len(text), len(truncated)
    )
    return truncated


_USERS_HEADERS = [
    "user_id",
    "credits",
    "bonus_granted",
    "economy_v2",
    "created_at",
    "updated_at",
    "notes",
    "username",
    "first_name",
    "last_name",
    "display_name",
    "tg_link",
]

_PAYMENTS_HEADERS = [
    "provider",
    "ext_id",
    "user_id",
    "amount_cp",
    "items",
    "status",
    "payload",
    "metadata",
    "created_at",
    "updated_at",
    "idempotency_key",
    "username",
    "package_id",
    "purchased_credits",
    "processed_at",
    "type",
    "ref_payment_id",
]

_JOBS_HEADERS = [
    "job_id",
    "user_id",
    "prompt",
    "image_file_id",
    "sora_req_id",
    "status",
    "video_url",
    "error",
    "created_at",
    "updated_at",
    "size",
    "seconds",
    "model",
    "cost_credits",
    "username",
    "corr_id",
    "content_type",
    "status_message_id",
    "status_message_index",
    "status_message_updated_at",
]

_ERRORS_HEADERS = [
    "ts",
    "user_id",
    "username",
    "corr_id",
    "job_id",
    "model",
    "size",
    "status_code",
    "error_type",
    "error_msg_short",
    "refunded",
    "preflight_blocked",
    "preflight_reason",
    "auto_sanitized",
    "sanitized_prompt",
    "error_scope",
    "error_json",
]

_ARCHIVE_HEADERS = [
    "ts",
    "corr_id",
    "archive_status",
    "channel_id",
    "message_id",
    "content_type",
    "model_name",
    "username",
    "user_id",
    "caption_len",
    "file_size",
    "duration_seconds",
    "attempts",
    "error_short",
]

_API_RETRY = dict(
    retry=retry_if_exception_type(gspread.exceptions.APIError),
    wait=wait_exponential(multiplier=0.5, max=8),
    stop=stop_after_attempt(5),
    reraise=True,
)


@dataclass
class _SheetState:
    worksheet: gspread.Worksheet
    headers: List[str]
    key_columns: Tuple[str, ...]
    index: Dict[Any, int]
    rows: Dict[Any, Dict[str, Any]]
    next_row: int
    lock: Lock


_CLIENT: Optional[gspread.Client] = None
_SPREADSHEET: Optional[gspread.Spreadsheet] = None

_USERS_STATE: Optional[_SheetState] = None
_PAYMENTS_STATE: Optional[_SheetState] = None
_JOBS_STATE: Optional[_SheetState] = None
_ERRORS_STATE: Optional[_SheetState] = None
_ARCHIVE_WORKSHEET: Optional[gspread.Worksheet] = None
_ARCHIVE_LOCK: Lock = Lock()
_ARCHIVE_SENT: Set[str] = set()


async def init() -> None:
    """Initialise the Google Sheets client and build indices."""

    await _ensure_spreadsheet()
    await _ensure_users_state()
    await _ensure_payments_state()
    await _ensure_jobs_state()
    await _ensure_errors_state()
    await _ensure_archive_sheet()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _env(key: str, default: Optional[str] = None) -> str:
    value = os.getenv(key, default)
    if value is None or not value.strip():
        raise RuntimeError(f"Environment variable {key} is required for Google Sheets access")
    return value.strip()


def _decode_service_account() -> Dict[str, Any]:
    raw = _env("GOOGLE_SA_JSON_BASE64")
    try:
        decoded = base64.b64decode(raw, validate=True)
        text = decoded.decode("utf-8")
    except (binascii.Error, UnicodeDecodeError):
        text = raw
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:  # pragma: no cover - configuration error
        raise RuntimeError("GOOGLE_SA_JSON_BASE64 must contain JSON or base64-encoded JSON") from exc


def _now() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat()


def _column_letter(idx: int) -> str:
    result = ""
    while idx > 0:
        idx, remainder = divmod(idx - 1, 26)
        result = chr(65 + remainder) + result
    return result or "A"


async def _to_thread(func, *args, **kwargs):
    return await asyncio.to_thread(func, *args, **kwargs)


@retry(**_API_RETRY)
def _worksheet_update(worksheet: gspread.Worksheet, range_name: str, values: List[List[Any]]) -> None:
    worksheet.update(range_name, values, value_input_option="USER_ENTERED")


@retry(**_API_RETRY)
def _worksheet_batch_update(worksheet: gspread.Worksheet, body: List[Dict[str, Any]]) -> None:
    safe_body = copy.deepcopy(body)
    worksheet.batch_update(safe_body)


@retry(**_API_RETRY)
def _worksheet_append(worksheet: gspread.Worksheet, values: List[Any]) -> None:
    worksheet.append_row(values, value_input_option="USER_ENTERED")


@retry(**_API_RETRY)
def _open_spreadsheet(client: gspread.Client, sheet_id: str) -> gspread.Spreadsheet:
    return client.open_by_key(sheet_id)


@retry(**_API_RETRY)
def _get_worksheet(spreadsheet: gspread.Spreadsheet, title: str) -> gspread.Worksheet:
    return spreadsheet.worksheet(title)


@retry(**_API_RETRY)
def _add_worksheet(spreadsheet: gspread.Spreadsheet, title: str, rows: int, cols: int) -> gspread.Worksheet:
    return spreadsheet.add_worksheet(title=title, rows=str(rows), cols=str(cols))


def _service_client() -> gspread.Client:
    global _CLIENT
    if _CLIENT is None:
        info = _decode_service_account()
        credentials = Credentials.from_service_account_info(
            info, scopes=["https://www.googleapis.com/auth/spreadsheets"]
        )
        _CLIENT = gspread.authorize(credentials)
    return _CLIENT


async def _ensure_spreadsheet() -> gspread.Spreadsheet:
    global _SPREADSHEET
    if _SPREADSHEET is not None:
        return _SPREADSHEET
    client = _service_client()
    sheet_id = _env("GOOGLE_SHEET_ID")
    _SPREADSHEET = await _to_thread(_open_spreadsheet, client, sheet_id)
    return _SPREADSHEET


async def _ensure_sheet(title: str, headers: List[str]) -> Tuple[gspread.Worksheet, List[str]]:
    spreadsheet = await _ensure_spreadsheet()
    try:
        worksheet = await _to_thread(_get_worksheet, spreadsheet, title)
    except gspread.exceptions.WorksheetNotFound:
        worksheet = await _to_thread(_add_worksheet, spreadsheet, title, 1000, len(headers) + 2)
    actual_headers = await _ensure_headers(worksheet, headers)
    return worksheet, actual_headers


async def _ensure_headers(worksheet: gspread.Worksheet, headers: List[str]) -> List[str]:
    current = await _to_thread(worksheet.row_values, 1)
    existing = [str(value).strip() if value is not None else "" for value in current]
    existing = [value for value in existing if value]
    if not existing:
        existing = []
    missing = [header for header in headers if header not in existing]
    final_headers = existing + missing
    if final_headers != existing:
        range_name = f"A1:{_column_letter(len(final_headers))}1"
        await _to_thread(_worksheet_update, worksheet, range_name, [final_headers])
    return final_headers


def _build_key(columns: Tuple[str, ...], row: Dict[str, Any]) -> Any:
    if len(columns) == 1:
        return row.get(columns[0])
    return tuple(row.get(col) for col in columns)


def _parse_int(value: Any, default: int = 0) -> int:
    if value in (None, "", " "):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return default


def _parse_str(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def _clean_username(value: Optional[str]) -> str:
    raw = _parse_str(value).strip()
    return raw.lstrip("@") if raw else ""


def _clean_name(value: Optional[str]) -> str:
    return _parse_str(value).strip()


def _build_display_name(first_name: str, last_name: str) -> str:
    parts = [part for part in (first_name, last_name) if part]
    return " ".join(parts)


def _build_tg_link(user_id: int, username: str) -> str:
    return f"https://t.me/{username}" if username else f"tg://user?id={user_id}"


def _prepare_profile(user_id: int, username: Optional[str], first_name: Optional[str], last_name: Optional[str]) -> Dict[str, str]:
    clean_username = _clean_username(username)
    clean_first = _clean_name(first_name)
    clean_last = _clean_name(last_name)
    display_name = _build_display_name(clean_first, clean_last)
    tg_link = _build_tg_link(user_id, clean_username)
    return {
        "username": clean_username,
        "first_name": clean_first,
        "last_name": clean_last,
        "display_name": display_name,
        "tg_link": tg_link,
    }


def _normalise_user(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "user_id": _parse_int(row.get("user_id")),
        "credits": _parse_int(row.get("credits")),
        "bonus_granted": bool(_parse_int(row.get("bonus_granted"))),
        "economy_v2": bool(_parse_int(row.get("economy_v2"))),
        "created_at": _parse_str(row.get("created_at")),
        "updated_at": _parse_str(row.get("updated_at")),
        "notes": _parse_str(row.get("notes")),
        "username": _parse_str(row.get("username")),
        "first_name": _parse_str(row.get("first_name")),
        "last_name": _parse_str(row.get("last_name")),
        "display_name": _parse_str(row.get("display_name")),
        "tg_link": _parse_str(row.get("tg_link")),
    }


def _normalise_payment(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "provider": _parse_str(row.get("provider")),
        "ext_id": _parse_str(row.get("ext_id")),
        "user_id": _parse_int(row.get("user_id")),
        "amount_cp": _parse_int(row.get("amount_cp")),
        "items": _parse_int(row.get("items")),
        "status": _parse_str(row.get("status")) or "pending",
        "payload": _parse_str(row.get("payload")),
        "metadata": _parse_str(row.get("metadata")),
        "created_at": _parse_str(row.get("created_at")),
        "updated_at": _parse_str(row.get("updated_at")),
        "idempotency_key": _parse_str(row.get("idempotency_key")),
        "username": _parse_str(row.get("username")),
    }


def _normalise_job(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "job_id": _parse_str(row.get("job_id")),
        "user_id": _parse_int(row.get("user_id")),
        "prompt": _parse_str(row.get("prompt")),
        "image_file_id": _parse_str(row.get("image_file_id")),
        "sora_req_id": _parse_str(row.get("sora_req_id")),
        "status": _parse_str(row.get("status")) or "queued",
        "video_url": _parse_str(row.get("video_url")),
        "error": _parse_str(row.get("error")),
        "created_at": _parse_str(row.get("created_at")),
        "updated_at": _parse_str(row.get("updated_at")),
        "size": _parse_str(row.get("size")),
        "seconds": _parse_int(row.get("seconds")),
        "model": _parse_str(row.get("model")),
        "cost_credits": _parse_int(row.get("cost_credits")),
        "username": _parse_str(row.get("username")),
        "corr_id": _parse_str(row.get("corr_id")),
        "content_type": _parse_str(row.get("content_type")) or "video",
        "status_message_id": _parse_int(row.get("status_message_id")),
        "status_message_index": _parse_int(row.get("status_message_index")),
        "status_message_updated_at": _parse_str(row.get("status_message_updated_at")),
    }


def _normalise_error(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "ts": _parse_str(row.get("ts")),
        "user_id": _parse_int(row.get("user_id")),
        "username": _parse_str(row.get("username")),
        "corr_id": _parse_str(row.get("corr_id")),
        "job_id": _parse_str(row.get("job_id")),
        "model": _parse_str(row.get("model")),
        "size": _parse_str(row.get("size")),
        "status_code": _parse_int(row.get("status_code")),
        "error_type": _parse_str(row.get("error_type")),
        "error_msg_short": _parse_str(row.get("error_msg_short")),
        "refunded": _parse_str(row.get("refunded")),
    }


def _build_state_data(
    worksheet: gspread.Worksheet,
    headers: List[str],
    key_columns: Tuple[str, ...],
    normalise,
) -> Tuple[Dict[Any, int], Dict[Any, Dict[str, Any]], int]:
    records = worksheet.get_all_records(default_blank="")
    index: Dict[Any, int] = {}
    rows: Dict[Any, Dict[str, Any]] = {}
    for offset, raw in enumerate(records, start=2):
        normalised = normalise(raw)
        key = _build_key(key_columns, normalised)
        if key in index:
            # keep first occurrence
            continue
        index[key] = offset
        rows[key] = normalised
    next_row = len(records) + 2
    if next_row < 2:
        next_row = 2
    return index, rows, next_row


async def _ensure_users_state() -> _SheetState:
    global _USERS_STATE
    if _USERS_STATE is None:
        title = os.getenv("GS_USERS_SHEET", "users")
        worksheet, headers = await _ensure_sheet(title, _USERS_HEADERS)
        index, rows, next_row = await _to_thread(
            _build_state_data, worksheet, headers, ("user_id",), _normalise_user
        )
        _USERS_STATE = _SheetState(
            worksheet=worksheet,
            headers=headers,
            key_columns=("user_id",),
            index=index,
            rows=rows,
            next_row=next_row,
            lock=Lock(),
        )
    return _USERS_STATE


async def _ensure_payments_state() -> _SheetState:
    global _PAYMENTS_STATE
    if _PAYMENTS_STATE is None:
        title = os.getenv("GS_PAYMENTS_SHEET", "payments")
        worksheet, headers = await _ensure_sheet(title, _PAYMENTS_HEADERS)
        index, rows, next_row = await _to_thread(
            _build_state_data,
            worksheet,
            headers,
            ("provider", "ext_id"),
            _normalise_payment,
        )
        _PAYMENTS_STATE = _SheetState(
            worksheet=worksheet,
            headers=headers,
            key_columns=("provider", "ext_id"),
            index=index,
            rows=rows,
            next_row=next_row,
            lock=Lock(),
        )
    return _PAYMENTS_STATE


async def _ensure_jobs_state() -> _SheetState:
    global _JOBS_STATE
    if _JOBS_STATE is None:
        title = os.getenv("GS_JOBS_SHEET", "jobs")
        worksheet, headers = await _ensure_sheet(title, _JOBS_HEADERS)
        index, rows, next_row = await _to_thread(
            _build_state_data, worksheet, headers, ("job_id",), _normalise_job
        )
        _JOBS_STATE = _SheetState(
            worksheet=worksheet,
            headers=headers,
            key_columns=("job_id",),
            index=index,
            rows=rows,
            next_row=next_row,
            lock=Lock(),
        )
    return _JOBS_STATE


async def _ensure_errors_state() -> _SheetState:
    global _ERRORS_STATE
    if _ERRORS_STATE is None:
        title = os.getenv("GS_ERRORS_SHEET", "errors")
        worksheet, headers = await _ensure_sheet(title, _ERRORS_HEADERS)
        index, rows, next_row = await _to_thread(
            _build_state_data,
            worksheet,
            headers,
            ("ts", "corr_id", "job_id"),
            _normalise_error,
        )
        _ERRORS_STATE = _SheetState(
            worksheet=worksheet,
            headers=headers,
            key_columns=("ts", "corr_id", "job_id"),
            index=index,
            rows=rows,
            next_row=next_row,
            lock=Lock(),
        )
    return _ERRORS_STATE


async def _ensure_archive_sheet() -> gspread.Worksheet:
    global _ARCHIVE_WORKSHEET, _ARCHIVE_SENT
    if _ARCHIVE_WORKSHEET is None:
        title = os.getenv("GS_ARCHIVE_SHEET", "archive")
        worksheet, _ = await _ensure_sheet(title, _ARCHIVE_HEADERS)
        records = await _to_thread(worksheet.get_all_records, default_blank="")
        sent: Set[str] = set()
        for record in records:
            status = str(record.get("archive_status", "")).strip().lower()
            corr_id = str(record.get("corr_id", "")).strip()
            if status == "sent" and corr_id:
                sent.add(corr_id)
        async with _ARCHIVE_LOCK:
            if _ARCHIVE_WORKSHEET is None:
                _ARCHIVE_WORKSHEET = worksheet
                _ARCHIVE_SENT = sent
    return _ARCHIVE_WORKSHEET


async def list_worksheet_titles() -> List[str]:
    spreadsheet = await _ensure_spreadsheet()
    worksheets = await _to_thread(spreadsheet.worksheets)
    return [sheet.title for sheet in worksheets]


def _row_range(state: _SheetState, row: int) -> str:
    return f"A{row}:{_column_letter(len(state.headers))}{row}"


def _ensure_now(value: Optional[str]) -> str:
    return value or _now()


async def _write_row(state: _SheetState, row_index: int, data: Dict[str, Any]) -> None:
    values = []
    for header in state.headers:
        sanitised = _sanitise_cell_value(data.get(header, ""), column=header)
        data[header] = sanitised
        values.append(sanitised)
    await _to_thread(_worksheet_update, state.worksheet, _row_range(state, row_index), [values])


async def _update_cells(state: _SheetState, updates: Dict[str, Tuple[int, Any]]) -> None:
    requests = []
    for column, (row, value) in updates.items():
        col_idx = state.headers.index(column) + 1
        col_letter = _column_letter(col_idx)
        sanitised = _sanitise_cell_value(value, column=column)
        requests.append({"range": f"{col_letter}{row}", "values": [[sanitised]]})
    if requests:
        await _to_thread(_worksheet_batch_update, state.worksheet, requests)


async def _create_user_record(state: _SheetState, user_id: int) -> Dict[str, Any]:
    now = _now()
    row_index = state.next_row
    record = {
        "user_id": user_id,
        "credits": 0,
        "bonus_granted": 0,
        "economy_v2": 0,
        "created_at": now,
        "updated_at": now,
        "notes": "",
        "username": "",
        "first_name": "",
        "last_name": "",
        "display_name": "",
        "tg_link": _build_tg_link(user_id, ""),
    }
    await _write_row(state, row_index, record)
    state.index[user_id] = row_index
    state.rows[user_id] = record
    state.next_row = row_index + 1
    return record


# ---------------------------------------------------------------------------
# User profile helpers
# ---------------------------------------------------------------------------


async def upsert_user_profile(
    user_id: int,
    *,
    username: Optional[str] = None,
    first_name: Optional[str] = None,
    last_name: Optional[str] = None,
) -> Dict[str, Any]:
    profile = _prepare_profile(user_id, username, first_name, last_name)
    state = await _ensure_users_state()
    async with state.lock:
        record = state.rows.get(user_id)
        if record is None:
            record = await _create_user_record(state, user_id)
        row = state.index[user_id]
        updates: Dict[str, Tuple[int, Any]] = {}
        changed = False
        for key, value in profile.items():
            current_value = _parse_str(record.get(key))
            if current_value != value:
                record[key] = value
                updates[key] = (row, value)
                changed = True
        if changed:
            record["updated_at"] = _now()
            updates["updated_at"] = (row, record["updated_at"])
            await _update_cells(state, updates)
        return dict(record)


_USER_PROFILE_BACKFILL_DONE = False


def _needs_backfill(record: Dict[str, Any]) -> bool:
    for field in ("username", "first_name", "last_name", "tg_link"):
        value = _parse_str(record.get(field))
        if not value:
            return True
    return False


async def backfill_user_profiles(
    fetcher: Callable[[int], Awaitable[Optional[Dict[str, Optional[str]]]]],
    *,
    delay: float = 0.2,
) -> None:
    global _USER_PROFILE_BACKFILL_DONE
    if _USER_PROFILE_BACKFILL_DONE:
        return
    _USER_PROFILE_BACKFILL_DONE = True
    try:
        state = await _ensure_users_state()
    except Exception:  # pragma: no cover - defensive initialisation
        log.warning("Failed to initialise users sheet for backfill", exc_info=True)
        return

    async with state.lock:
        candidates = [
            user_id
            for user_id, record in state.rows.items()
            if _needs_backfill(record)
        ]

    if not candidates:
        return

    log.info("Backfilling Telegram profiles for %s users", len(candidates))
    for user_id in candidates:
        try:
            profile = await fetcher(user_id)
        except Exception:  # pragma: no cover - external dependency
            log.warning("Backfill fetch failed for user %s", user_id, exc_info=True)
            continue
        if not profile:
            continue
        try:
            await upsert_user_profile(
                user_id,
                username=profile.get("username"),
                first_name=profile.get("first_name"),
                last_name=profile.get("last_name"),
            )
        except Exception:  # pragma: no cover - keep bot running
            log.warning("Backfill update failed for user %s", user_id, exc_info=True)
        if delay:
            await asyncio.sleep(delay)


# ---------------------------------------------------------------------------
# Users
# ---------------------------------------------------------------------------


async def get_or_create_user(user_id: int) -> Dict[str, Any]:
    state = await _ensure_users_state()
    async with state.lock:
        record = state.rows.get(user_id)
        if record is None:
            record = await _create_user_record(state, user_id)
        return dict(record)


async def add_credits(user_id: int, delta: int) -> int:
    state = await _ensure_users_state()
    async with state.lock:
        record = state.rows.get(user_id)
        if record is None:
            record = await _create_user_record(state, user_id)
        new_balance = record["credits"] + delta
        if new_balance < 0:
            raise ValueError("Insufficient credits")
        record["credits"] = new_balance
        record["updated_at"] = _now()
        row = state.index[user_id]
        await _update_cells(
            state,
            {
                "credits": (row, new_balance),
                "updated_at": (row, record["updated_at"]),
            },
        )
        return new_balance


async def set_user_credits(user_id: int, value: int) -> int:
    state = await _ensure_users_state()
    async with state.lock:
        record = state.rows.get(user_id)
        if record is None:
            record = await _create_user_record(state, user_id)
        record["credits"] = value
        record["updated_at"] = _now()
        row = state.index[user_id]
        await _update_cells(
            state,
            {
                "credits": (row, value),
                "updated_at": (row, record["updated_at"]),
            },
        )
        return value


async def mark_bonus_granted(user_id: int) -> None:
    state = await _ensure_users_state()
    async with state.lock:
        record = state.rows.get(user_id)
        if record is None:
            record = await _create_user_record(state, user_id)
        if record.get("bonus_granted"):
            return
        record["bonus_granted"] = True
        record["updated_at"] = _now()
        row = state.index[user_id]
        await _update_cells(
            state,
            {
                "bonus_granted": (row, 1),
                "updated_at": (row, record["updated_at"]),
            },
        )


async def mark_economy_v2(user_id: int) -> None:
    state = await _ensure_users_state()
    async with state.lock:
        record = state.rows.get(user_id)
        if record is None:
            record = await _create_user_record(state, user_id)
        if record.get("economy_v2"):
            return
        record["economy_v2"] = 1
        record["updated_at"] = _now()
        row = state.index[user_id]
        await _update_cells(
            state,
            {
                "economy_v2": (row, 1),
                "updated_at": (row, record["updated_at"]),
            },
        )


async def list_users() -> List[Dict[str, Any]]:
    state = await _ensure_users_state()
    async with state.lock:
        return [dict(record) for record in state.rows.values()]


# ---------------------------------------------------------------------------
# Payments
# ---------------------------------------------------------------------------


async def create_payment_record(
    provider: str,
    ext_id: str,
    user_id: int,
    amount_cp: int,
    items: int,
    status: str,
    payload: str,
    metadata: Dict[str, Any],
    idempotency_key: str = "",
    username: Optional[str] = None,
    *,
    package_id: Optional[str] = None,
    purchased_credits: Optional[int] = None,
    processed_at: Optional[str] = None,
    record_type: str = "payment",
    ref_payment_id: Optional[str] = None,
) -> None:
    state = await _ensure_payments_state()
    key = (provider, ext_id)
    username_clean = _clean_username(username) if username is not None else None
    async with state.lock:
        record = state.rows.get(key)
        now = _now()
        metadata_str = json.dumps(metadata) if isinstance(metadata, dict) else _parse_str(metadata)
        if record:
            updates: Dict[str, Tuple[int, Any]] = {}
            row = state.index[key]
            if status and record.get("status") != status:
                record["status"] = status
                updates["status"] = (row, record["status"])
            if payload and record.get("payload") != payload:
                record["payload"] = payload
                updates["payload"] = (row, record["payload"])
            if metadata_str and record.get("metadata") != metadata_str:
                record["metadata"] = metadata_str
                updates["metadata"] = (row, record["metadata"])
            if username_clean is not None and record.get("username") != username_clean:
                record["username"] = username_clean
                updates["username"] = (row, username_clean)
            if package_id is not None and record.get("package_id") != package_id:
                record["package_id"] = package_id
                updates["package_id"] = (row, package_id)
            if purchased_credits is not None and record.get("purchased_credits") != purchased_credits:
                record["purchased_credits"] = purchased_credits
                updates["purchased_credits"] = (row, purchased_credits)
            if processed_at is not None and record.get("processed_at") != processed_at:
                record["processed_at"] = processed_at
                updates["processed_at"] = (row, processed_at)
            if record_type and record.get("type") != record_type:
                record["type"] = record_type
                updates["type"] = (row, record_type)
            if ref_payment_id is not None and record.get("ref_payment_id") != ref_payment_id:
                record["ref_payment_id"] = ref_payment_id
                updates["ref_payment_id"] = (row, ref_payment_id)
            if updates:
                record["updated_at"] = now
                updates["updated_at"] = (row, record["updated_at"])
                await _update_cells(state, updates)
            return
        row_index = state.next_row
        record = {
            "provider": provider,
            "ext_id": ext_id,
            "user_id": user_id,
            "amount_cp": amount_cp,
            "items": items,
            "status": status or "pending",
            "payload": payload or "",
            "metadata": metadata_str,
            "created_at": now,
            "updated_at": now,
            "idempotency_key": idempotency_key,
            "username": username_clean or "",
            "package_id": package_id or "",
            "purchased_credits": purchased_credits if purchased_credits is not None else items,
            "processed_at": processed_at or "",
            "type": record_type or "payment",
            "ref_payment_id": ref_payment_id or "",
        }
        await _write_row(state, row_index, record)
        state.index[key] = row_index
        state.rows[key] = record
        state.next_row = row_index + 1


async def update_payment_status_by_ext(
    provider: str,
    ext_id: str,
    status: str,
    *,
    metadata: Optional[str] = None,
    payload: Optional[str] = None,
    idempotency_key: Optional[str] = None,
    processed_at: Optional[str] = None,
    package_id: Optional[str] = None,
    purchased_credits: Optional[int] = None,
    record_type: Optional[str] = None,
    ref_payment_id: Optional[str] = None,
) -> None:
    state = await _ensure_payments_state()
    key = (provider, ext_id)
    async with state.lock:
        record = state.rows.get(key)
        if not record:
            return
        updates: Dict[str, Tuple[int, Any]] = {}
        row = state.index[key]
        if status and record.get("status") != status:
            record["status"] = status
            updates["status"] = (row, status)
        if payload is not None and payload != record.get("payload"):
            record["payload"] = payload
            updates["payload"] = (row, payload)
        if metadata is not None and metadata != record.get("metadata"):
            record["metadata"] = metadata
            updates["metadata"] = (row, metadata)
        if idempotency_key is not None and idempotency_key != record.get("idempotency_key"):
            record["idempotency_key"] = idempotency_key
            updates["idempotency_key"] = (row, idempotency_key)
        if processed_at is not None and processed_at != record.get("processed_at"):
            record["processed_at"] = processed_at
            updates["processed_at"] = (row, processed_at)
        if package_id is not None and package_id != record.get("package_id"):
            record["package_id"] = package_id
            updates["package_id"] = (row, package_id)
        if purchased_credits is not None and purchased_credits != record.get("purchased_credits"):
            record["purchased_credits"] = purchased_credits
            updates["purchased_credits"] = (row, purchased_credits)
        if record_type is not None and record_type != record.get("type"):
            record["type"] = record_type
            updates["type"] = (row, record_type)
        if ref_payment_id is not None and ref_payment_id != record.get("ref_payment_id"):
            record["ref_payment_id"] = ref_payment_id
            updates["ref_payment_id"] = (row, ref_payment_id)
        if updates:
            record["updated_at"] = _now()
            updates["updated_at"] = (row, record["updated_at"])
            await _update_cells(state, updates)


async def get_payment_by_ext(provider: str, ext_id: str) -> Optional[Dict[str, Any]]:
    state = await _ensure_payments_state()
    key = (provider, ext_id)
    async with state.lock:
        record = state.rows.get(key)
        if not record:
            return None
        return dict(record)


async def list_payments_by_status(
    *,
    provider: str,
    statuses: Iterable[str],
    limit: int = 50,
) -> List[Dict[str, Any]]:
    state = await _ensure_payments_state()
    statuses_set = {status for status in statuses}
    if not statuses_set:
        return []
    results: List[Dict[str, Any]] = []
    async with state.lock:
        for record in state.rows.values():
            if record.get("provider") != provider:
                continue
            if record.get("status") not in statuses_set:
                continue
            results.append(dict(record))
    results.sort(key=lambda item: item.get("created_at", ""))
    return results[:limit]


async def get_payment_by_order_id(provider: str, order_id: str) -> Optional[Dict[str, Any]]:
    state = await _ensure_payments_state()
    async with state.lock:
        for record in state.rows.values():
            if record.get("provider") != provider:
                continue
            metadata = record.get("metadata") or ""
            try:
                parsed = json.loads(metadata) if metadata else {}
            except json.JSONDecodeError:
                continue
            if parsed.get("idemp") == order_id or parsed.get("order_id") == order_id:
                return dict(record)
    return None


async def list_payments(provider: Optional[str] = None) -> List[Dict[str, Any]]:
    state = await _ensure_payments_state()
    async with state.lock:
        records = [
            dict(record)
            for record in state.rows.values()
            if provider is None or record.get("provider") == provider
        ]
    records.sort(key=lambda item: item.get("created_at", ""))
    return records


# ---------------------------------------------------------------------------
# Jobs
# ---------------------------------------------------------------------------


async def create_job(
    job_id: str,
    user_id: int,
    prompt: str,
    image_file_id: Optional[str],
    size: str,
    seconds: int,
    model: str,
    cost_credits: int,
    corr_id: Optional[str],
    username: Optional[str] = None,
    content_type: str = "video",
) -> None:
    state = await _ensure_jobs_state()
    async with state.lock:
        if job_id in state.rows:
            return
        now = _now()
        row_index = state.next_row
        username_clean = _clean_username(username) if username is not None else ""
        record = {
            "job_id": job_id,
            "user_id": user_id,
            "prompt": prompt,
            "image_file_id": image_file_id or "",
            "sora_req_id": "",
            "status": "queued",
            "video_url": "",
            "error": "",
            "created_at": now,
            "updated_at": now,
            "size": size,
            "seconds": seconds,
            "model": model,
            "cost_credits": cost_credits,
            "username": username_clean,
            "corr_id": corr_id or "",
            "content_type": (content_type or "video"),
            "status_message_id": "",
            "status_message_index": 0,
            "status_message_updated_at": "",
        }
        await _write_row(state, row_index, record)
        state.index[job_id] = row_index
        state.rows[job_id] = record
        state.next_row = row_index + 1


async def update_job_status(job_id: str, status: str, **fields: Any) -> None:
    state = await _ensure_jobs_state()
    async with state.lock:
        record = state.rows.get(job_id)
        if not record:
            return
        row = state.index[job_id]
        updates: Dict[str, Tuple[int, Any]] = {}
        if status:
            sanitised_status = _sanitise_cell_value(status, column="status")
            record["status"] = sanitised_status
            updates["status"] = (row, sanitised_status)
        for key, value in fields.items():
            if key not in state.headers:
                continue
            sanitised_value = _sanitise_cell_value(value, column=key)
            record[key] = sanitised_value
            updates[key] = (row, sanitised_value)
        record["updated_at"] = _now()
        updates["updated_at"] = (row, record["updated_at"])
        await _update_cells(state, updates)


async def get_job(job_id: str) -> Optional[Dict[str, Any]]:
    state = await _ensure_jobs_state()
    async with state.lock:
        record = state.rows.get(job_id)
        if not record:
            return None
        return dict(record)


async def list_jobs_by_status(statuses: Iterable[str]) -> List[Dict[str, Any]]:
    state = await _ensure_jobs_state()
    statuses_set = set(statuses)
    async with state.lock:
        results = [dict(record) for record in state.rows.values() if record.get("status") in statuses_set]
    results.sort(key=lambda item: item.get("created_at", ""))
    return results


async def count_jobs_by_status(user_id: int, statuses: Iterable[str]) -> int:
    state = await _ensure_jobs_state()
    statuses_set = set(statuses)
    async with state.lock:
        count = 0
        for record in state.rows.values():
            if record.get("user_id") == user_id and record.get("status") in statuses_set:
                count += 1
        return count


async def append_error_record(**record: Any) -> None:
    state = await _ensure_errors_state()
    async with state.lock:
        row_index = state.next_row
        payload: Dict[str, Any] = {header: "" for header in state.headers}
        payload.update({k: v for k, v in record.items() if k in payload})
        payload["ts"] = payload.get("ts") or _now()
        refunded_value = payload.get("refunded")
        if isinstance(refunded_value, bool):
            payload["refunded"] = "TRUE" if refunded_value else "FALSE"
        for flag in ("preflight_blocked", "auto_sanitized"):
            flag_value = payload.get(flag)
            if isinstance(flag_value, bool):
                payload[flag] = "TRUE" if flag_value else "FALSE"
        username_raw = payload.get("username")
        if username_raw is not None:
            payload["username"] = _clean_username(username_raw)
        await _write_row(state, row_index, payload)
        key = _build_key(state.key_columns, payload)
        state.index[key] = row_index
        state.rows[key] = payload
        state.next_row = row_index + 1


async def append_archive_log(**record: Any) -> None:
    worksheet = await _ensure_archive_sheet()
    payload: Dict[str, Any] = {header: "" for header in _ARCHIVE_HEADERS}
    for key, value in record.items():
        if key in payload:
            payload[key] = value
    if not payload.get("ts"):
        payload["ts"] = _now()
    corr_id = str(payload.get("corr_id") or "").strip()
    status = str(payload.get("archive_status") or "").strip().lower()
    values = [payload.get(header, "") for header in _ARCHIVE_HEADERS]
    async with _ARCHIVE_LOCK:
        await _to_thread(_worksheet_append, worksheet, values)
        if status == "sent" and corr_id:
            _ARCHIVE_SENT.add(corr_id)


async def archive_was_sent(corr_id: str) -> bool:
    if not corr_id:
        return False
    await _ensure_archive_sheet()
    async with _ARCHIVE_LOCK:
        return corr_id in _ARCHIVE_SENT


__all__ = [
    "init",
    "get_or_create_user",
    "upsert_user_profile",
    "add_credits",
    "set_user_credits",
    "mark_bonus_granted",
    "mark_economy_v2",
    "list_users",
    "create_payment_record",
    "update_payment_status_by_ext",
    "get_payment_by_ext",
    "list_payments",
    "list_payments_by_status",
    "get_payment_by_order_id",
    "backfill_user_profiles",
    "create_job",
    "update_job_status",
    "get_job",
    "list_jobs_by_status",
    "count_jobs_by_status",
    "append_error_record",
    "append_archive_log",
    "archive_was_sent",
    "list_worksheet_titles",
]
