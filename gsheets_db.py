"""Google Sheets-backed data access layer for the Sora bot."""
from __future__ import annotations

import asyncio
import base64
import binascii
import json
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple

import gspread
from google.oauth2.service_account import Credentials
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from asyncio import Lock


_USERS_HEADERS = [
    "user_id",
    "credits",
    "bonus_granted",
    "created_at",
    "updated_at",
    "notes",
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


async def init() -> None:
    """Initialise the Google Sheets client and build indices."""

    await _ensure_spreadsheet()
    await _ensure_users_state()
    await _ensure_payments_state()
    await _ensure_jobs_state()


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
    worksheet.batch_update(body)


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


async def _ensure_sheet(title: str, headers: List[str]) -> gspread.Worksheet:
    spreadsheet = await _ensure_spreadsheet()
    try:
        worksheet = await _to_thread(_get_worksheet, spreadsheet, title)
    except gspread.exceptions.WorksheetNotFound:
        worksheet = await _to_thread(_add_worksheet, spreadsheet, title, 1000, len(headers) + 2)
    await _ensure_headers(worksheet, headers)
    return worksheet


async def _ensure_headers(worksheet: gspread.Worksheet, headers: List[str]) -> None:
    current = await _to_thread(worksheet.row_values, 1)
    if current[: len(headers)] == headers:
        return
    range_name = f"A1:{_column_letter(len(headers))}1"
    await _to_thread(_worksheet_update, worksheet, range_name, [headers])


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


def _normalise_user(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "user_id": _parse_int(row.get("user_id")),
        "credits": _parse_int(row.get("credits")),
        "bonus_granted": bool(_parse_int(row.get("bonus_granted"))),
        "created_at": _parse_str(row.get("created_at")),
        "updated_at": _parse_str(row.get("updated_at")),
        "notes": _parse_str(row.get("notes")),
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
        worksheet = await _ensure_sheet(title, _USERS_HEADERS)
        index, rows, next_row = await _to_thread(
            _build_state_data, worksheet, _USERS_HEADERS, ("user_id",), _normalise_user
        )
        _USERS_STATE = _SheetState(
            worksheet=worksheet,
            headers=_USERS_HEADERS,
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
        worksheet = await _ensure_sheet(title, _PAYMENTS_HEADERS)
        index, rows, next_row = await _to_thread(
            _build_state_data,
            worksheet,
            _PAYMENTS_HEADERS,
            ("provider", "ext_id"),
            _normalise_payment,
        )
        _PAYMENTS_STATE = _SheetState(
            worksheet=worksheet,
            headers=_PAYMENTS_HEADERS,
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
        worksheet = await _ensure_sheet(title, _JOBS_HEADERS)
        index, rows, next_row = await _to_thread(
            _build_state_data, worksheet, _JOBS_HEADERS, ("job_id",), _normalise_job
        )
        _JOBS_STATE = _SheetState(
            worksheet=worksheet,
            headers=_JOBS_HEADERS,
            key_columns=("job_id",),
            index=index,
            rows=rows,
            next_row=next_row,
            lock=Lock(),
        )
    return _JOBS_STATE


def _row_range(state: _SheetState, row: int) -> str:
    return f"A{row}:{_column_letter(len(state.headers))}{row}"


def _ensure_now(value: Optional[str]) -> str:
    return value or _now()


async def _write_row(state: _SheetState, row_index: int, data: Dict[str, Any]) -> None:
    values = [data.get(header, "") for header in state.headers]
    await _to_thread(_worksheet_update, state.worksheet, _row_range(state, row_index), [values])


async def _update_cells(state: _SheetState, updates: Dict[str, Tuple[int, Any]]) -> None:
    requests = []
    for column, (row, value) in updates.items():
        col_idx = state.headers.index(column) + 1
        col_letter = _column_letter(col_idx)
        requests.append({"range": f"{col_letter}{row}", "values": [[value]]})
    if requests:
        await _to_thread(_worksheet_batch_update, state.worksheet, requests)


async def _create_user_record(state: _SheetState, user_id: int) -> Dict[str, Any]:
    now = _now()
    row_index = state.next_row
    record = {
        "user_id": user_id,
        "credits": 0,
        "bonus_granted": 0,
        "created_at": now,
        "updated_at": now,
        "notes": "",
    }
    await _write_row(state, row_index, record)
    state.index[user_id] = row_index
    state.rows[user_id] = record
    state.next_row = row_index + 1
    return record


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
) -> None:
    state = await _ensure_payments_state()
    key = (provider, ext_id)
    async with state.lock:
        record = state.rows.get(key)
        now = _now()
        metadata_str = json.dumps(metadata) if isinstance(metadata, dict) else _parse_str(metadata)
        if record:
            changed = False
            if status and record.get("status") != status:
                record["status"] = status
                changed = True
            if payload and record.get("payload") != payload:
                record["payload"] = payload
                changed = True
            if metadata_str and record.get("metadata") != metadata_str:
                record["metadata"] = metadata_str
                changed = True
            if changed:
                record["updated_at"] = now
                row = state.index[key]
                await _update_cells(
                    state,
                    {
                        "status": (row, record["status"]),
                        "payload": (row, record["payload"]),
                        "metadata": (row, record["metadata"]),
                        "updated_at": (row, record["updated_at"]),
                    },
                )
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
) -> None:
    state = await _ensure_jobs_state()
    async with state.lock:
        if job_id in state.rows:
            return
        now = _now()
        row_index = state.next_row
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
            record["status"] = status
            updates["status"] = (row, status)
        for key, value in fields.items():
            if key not in state.headers:
                continue
            record[key] = value if value is not None else ""
            updates[key] = (row, record[key])
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


__all__ = [
    "init",
    "get_or_create_user",
    "add_credits",
    "mark_bonus_granted",
    "create_payment_record",
    "update_payment_status_by_ext",
    "get_payment_by_ext",
    "list_payments_by_status",
    "get_payment_by_order_id",
    "create_job",
    "update_job_status",
    "get_job",
    "list_jobs_by_status",
    "count_jobs_by_status",
]
