"""Helpers for referral tracking in Google Sheets."""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional

import gspread

try:
    from gsheets_db import _service_client  # type: ignore
except Exception:  # pragma: no cover - configuration error during import
    _service_client = None  # type: ignore


log = logging.getLogger(__name__)

SHEET_ID = "1V_IzjMpgV_u5S5vmLpV70-9sf-wilVt1xDr-fuJQ95I"
SHEET_NAME = "referrals"
HEADERS = [
    "ref_code",
    "referrer_user_id",
    "new_user_id",
    "created_at_iso",
    "credited",
    "credited_at_iso",
    "topup_amount",
    "topup_currency",
    "payment_id",
    "note",
]

_WORKSHEET: Optional[gspread.Worksheet] = None
_WORKSHEET_LOCK = asyncio.Lock()


async def get_ws_referrals() -> Optional[gspread.Worksheet]:
    """Return the referrals worksheet, creating it when necessary."""

    async with _WORKSHEET_LOCK:
        if _WORKSHEET is not None:
            return _WORKSHEET
        if _service_client is None:
            log.error("Google Sheets client is not available for referrals")
            return None
        try:
            client = _service_client()
        except Exception:
            log.exception("Failed to initialise Google Sheets client for referrals")
            return None
        try:
            spreadsheet = await asyncio.to_thread(client.open_by_key, SHEET_ID)
        except Exception:
            log.exception("Failed to open Google Sheet %s for referrals", SHEET_ID)
            return None
        try:
            worksheet = await asyncio.to_thread(spreadsheet.worksheet, SHEET_NAME)
        except gspread.exceptions.WorksheetNotFound:
            try:
                worksheet = await asyncio.to_thread(
                    spreadsheet.add_worksheet,
                    title=SHEET_NAME,
                    rows="1000",
                    cols=str(len(HEADERS) + 2),
                )
            except Exception:
                log.exception("Failed to create referrals worksheet %s", SHEET_NAME)
                return None
        except Exception:
            log.exception("Failed to fetch referrals worksheet %s", SHEET_NAME)
            return None
        try:
            await _ensure_headers(worksheet)
        except Exception:
            log.exception("Failed to ensure headers for referrals worksheet")
            return None
        globals()["_WORKSHEET"] = worksheet
        return worksheet


async def find_ref_by_new_user_id(
    new_user_id: int, *, only_pending: bool = False
) -> Optional[Dict[str, Any]]:
    """Return the referral row for *new_user_id*.

    When *only_pending* is ``True``, only rows with ``credited`` flag unset
    are considered.
    """

    worksheet = await get_ws_referrals()
    if worksheet is None:
        return None
    try:
        rows = await _fetch_rows(worksheet)
    except Exception:
        log.exception("Failed to load referrals rows for lookup")
        return None
    new_user_str = str(new_user_id)
    for row in rows:
        if str(row.get("new_user_id") or "").strip() != new_user_str:
            continue
        if only_pending and _as_bool(row.get("credited")):
            continue
        return row
    return None


async def append_ref_row(referrer_user_id: int, new_user_id: int) -> bool:
    """Append a new referral tracking row."""

    worksheet = await get_ws_referrals()
    if worksheet is None:
        return False
    created_at = datetime.utcnow().replace(microsecond=0).isoformat()
    values = [
        f"ref_{referrer_user_id}",
        str(referrer_user_id),
        str(new_user_id),
        created_at,
        False,
        "",
        "",
        "",
        "",
        "",
    ]
    try:
        await asyncio.to_thread(
            worksheet.append_row,
            values,
            value_input_option="USER_ENTERED",
        )
    except Exception:
        log.exception("Failed to append referral row referrer=%s new_user=%s", referrer_user_id, new_user_id)
        return False
    return True


async def mark_credited(
    row: Dict[str, Any], *, amount: Any, currency: Any, payment_id: Any
) -> bool:
    """Mark the provided referral *row* as credited."""

    worksheet = await get_ws_referrals()
    if worksheet is None:
        return False
    row_number = _parse_int(row.get("row_number"))
    if row_number <= 0:
        log.warning("Referral row missing row_number: %s", row)
        return False
    credited_at = datetime.utcnow().replace(microsecond=0).isoformat()
    amount_text = _stringify_amount(amount)
    currency_text = str(currency or "")
    payment_text = str(payment_id or "")
    values = [
        True,
        credited_at,
        amount_text,
        currency_text,
        payment_text,
        "bonus +100 on first topup by referral",
    ]
    range_name = f"E{row_number}:J{row_number}"
    try:
        await asyncio.to_thread(
            worksheet.update,
            range_name,
            [values],
            value_input_option="USER_ENTERED",
        )
    except Exception:
        log.exception(
            "Failed to update referral row %s as credited (payment=%s)",
            row_number,
            payment_id,
        )
        return False
    return True


async def _ensure_headers(worksheet: gspread.Worksheet) -> None:
    current = await asyncio.to_thread(worksheet.row_values, 1)
    existing = [str(value).strip() for value in current if value]
    if not existing:
        headers = HEADERS
    else:
        missing = [header for header in HEADERS if header not in existing]
        headers = existing + missing
    if headers != existing:
        range_name = f"A1:{_column_letter(len(headers))}1"
        await asyncio.to_thread(
            worksheet.update,
            range_name,
            [headers],
            value_input_option="USER_ENTERED",
        )


async def _fetch_rows(worksheet: gspread.Worksheet) -> List[Dict[str, Any]]:
    values = await asyncio.to_thread(worksheet.get_all_values)
    if not values:
        return []
    headers = [str(h).strip() for h in values[0]]
    if not headers:
        headers = HEADERS
    rows: List[Dict[str, Any]] = []
    for idx, row_values in enumerate(values[1:], start=2):
        data: Dict[str, Any] = {}
        for col_idx, header in enumerate(headers):
            data[header] = row_values[col_idx] if col_idx < len(row_values) else ""
        for header in HEADERS:
            data.setdefault(header, "")
        data["row_number"] = idx
        rows.append(data)
    return rows


def _column_letter(idx: int) -> str:
    result = ""
    while idx > 0:
        idx, remainder = divmod(idx - 1, 26)
        result = chr(65 + remainder) + result
    return result or "A"


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value or "").strip().lower()
    return text in {"true", "1", "yes", "y"}


def _parse_int(value: Any) -> int:
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return 0


def _stringify_amount(value: Any) -> str:
    if value in (None, ""):
        return ""
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, Decimal):
        return format(value, "f")
    return str(value)


__all__ = [
    "append_ref_row",
    "find_ref_by_new_user_id",
    "get_ws_referrals",
    "mark_credited",
]
