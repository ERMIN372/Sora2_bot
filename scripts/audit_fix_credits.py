"""Audit YooKassa payments and correct excessive credit balances."""

import argparse
import asyncio
import logging
import uuid
from collections import defaultdict
from datetime import datetime
from typing import Dict, Iterable, Optional

from config import load_config
from db import DatabaseInterface, create_database

log = logging.getLogger(__name__)


def _parse_optional_int(value: object) -> Optional[int]:
    if value in (None, "", " "):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat()


async def audit_fix_credits(
    db: DatabaseInterface, *, allowed_packages: Dict[str, int], dry_run: bool
) -> None:
    payments = await db.list_payments(provider="yookassa")
    allowed_values = set(allowed_packages.values())
    totals_by_user = defaultdict(int)
    anomalies_by_user = defaultdict(int)

    for record in payments:
        status = (record.get("status") or "").lower()
        if status != "succeeded":
            continue
        user_id = _parse_optional_int(record.get("user_id")) or 0
        if not user_id:
            continue
        record_type = (record.get("type") or "payment").lower()
        purchased = _parse_optional_int(record.get("purchased_credits"))
        if purchased is None:
            purchased = _parse_optional_int(record.get("items"))
        if purchased is None:
            continue
        if record_type == "adjustment":
            totals_by_user[user_id] += purchased
            continue
        package_id = (record.get("package_id") or "").strip()
        expected = None
        if package_id and package_id in allowed_packages:
            expected = allowed_packages[package_id]
        elif purchased in allowed_values:
            expected = purchased
        if purchased > max(allowed_values):
            log.warning(
                "Audit: suspicious high credit purchase ext_id=%s user=%s credits=%s",
                record.get("ext_id"),
                user_id,
                purchased,
            )
        if expected is None:
            anomalies_by_user[user_id] += purchased
            log.warning(
                "Audit: unknown package for payment ext_id=%s user=%s credits=%s package_id=%s",
                record.get("ext_id"),
                user_id,
                purchased,
                package_id or "",
            )
            continue
        if purchased > expected:
            anomalies_by_user[user_id] += purchased - expected
            log.warning(
                "Audit: payment ext_id=%s user=%s purchased=%s expected=%s",
                record.get("ext_id"),
                user_id,
                purchased,
                expected,
            )
        totals_by_user[user_id] += expected

    if not totals_by_user:
        log.info("No succeeded YooKassa payments to audit")

    users = await db.list_users()
    adjustments = []
    for record in users:
        user_id = _parse_optional_int(record.get("user_id")) or 0
        if not user_id:
            continue
        balance = _parse_optional_int(record.get("credits")) or 0
        expected_balance = max(0, totals_by_user.get(user_id, 0))
        extra = balance - expected_balance
        if extra <= 0:
            continue
        if totals_by_user.get(user_id, 0) <= 0 and anomalies_by_user.get(user_id, 0) <= 0:
            continue
        adjustments.append((user_id, balance, expected_balance, extra))

    if not adjustments:
        log.info("No credit anomalies detected")
        return

    for user_id, balance, expected_balance, extra in adjustments:
        log.warning(
            "Audit: user %s has %s extra credits (balance=%s expected=%s)",
            user_id,
            extra,
            balance,
            expected_balance,
        )
        if dry_run:
            continue
        processed_at = _now_iso()
        try:
            new_balance = await db.add_credits(user_id, -extra)
        except ValueError:
            log.exception("Failed to apply adjustment for user %s", user_id)
            continue
        ext_id = f"adjustment-{uuid.uuid4()}"
        metadata = {
            "reason": "audit_fix_credits",
            "excess": extra,
            "balance_before": balance,
            "balance_after": new_balance,
        }
        await db.create_payment_record(
            "system",
            ext_id,
            user_id,
            0,
            -extra,
            "succeeded",
            payload="audit_fix_credits",
            metadata=metadata,
            package_id="adjustment",
            purchased_credits=-extra,
            processed_at=processed_at,
            record_type="adjustment",
            ref_payment_id="audit_fix_credits",
        )
        log.info(
            "Applied audit adjustment ext_id=%s user=%s removed=%s new_balance=%s",
            ext_id,
            user_id,
            extra,
            new_balance,
        )


async def _run(dry_run: bool) -> None:
    config = load_config()
    logging.basicConfig(level=logging.INFO)
    db = create_database(config)
    await db.init()
    try:
        allowed = {package.package_id: package.credits_int for package in config.credit_packages}
        await audit_fix_credits(db, allowed_packages=allowed, dry_run=dry_run)
    finally:
        await db.close()


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Only report anomalies without applying adjustments")
    args = parser.parse_args(argv)
    asyncio.run(_run(dry_run=args.dry_run))


if __name__ == "__main__":
    main()
