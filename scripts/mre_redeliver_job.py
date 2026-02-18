#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
from typing import Any, Dict, Optional

from aiogram import Bot

from config import load_config
from db import create_database
from handlers import _send_job_update
from services.error_reporter import ErrorReporter


class _NoopArchive:
    pass


def _extract_saved_asset_url(extra: Dict[str, Any]) -> Optional[str]:
    assets = extra.get("assets_meta")
    if not isinstance(assets, list):
        return None
    for entry in assets:
        if not isinstance(entry, dict):
            continue
        value = entry.get("url") or entry.get("file_id")
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


async def _run(job_id: str, force_completed: bool) -> int:
    config = load_config()
    db = create_database(config)
    await db.init()
    bot = Bot(token=config.bot_token)
    reporter = ErrorReporter(bot=bot, config=config)
    dp = type("_Dispatcher", (), {"bot": bot})()
    try:
        job = await db.get_job(job_id)
        if job is None:
            print(f"job_not_found id={job_id}")
            return 2

        extra = job.extra if isinstance(job.extra, dict) else {}
        saved_url = (job.file_url or job.video_url or "").strip()
        if not saved_url:
            saved_url = _extract_saved_asset_url(extra) or ""
            if saved_url:
                await db.update_job(job.id, job.status or "completed", file_url=saved_url)
                job.file_url = saved_url

        if not saved_url:
            print(f"no_saved_asset_url job_id={job.id}")
            return 3

        if force_completed:
            job.status = "completed"

        await _send_job_update(
            dp=dp,
            job=job,
            archive=None,
            config=config,
            db=db,
            error_reporter=reporter,
        )
        print(f"redelivery_done job_id={job.id} status={job.status} url={saved_url}")
        return 0
    finally:
        await db.close()
        await bot.session.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Re-deliver an existing completed job without enqueue")
    parser.add_argument("--job_id", required=True)
    parser.add_argument(
        "--force-completed",
        action="store_true",
        help="Temporarily mark job object as completed for delivery even if stored status differs",
    )
    args = parser.parse_args()
    return asyncio.run(_run(args.job_id, args.force_completed))


if __name__ == "__main__":
    raise SystemExit(main())
