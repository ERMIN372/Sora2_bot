from __future__ import annotations

from typing import TYPE_CHECKING

from .dual_write import DualWriteDatabase
from .interface import DatabaseInterface
from .models import (
    ArchiveLogRecord,
    ErrorLogRecord,
    GenerationJobRecord,
    User,
)
from .postgres_adapter import PostgresDatabase
from .sheets_adapter import SheetsDatabase

if TYPE_CHECKING:
    from config import Config


def create_database(config: "Config") -> DatabaseInterface:
    """Build the database backend according to configuration."""

    if not config.postgres_enabled and not config.dual_write_enabled:
        return SheetsDatabase()

    if not config.database_url:
        raise RuntimeError("DATABASE_URL is required when PostgreSQL is enabled")

    postgres = PostgresDatabase(
        dsn=config.database_url,
        min_pool_size=config.postgres_pool_min_size,
        max_pool_size=config.postgres_pool_max_size,
    )
    sheets = SheetsDatabase()

    if config.dual_write_enabled:
        if config.dual_write_primary == "sheets":
            primary: DatabaseInterface = sheets
            secondary: DatabaseInterface = postgres
        else:
            primary = postgres
            secondary = sheets
        return DualWriteDatabase(primary, secondary)

    return postgres


# Backwards-compatible alias used in type hints
Database = DatabaseInterface

__all__ = [
    "ArchiveLogRecord",
    "Database",
    "DatabaseInterface",
    "DualWriteDatabase",
    "ErrorLogRecord",
    "GenerationJobRecord",
    "PostgresDatabase",
    "SheetsDatabase",
    "User",
    "create_database",
]
