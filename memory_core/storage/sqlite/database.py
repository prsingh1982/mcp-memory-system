"""SQLite connection and schema bootstrap helpers."""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from .schema import SCHEMA_STATEMENTS


class SQLiteDatabase:
    """Creates connections and ensures the local schema exists."""

    def __init__(self, db_path: str | Path, initialize: bool = True) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        if initialize:
            self.initialize()

    def initialize(self) -> None:
        """Create the SQLite schema if it does not already exist."""
        with self.connection() as conn:
            for statement in SCHEMA_STATEMENTS:
                conn.execute(statement)

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        """Yield a configured SQLite connection with commit/rollback handling."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
