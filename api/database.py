"""
SQLite database for Grok Underrated Recruiter.
Stores saved candidates and DM history.
"""

import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import List, Optional

DATABASE_PATH = Path(__file__).parent.parent / "data" / "recruiter.db"


def init_db():
    """Initialize the SQLite database with required tables."""
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)

    with get_db() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS saved_candidates (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                handle TEXT UNIQUE NOT NULL,
                saved_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                notes TEXT,
                tags TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_saved_handle ON saved_candidates(handle);
            CREATE INDEX IF NOT EXISTS idx_saved_at ON saved_candidates(saved_at);

            CREATE TABLE IF NOT EXISTS dm_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                handle TEXT NOT NULL,
                custom_context TEXT,
                generated_message TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE INDEX IF NOT EXISTS idx_dm_handle ON dm_history(handle);
        """)


@contextmanager
def get_db():
    """Context manager for database connections."""
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


# --- Saved Candidates Operations ---


def save_candidate(handle: str, notes: Optional[str] = None) -> dict:
    """Save a candidate to the database."""
    handle = handle.lower().lstrip("@")
    saved_at = datetime.utcnow().isoformat()

    with get_db() as conn:
        conn.execute(
            "INSERT INTO saved_candidates (handle, saved_at, notes) VALUES (?, ?, ?)",
            (handle, saved_at, notes),
        )

    return {"handle": handle, "saved_at": saved_at, "notes": notes}


def unsave_candidate(handle: str) -> bool:
    """Remove a candidate from saved list. Returns True if deleted."""
    handle = handle.lower().lstrip("@")

    with get_db() as conn:
        result = conn.execute(
            "DELETE FROM saved_candidates WHERE handle = ?", (handle,)
        )
        return result.rowcount > 0


def get_saved_handles() -> List[str]:
    """Get list of all saved candidate handles."""
    with get_db() as conn:
        rows = conn.execute(
            "SELECT handle FROM saved_candidates ORDER BY saved_at DESC"
        ).fetchall()

    return [row["handle"] for row in rows]


def get_saved_candidates() -> List[dict]:
    """Get all saved candidates with metadata."""
    with get_db() as conn:
        rows = conn.execute(
            "SELECT handle, saved_at, notes, tags FROM saved_candidates ORDER BY saved_at DESC"
        ).fetchall()

    return [dict(row) for row in rows]


def is_candidate_saved(handle: str) -> bool:
    """Check if a candidate is saved."""
    handle = handle.lower().lstrip("@")

    with get_db() as conn:
        row = conn.execute(
            "SELECT 1 FROM saved_candidates WHERE handle = ?", (handle,)
        ).fetchone()

    return row is not None


def get_saved_count() -> int:
    """Get count of saved candidates."""
    with get_db() as conn:
        count = conn.execute("SELECT COUNT(*) FROM saved_candidates").fetchone()[0]

    return count


# --- DM History Operations ---


def save_dm_history(
    handle: str, custom_context: str, generated_message: str
) -> dict:
    """Save a generated DM to history."""
    handle = handle.lower().lstrip("@")
    created_at = datetime.utcnow().isoformat()

    with get_db() as conn:
        conn.execute(
            "INSERT INTO dm_history (handle, custom_context, generated_message, created_at) VALUES (?, ?, ?, ?)",
            (handle, custom_context, generated_message, created_at),
        )

    return {
        "handle": handle,
        "custom_context": custom_context,
        "generated_message": generated_message,
        "created_at": created_at,
    }


def get_dm_history(handle: Optional[str] = None, limit: int = 50) -> List[dict]:
    """Get DM history, optionally filtered by handle."""
    with get_db() as conn:
        if handle:
            handle = handle.lower().lstrip("@")
            rows = conn.execute(
                "SELECT * FROM dm_history WHERE handle = ? ORDER BY created_at DESC LIMIT ?",
                (handle, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM dm_history ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()

    return [dict(row) for row in rows]
