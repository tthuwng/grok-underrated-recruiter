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

            CREATE TABLE IF NOT EXISTS submissions (
                id TEXT PRIMARY KEY,
                handle TEXT UNIQUE NOT NULL,
                submitted_by TEXT,
                submitted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                stage TEXT NOT NULL DEFAULT 'pending',
                status TEXT NOT NULL DEFAULT 'pending',
                approval_status TEXT NOT NULL DEFAULT 'pending',
                started_at TIMESTAMP,
                completed_at TIMESTAMP,
                fast_screen_result TEXT,
                deep_eval_result TEXT,
                error_message TEXT,
                priority INTEGER DEFAULT 0,
                attempts INTEGER DEFAULT 0
            );

            CREATE INDEX IF NOT EXISTS idx_submission_handle ON submissions(handle);
            CREATE INDEX IF NOT EXISTS idx_submission_status ON submissions(status);
            CREATE INDEX IF NOT EXISTS idx_submission_approval ON submissions(approval_status);
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


# --- Submissions Operations ---


def create_submission(
    submission_id: str,
    handle: str,
    submitted_by: Optional[str] = None,
) -> dict:
    """Create a new submission record."""
    handle = handle.lower().lstrip("@")
    submitted_at = datetime.utcnow().isoformat()

    with get_db() as conn:
        conn.execute(
            """INSERT INTO submissions
               (id, handle, submitted_by, submitted_at, stage, status, approval_status)
               VALUES (?, ?, ?, ?, 'pending', 'pending', 'pending')""",
            (submission_id, handle, submitted_by, submitted_at),
        )

    return {
        "id": submission_id,
        "handle": handle,
        "submitted_by": submitted_by,
        "submitted_at": submitted_at,
        "stage": "pending",
        "status": "pending",
        "approval_status": "pending",
    }


def get_submission(submission_id: str) -> Optional[dict]:
    """Get a submission by ID."""
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM submissions WHERE id = ?", (submission_id,)
        ).fetchone()

    return dict(row) if row else None


def get_submission_by_handle(handle: str) -> Optional[dict]:
    """Get a submission by handle."""
    handle = handle.lower().lstrip("@")

    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM submissions WHERE handle = ?", (handle,)
        ).fetchone()

    return dict(row) if row else None


def update_submission_status(
    submission_id: str,
    status: str,
    stage: str,
    fast_screen_result: Optional[str] = None,
    deep_eval_result: Optional[str] = None,
    error_message: Optional[str] = None,
) -> bool:
    """Update submission status and stage."""
    now = datetime.utcnow().isoformat()

    with get_db() as conn:
        # Build update query dynamically
        updates = ["status = ?", "stage = ?"]
        params = [status, stage]

        if status == "processing" and stage == "fast_screen":
            updates.append("started_at = ?")
            params.append(now)

        if status in ("completed", "filtered_out", "failed"):
            updates.append("completed_at = ?")
            params.append(now)

        if fast_screen_result:
            updates.append("fast_screen_result = ?")
            params.append(fast_screen_result)

        if deep_eval_result:
            updates.append("deep_eval_result = ?")
            params.append(deep_eval_result)

        if error_message:
            updates.append("error_message = ?")
            params.append(error_message)

        params.append(submission_id)
        result = conn.execute(
            f"UPDATE submissions SET {', '.join(updates)} WHERE id = ?",
            params,
        )
        return result.rowcount > 0


def update_submission_approval(
    submission_id: str,
    approval_status: str,
    approved_by: Optional[str] = None,
) -> bool:
    """Update submission approval status."""
    now = datetime.utcnow().isoformat()

    with get_db() as conn:
        result = conn.execute(
            """UPDATE submissions
               SET approval_status = ?, started_at = ?
               WHERE id = ?""",
            (approval_status, now if approval_status == "approved" else None, submission_id),
        )
        return result.rowcount > 0


def get_pending_approvals() -> List[dict]:
    """Get submissions pending approval."""
    with get_db() as conn:
        rows = conn.execute(
            """SELECT id, handle, submitted_by, submitted_at
               FROM submissions
               WHERE approval_status = 'pending'
               ORDER BY submitted_at ASC"""
        ).fetchall()

    return [dict(row) for row in rows]


def get_approved_pending_submissions() -> List[dict]:
    """Get approved submissions that haven't been processed yet."""
    with get_db() as conn:
        rows = conn.execute(
            """SELECT * FROM submissions
               WHERE approval_status = 'approved' AND status = 'pending'
               ORDER BY priority DESC, submitted_at ASC"""
        ).fetchall()

    return [dict(row) for row in rows]


def get_submission_queue_position(submission_id: str) -> int:
    """Get position in the processing queue."""
    with get_db() as conn:
        row = conn.execute(
            """SELECT COUNT(*) FROM submissions
               WHERE approval_status = 'approved'
               AND status = 'pending'
               AND submitted_at < (SELECT submitted_at FROM submissions WHERE id = ?)""",
            (submission_id,)
        ).fetchone()

    return row[0] + 1 if row else 0


def increment_submission_attempts(submission_id: str) -> int:
    """Increment the retry attempts counter for a submission."""
    with get_db() as conn:
        conn.execute(
            "UPDATE submissions SET attempts = attempts + 1 WHERE id = ?",
            (submission_id,)
        )
        row = conn.execute(
            "SELECT attempts FROM submissions WHERE id = ?",
            (submission_id,)
        ).fetchone()

    return row[0] if row else 0
