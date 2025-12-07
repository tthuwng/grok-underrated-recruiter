"""
SQLite database for Grok Underrated Recruiter.
Stores saved candidates and DM history.
"""

import json
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
                handle TEXT NOT NULL,
                user_id TEXT NOT NULL,
                saved_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                notes TEXT,
                tags TEXT,
                UNIQUE(handle, user_id)
            );

            CREATE INDEX IF NOT EXISTS idx_saved_handle ON saved_candidates(handle);
            CREATE INDEX IF NOT EXISTS idx_saved_at ON saved_candidates(saved_at);
            CREATE INDEX IF NOT EXISTS idx_saved_user ON saved_candidates(user_id);

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

            -- Graph nodes table
            CREATE TABLE IF NOT EXISTS graph_nodes (
                id TEXT PRIMARY KEY,
                handle TEXT UNIQUE NOT NULL,
                name TEXT,
                bio TEXT,
                followers_count INTEGER DEFAULT 0,
                following_count INTEGER DEFAULT 0,
                is_seed INTEGER DEFAULT 0,
                is_candidate INTEGER DEFAULT 0,
                discovered_via TEXT,
                depth INTEGER DEFAULT 0,
                pagerank_score REAL DEFAULT 0.0,
                underratedness_score REAL DEFAULT 0.0,
                grok_relevant INTEGER,
                grok_role TEXT,
                grok_evaluated INTEGER DEFAULT 0,
                final_score REAL,
                deep_eval_data TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE INDEX IF NOT EXISTS idx_graph_node_handle ON graph_nodes(handle);
            CREATE INDEX IF NOT EXISTS idx_graph_node_seed ON graph_nodes(is_seed);
            CREATE INDEX IF NOT EXISTS idx_graph_node_pagerank ON graph_nodes(pagerank_score);

            -- Graph edges table
            CREATE TABLE IF NOT EXISTS graph_edges (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_id TEXT NOT NULL,
                target_id TEXT NOT NULL,
                weight REAL DEFAULT 1.0,
                interaction_type TEXT DEFAULT 'follow',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(source_id, target_id),
                FOREIGN KEY (source_id) REFERENCES graph_nodes(id),
                FOREIGN KEY (target_id) REFERENCES graph_nodes(id)
            );

            CREATE INDEX IF NOT EXISTS idx_graph_edge_source ON graph_edges(source_id);
            CREATE INDEX IF NOT EXISTS idx_graph_edge_target ON graph_edges(target_id);
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


def migrate_db():
    """Run database migrations for schema changes."""
    with get_db() as conn:
        # Check if user_id column exists in saved_candidates
        cursor = conn.execute("PRAGMA table_info(saved_candidates)")
        columns = [row[1] for row in cursor.fetchall()]

        if "user_id" not in columns:
            # Add user_id column with default value for existing rows
            conn.execute(
                "ALTER TABLE saved_candidates ADD COLUMN user_id TEXT DEFAULT 'legacy'"
            )
            # Create index for user_id
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_saved_user ON saved_candidates(user_id)"
            )

        # Check if deep_eval_data column exists in graph_nodes
        cursor = conn.execute("PRAGMA table_info(graph_nodes)")
        graph_columns = [row[1] for row in cursor.fetchall()]

        if "deep_eval_data" not in graph_columns:
            # Add deep_eval_data column to store full evaluation JSON
            conn.execute(
                "ALTER TABLE graph_nodes ADD COLUMN deep_eval_data TEXT"
            )


# --- Saved Candidates Operations ---


def save_candidate(handle: str, user_id: str, notes: Optional[str] = None) -> dict:
    """Save a candidate to the database for a specific user."""
    handle = handle.lower().lstrip("@")
    user_id = user_id.lower()
    saved_at = datetime.utcnow().isoformat()

    with get_db() as conn:
        conn.execute(
            "INSERT INTO saved_candidates (handle, user_id, saved_at, notes) VALUES (?, ?, ?, ?)",
            (handle, user_id, saved_at, notes),
        )

    return {"handle": handle, "user_id": user_id, "saved_at": saved_at, "notes": notes}


def unsave_candidate(handle: str, user_id: str) -> bool:
    """Remove a candidate from a user's saved list. Returns True if deleted."""
    handle = handle.lower().lstrip("@")
    user_id = user_id.lower()

    with get_db() as conn:
        result = conn.execute(
            "DELETE FROM saved_candidates WHERE handle = ? AND user_id = ?",
            (handle, user_id),
        )
        return result.rowcount > 0


def get_saved_handles(user_id: str) -> List[str]:
    """Get list of all saved candidate handles for a user."""
    user_id = user_id.lower()
    with get_db() as conn:
        rows = conn.execute(
            "SELECT handle FROM saved_candidates WHERE user_id = ? ORDER BY saved_at DESC",
            (user_id,),
        ).fetchall()

    return [row["handle"] for row in rows]


def get_saved_candidates(user_id: str) -> List[dict]:
    """Get all saved candidates with metadata for a user."""
    user_id = user_id.lower()
    with get_db() as conn:
        rows = conn.execute(
            "SELECT handle, saved_at, notes, tags FROM saved_candidates WHERE user_id = ? ORDER BY saved_at DESC",
            (user_id,),
        ).fetchall()

    return [dict(row) for row in rows]


def is_candidate_saved(handle: str, user_id: str) -> bool:
    """Check if a candidate is saved by a specific user."""
    handle = handle.lower().lstrip("@")
    user_id = user_id.lower()

    with get_db() as conn:
        row = conn.execute(
            "SELECT 1 FROM saved_candidates WHERE handle = ? AND user_id = ?",
            (handle, user_id),
        ).fetchone()

    return row is not None


def get_saved_count(user_id: str) -> int:
    """Get count of saved candidates for a user."""
    user_id = user_id.lower()
    with get_db() as conn:
        count = conn.execute(
            "SELECT COUNT(*) FROM saved_candidates WHERE user_id = ?",
            (user_id,),
        ).fetchone()[0]

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


def update_submission_stage(submission_id: str, stage: str) -> bool:
    """Update just the stage of a submission."""
    with get_db() as conn:
        result = conn.execute(
            "UPDATE submissions SET stage = ? WHERE id = ?",
            (stage, submission_id)
        )
        return result.rowcount > 0


def delete_submission(submission_id: str) -> bool:
    """Delete a submission by ID."""
    with get_db() as conn:
        result = conn.execute(
            "DELETE FROM submissions WHERE id = ?",
            (submission_id,)
        )
        return result.rowcount > 0


def delete_failed_submissions() -> int:
    """Delete all failed submissions. Returns count of deleted rows."""
    with get_db() as conn:
        result = conn.execute(
            "DELETE FROM submissions WHERE status = 'failed'"
        )
        return result.rowcount


def get_all_submissions() -> List[dict]:
    """Get all submissions."""
    with get_db() as conn:
        rows = conn.execute(
            """SELECT id, handle, submitted_by, submitted_at, stage, status,
                      approval_status, error_message
               FROM submissions
               ORDER BY submitted_at DESC"""
        ).fetchall()

    return [dict(row) for row in rows]


# --- Graph Operations ---


def upsert_graph_node(node_data: dict) -> dict:
    """Insert or update a graph node."""
    now = datetime.utcnow().isoformat()

    with get_db() as conn:
        conn.execute(
            """INSERT INTO graph_nodes (
                id, handle, name, bio, followers_count, following_count,
                is_seed, is_candidate, discovered_via, depth,
                pagerank_score, underratedness_score, grok_relevant, grok_role,
                grok_evaluated, final_score, deep_eval_data, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                handle = excluded.handle,
                name = excluded.name,
                bio = excluded.bio,
                followers_count = excluded.followers_count,
                following_count = excluded.following_count,
                is_seed = excluded.is_seed,
                is_candidate = excluded.is_candidate,
                discovered_via = excluded.discovered_via,
                depth = excluded.depth,
                pagerank_score = excluded.pagerank_score,
                underratedness_score = excluded.underratedness_score,
                grok_relevant = excluded.grok_relevant,
                grok_role = excluded.grok_role,
                grok_evaluated = excluded.grok_evaluated,
                final_score = excluded.final_score,
                deep_eval_data = excluded.deep_eval_data,
                updated_at = excluded.updated_at
            """,
            (
                node_data.get("id"),
                node_data.get("handle", "").lower(),
                node_data.get("name"),
                node_data.get("bio"),
                node_data.get("followers_count", 0),
                node_data.get("following_count", 0),
                1 if node_data.get("is_seed") else 0,
                1 if node_data.get("is_candidate") else 0,
                node_data.get("discovered_via"),
                node_data.get("depth", 0),
                node_data.get("pagerank_score", 0.0),
                node_data.get("underratedness_score", 0.0),
                1 if node_data.get("grok_relevant") else (0 if node_data.get("grok_relevant") is False else None),
                node_data.get("grok_role"),
                1 if node_data.get("grok_evaluated") else 0,
                node_data.get("final_score"),
                node_data.get("deep_eval_data"),
                now,
                now,
            ),
        )

    return node_data


def get_graph_node(node_id: str) -> Optional[dict]:
    """Get a graph node by ID."""
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM graph_nodes WHERE id = ?", (node_id,)
        ).fetchone()

    if not row:
        return None

    return _row_to_node_dict(row)


def get_graph_node_by_handle(handle: str) -> Optional[dict]:
    """Get a graph node by handle."""
    handle = handle.lower().lstrip("@")
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM graph_nodes WHERE handle = ?", (handle,)
        ).fetchone()

    if not row:
        return None

    return _row_to_node_dict(row)


def _row_to_node_dict(row) -> dict:
    """Convert a database row to a node dict."""
    # Parse deep_eval_data JSON if present
    deep_eval_data = None
    if row["deep_eval_data"]:
        try:
            deep_eval_data = json.loads(row["deep_eval_data"])
        except (json.JSONDecodeError, TypeError):
            pass

    return {
        "id": row["id"],
        "handle": row["handle"],
        "name": row["name"],
        "bio": row["bio"],
        "followers_count": row["followers_count"],
        "following_count": row["following_count"],
        "is_seed": bool(row["is_seed"]),
        "is_candidate": bool(row["is_candidate"]),
        "discovered_via": row["discovered_via"],
        "depth": row["depth"],
        "pagerank_score": row["pagerank_score"] or 0.0,
        "underratedness_score": row["underratedness_score"] or 0.0,
        "grok_relevant": None if row["grok_relevant"] is None else bool(row["grok_relevant"]),
        "grok_role": row["grok_role"],
        "grok_evaluated": bool(row["grok_evaluated"]),
        "final_score": row["final_score"],
        "deep_eval_data": deep_eval_data,
    }


def get_all_graph_nodes(limit: int = None) -> List[dict]:
    """Get all graph nodes. If limit is None, returns all nodes."""
    with get_db() as conn:
        if limit:
            rows = conn.execute(
                "SELECT * FROM graph_nodes ORDER BY pagerank_score DESC LIMIT ?",
                (limit,)
            ).fetchall()
        else:
            # Load all nodes - important for graph visualization
            rows = conn.execute(
                "SELECT * FROM graph_nodes ORDER BY pagerank_score DESC"
            ).fetchall()

    return [_row_to_node_dict(row) for row in rows]


def get_seed_nodes() -> List[dict]:
    """Get all seed nodes."""
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM graph_nodes WHERE is_seed = 1"
        ).fetchall()

    return [_row_to_node_dict(row) for row in rows]


def update_node_pagerank(node_id: str, pagerank_score: float) -> bool:
    """Update PageRank score for a node."""
    with get_db() as conn:
        result = conn.execute(
            "UPDATE graph_nodes SET pagerank_score = ?, updated_at = ? WHERE id = ?",
            (pagerank_score, datetime.utcnow().isoformat(), node_id)
        )
        return result.rowcount > 0


def update_node_grok_result(
    node_id: str,
    grok_relevant: bool,
    grok_role: Optional[str] = None,
    final_score: Optional[float] = None,
) -> bool:
    """Update Grok evaluation results for a node."""
    now = datetime.utcnow().isoformat()
    with get_db() as conn:
        result = conn.execute(
            """UPDATE graph_nodes
               SET grok_relevant = ?, grok_role = ?, grok_evaluated = 1,
                   final_score = ?, updated_at = ?
               WHERE id = ?""",
            (1 if grok_relevant else 0, grok_role, final_score, now, node_id)
        )
        return result.rowcount > 0


def add_graph_edge(source_id: str, target_id: str, weight: float = 1.0, interaction_type: str = "follow") -> bool:
    """Add an edge between two nodes."""
    try:
        with get_db() as conn:
            conn.execute(
                """INSERT INTO graph_edges (source_id, target_id, weight, interaction_type)
                   VALUES (?, ?, ?, ?)
                   ON CONFLICT(source_id, target_id) DO UPDATE SET
                       weight = excluded.weight,
                       interaction_type = excluded.interaction_type""",
                (source_id, target_id, weight, interaction_type)
            )
        return True
    except Exception:
        return False


def get_all_graph_edges() -> List[dict]:
    """Get all graph edges."""
    with get_db() as conn:
        rows = conn.execute(
            "SELECT source_id, target_id, weight, interaction_type FROM graph_edges"
        ).fetchall()

    return [
        {
            "source": row["source_id"],
            "target": row["target_id"],
            "weight": row["weight"],
            "interaction_type": row["interaction_type"],
        }
        for row in rows
    ]


def get_graph_stats() -> dict:
    """Get graph statistics."""
    with get_db() as conn:
        node_count = conn.execute("SELECT COUNT(*) FROM graph_nodes").fetchone()[0]
        edge_count = conn.execute("SELECT COUNT(*) FROM graph_edges").fetchone()[0]
        seed_count = conn.execute("SELECT COUNT(*) FROM graph_nodes WHERE is_seed = 1").fetchone()[0]
        candidate_count = conn.execute("SELECT COUNT(*) FROM graph_nodes WHERE is_candidate = 1").fetchone()[0]
        filtered_count = conn.execute("SELECT COUNT(*) FROM graph_nodes WHERE grok_relevant = 1").fetchone()[0]
        evaluated_count = conn.execute("SELECT COUNT(*) FROM graph_nodes WHERE grok_evaluated = 1").fetchone()[0]

    return {
        "total_nodes": node_count,
        "total_edges": edge_count,
        "seeds": seed_count,
        "candidates": candidate_count,
        "filtered_count": filtered_count,
        "evaluated_count": evaluated_count,
    }


def delete_graph_node(node_id: str) -> bool:
    """Delete a node and its edges."""
    with get_db() as conn:
        conn.execute("DELETE FROM graph_edges WHERE source_id = ? OR target_id = ?", (node_id, node_id))
        result = conn.execute("DELETE FROM graph_nodes WHERE id = ?", (node_id,))
        return result.rowcount > 0


def get_evaluated_candidates() -> list:
    """Get all candidates that have been evaluated (grok_evaluated=True or final_score set)."""
    with get_db() as conn:
        rows = conn.execute(
            """SELECT * FROM graph_nodes
               WHERE grok_evaluated = 1 OR final_score IS NOT NULL
               ORDER BY final_score DESC"""
        ).fetchall()

    results = []
    for row in rows:
        # Parse deep_eval_data JSON if present
        deep_eval_data = None
        if row["deep_eval_data"]:
            try:
                deep_eval_data = json.loads(row["deep_eval_data"])
            except (json.JSONDecodeError, TypeError):
                pass

        candidate = {
            "handle": row["handle"],
            "name": row["name"],
            "bio": row["bio"],
            "followers_count": row["followers_count"],
            "following_count": row["following_count"],
            "pagerank_score": row["pagerank_score"],
            "underratedness_score": row["underratedness_score"],
            "final_score": row["final_score"],
            "grok_role": row["grok_role"],
            "grok_relevant": row["grok_relevant"],
            "recommended_role": row["grok_role"],  # Alias for compatibility
        }

        # Merge deep_eval_data fields if available
        if deep_eval_data:
            candidate["summary"] = deep_eval_data.get("summary")
            candidate["strengths"] = deep_eval_data.get("strengths")
            candidate["concerns"] = deep_eval_data.get("concerns")
            candidate["recommended_role"] = deep_eval_data.get("recommended_role", row["grok_role"])
            candidate["technical_depth"] = deep_eval_data.get("technical_depth")
            candidate["project_evidence"] = deep_eval_data.get("project_evidence")
            candidate["mission_alignment"] = deep_eval_data.get("mission_alignment")
            candidate["exceptional_ability"] = deep_eval_data.get("exceptional_ability")
            candidate["communication"] = deep_eval_data.get("communication")
            candidate["github_url"] = deep_eval_data.get("github_url")
            candidate["linkedin_url"] = deep_eval_data.get("linkedin_url")
            candidate["top_repos"] = deep_eval_data.get("top_repos")

        results.append(candidate)

    return results


def get_relevant_candidates() -> list:
    """Get all candidates that passed Grok relevance filter."""
    with get_db() as conn:
        rows = conn.execute(
            """SELECT * FROM graph_nodes
               WHERE grok_relevant = 1
               ORDER BY pagerank_score DESC"""
        ).fetchall()

    return [
        {
            "handle": row["handle"],
            "name": row["name"],
            "bio": row["bio"],
            "followers_count": row["followers_count"],
            "following_count": row["following_count"],
            "pagerank_score": row["pagerank_score"],
            "underratedness_score": row["underratedness_score"],
            "final_score": row["final_score"],
            "grok_role": row["grok_role"],
            "grok_relevant": row["grok_relevant"],
            "recommended_role": row["grok_role"],
        }
        for row in rows
    ]
