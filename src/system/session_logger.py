"""
SQLite Session & Event Analytics Logger for Smart Posture and Ergonomics Tracker.

Privacy-first local persistence:
- Logs session metadata, duration, upright ratio, and discrete posture events.
- Strictly stores numeric and textual metadata in RAM and SQLite.
- NEVER saves webcam video, photos, or biometric face meshes to persistent storage.
"""
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import sqlite3
from typing import Any, Dict, Generator, List, Optional
import uuid

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data",
    "posture_tracker.db",
)


class SessionLogger:
    """
    Manages SQLite database creation, session tracking, and ergonomic event logging.
    """

    def __init__(self, db_path: Optional[str] = None) -> None:
        self.db_path = db_path or DEFAULT_DB_PATH
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_tables()

    @contextmanager
    def _get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        try:
            with conn:
                yield conn
        finally:
            conn.close()


    def _init_tables(self) -> None:
        """Create sessions and events relational tables if not present."""
        with self._get_connection() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    start_time TEXT NOT NULL,
                    end_time TEXT,
                    duration_sec REAL DEFAULT 0.0,
                    good_frames INTEGER DEFAULT 0,
                    total_frames INTEGER DEFAULT 0,
                    upright_ratio REAL DEFAULT 1.0,
                    warning_count INTEGER DEFAULT 0,
                    notes TEXT DEFAULT ''
                );

                CREATE TABLE IF NOT EXISTS posture_events (
                    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    issue_description TEXT DEFAULT '',
                    metrics_json TEXT DEFAULT '{}',
                    FOREIGN KEY (session_id) REFERENCES sessions (session_id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS calibrations (
                    calibration_id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    profile_json TEXT NOT NULL
                );
            """)
        logger.debug("SQLite database initialized at %s", self.db_path)

    def start_session(self, session_id: Optional[str] = None) -> str:
        """Record the start of a monitoring session."""
        sid = session_id or str(uuid.uuid4())
        now_iso = datetime.now(timezone.utc).isoformat()
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO sessions (session_id, start_time)
                VALUES (?, ?)
                """,
                (sid, now_iso),
            )
        logger.info("Session started and logged: %s", sid)
        return sid

    def log_event(
        self,
        session_id: str,
        event_type: str,
        issue_description: str = "",
        metrics: Optional[Dict[str, Any]] = None,
    ) -> int:
        """
        Record a discrete posture event (WARNING, OVERLAY, CORRECTION, CALIBRATION).
        """
        now_iso = datetime.now(timezone.utc).isoformat()
        metrics_str = json.dumps(metrics or {})
        with self._get_connection() as conn:
            cur = conn.execute(
                """
                INSERT INTO posture_events (session_id, timestamp, event_type, issue_description, metrics_json)
                VALUES (?, ?, ?, ?, ?)
                """,
                (session_id, now_iso, event_type, issue_description, metrics_str),
            )
            event_id = cur.lastrowid
        logger.debug("Logged event #%s: %s (%s)", event_id, event_type, issue_description)
        return event_id or 0

    def end_session(
        self,
        session_id: str,
        good_frames: int,
        total_frames: int,
        warning_count: int = 0,
        notes: str = "",
    ) -> Optional[Dict[str, Any]]:
        """
        Finalize session with total duration, frame counts, and upright percentage.
        """
        now = datetime.now(timezone.utc)
        now_iso = now.isoformat()

        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT start_time FROM sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            if not row:
                logger.warning("Session %s not found for end_session.", session_id)
                return None

            start_dt = datetime.fromisoformat(row["start_time"])
            duration_sec = max(0.0, (now - start_dt).total_seconds())
            upright_ratio = (good_frames / total_frames) if total_frames > 0 else 1.0

            conn.execute(
                """
                UPDATE sessions
                SET end_time = ?,
                    duration_sec = ?,
                    good_frames = ?,
                    total_frames = ?,
                    upright_ratio = ?,
                    warning_count = ?,
                    notes = ?
                WHERE session_id = ?
                """,
                (
                    now_iso,
                    duration_sec,
                    good_frames,
                    total_frames,
                    round(upright_ratio, 4),
                    warning_count,
                    notes,
                    session_id,
                ),
            )

        logger.info("Session %s ended (duration: %.1fs, upright: %.1f%%)", session_id, duration_sec, upright_ratio * 100)
        return {
            "session_id": session_id,
            "duration_sec": duration_sec,
            "good_frames": good_frames,
            "total_frames": total_frames,
            "upright_ratio": upright_ratio,
            "warning_count": warning_count,
        }

    def get_recent_sessions(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Retrieve recent completed monitoring sessions."""
        with self._get_connection() as conn:
            rows = conn.execute(
                """
                SELECT session_id, start_time, end_time, duration_sec,
                       good_frames, total_frames, upright_ratio, warning_count
                FROM sessions
                WHERE end_time IS NOT NULL
                ORDER BY start_time DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
            return [dict(r) for r in rows]

    def get_today_summary(self) -> Dict[str, Any]:
        """Aggregate total tracked duration and upright percentage for today."""
        today_prefix = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        with self._get_connection() as conn:
            row = conn.execute(
                """
                SELECT COUNT(*) as session_count,
                       COALESCE(SUM(duration_sec), 0.0) as total_duration_sec,
                       COALESCE(SUM(good_frames), 0) as total_good_frames,
                       COALESCE(SUM(total_frames), 0) as total_frames,
                       COALESCE(SUM(warning_count), 0) as total_warnings
                FROM sessions
                WHERE start_time LIKE ? AND end_time IS NOT NULL
                """,
                (f"{today_prefix}%",),
            ).fetchone()

            tot_frames = row["total_frames"]
            good_frames = row["total_good_frames"]
            avg_score = (good_frames / tot_frames * 100) if tot_frames > 0 else 100.0

            return {
                "session_count": row["session_count"],
                "total_duration_sec": row["total_duration_sec"],
                "upright_percentage": round(avg_score, 1),
                "total_warnings": row["total_warnings"],
            }

    def save_calibration(
        self, profile_data: Dict[str, Any], calibration_id: Optional[str] = None
    ) -> str:
        """Persist a calibration profile JSON."""
        cid = calibration_id or str(uuid.uuid4())
        now_iso = datetime.now(timezone.utc).isoformat()
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO calibrations (calibration_id, timestamp, profile_json)
                VALUES (?, ?, ?)
                """,
                (cid, now_iso, json.dumps(profile_data)),
            )
        logger.info("Saved calibration profile: %s", cid)
        return cid

    def get_latest_calibration(self) -> Optional[Dict[str, Any]]:
        """Retrieve the most recent calibration profile if available."""
        with self._get_connection() as conn:
            row = conn.execute(
                """
                SELECT calibration_id, timestamp, profile_json
                FROM calibrations
                ORDER BY timestamp DESC
                LIMIT 1
                """
            ).fetchone()
            if row:
                return json.loads(row["profile_json"])
            return None

    def verify_privacy_compliance(self) -> bool:
        """
        Audit the database schema to ensure zero BLOB or image storage columns exist.
        Returns True if database strictly maintains metadata-only privacy.
        """
        with self._get_connection() as conn:
            tables = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
            for t in tables:
                cols = conn.execute(f"PRAGMA table_info({t['name']})").fetchall()
                for c in cols:
                    col_type = str(c["type"]).upper()
                    if "BLOB" in col_type or "IMAGE" in col_type or "VIDEO" in col_type:
                        return False
        return True

