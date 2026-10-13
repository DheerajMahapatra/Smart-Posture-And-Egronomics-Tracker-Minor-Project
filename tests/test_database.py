"""
Unit and integration tests for SQLite session logger and privacy compliance.
"""
import os
import sqlite3
import tempfile
import pytest

from src.system.session_logger import SessionLogger


@pytest.fixture
def temp_db():
    """Create a temporary database path for test isolation."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    yield path
    try:
        if os.path.exists(path):
            os.remove(path)
    except PermissionError:
        pass


def test_database_initialization(temp_db):
    """Verify that tables are created with proper schemas upon initialization."""
    logger = SessionLogger(db_path=temp_db)
    assert os.path.exists(temp_db)

    conn = sqlite3.connect(temp_db)
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row[0] for row in cursor.fetchall()}
    finally:
        conn.close()

    assert "sessions" in tables
    assert "posture_events" in tables
    assert "calibrations" in tables



def test_session_lifecycle(temp_db):
    """Test starting, logging events, and ending a monitoring session."""
    logger = SessionLogger(db_path=temp_db)

    # 1. Start session
    session_id = logger.start_session()
    assert session_id is not None
    assert len(session_id) > 0

    # 2. Log events
    ev1 = logger.log_event(
        session_id=session_id,
        event_type="WARNING_1",
        issue_description="Slouching detected for 20s",
        metrics={"slouch_ratio": 0.58, "warning_level": 1},
    )
    assert ev1 > 0

    ev2 = logger.log_event(
        session_id=session_id,
        event_type="OVERLAY_TRIGGERED",
        issue_description="3 warnings ignored; overlay active",
    )
    assert ev2 > ev1

    # 3. End session
    summary = logger.end_session(
        session_id=session_id,
        good_frames=800,
        total_frames=1000,
        warning_count=3,
        notes="Desk work test session",
    )

    assert summary is not None
    assert summary["session_id"] == session_id
    assert summary["good_frames"] == 800
    assert summary["total_frames"] == 1000
    assert summary["upright_ratio"] == 0.8
    assert summary["warning_count"] == 3
    assert summary["duration_sec"] >= 0.0


def test_calibration_storage(temp_db):
    """Test storing and retrieving personalized calibration profiles."""
    logger = SessionLogger(db_path=temp_db)

    profile_data = {
        "ear_to_shoulder_dist": 0.22,
        "shoulder_width": 0.45,
        "eye_distance": 0.12,
        "shoulder_tilt_angle": 1.5,
        "head_forward_delta": 0.03,
        "confidence": 0.94,
        "created_at": "2026-10-13T12:00:00Z",
    }

    cid = logger.save_calibration(profile_data)
    assert cid is not None

    latest = logger.get_latest_calibration()
    assert latest is not None
    assert latest["ear_to_shoulder_dist"] == 0.22
    assert latest["shoulder_width"] == 0.45
    assert latest["confidence"] == 0.94


def test_recent_sessions_and_today_summary(temp_db):
    """Test recent session retrieval and daily aggregate statistics."""
    logger = SessionLogger(db_path=temp_db)

    # Create and finish two sessions
    s1 = logger.start_session()
    logger.end_session(s1, good_frames=90, total_frames=100, warning_count=1)

    s2 = logger.start_session()
    logger.end_session(s2, good_frames=180, total_frames=200, warning_count=2)

    recent = logger.get_recent_sessions(limit=5)
    assert len(recent) == 2
    assert recent[0]["session_id"] == s2  # Most recent first
    assert recent[1]["session_id"] == s1

    today = logger.get_today_summary()
    assert today["session_count"] == 2
    assert today["total_warnings"] == 3
    # 270 good frames out of 300 total frames = 90.0%
    assert today["upright_percentage"] == 90.0


def test_privacy_compliance_guarantee(temp_db):
    """
    CRITICAL PRIVACY TEST:
    Verifies that the database schema NEVER stores raw images, video frames,
    or binary biometric data.
    """
    logger = SessionLogger(db_path=temp_db)
    is_compliant = logger.verify_privacy_compliance()
    assert is_compliant is True

    # Double-check table columns directly
    conn = sqlite3.connect(temp_db)
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [r[0] for r in cursor.fetchall() if not r[0].startswith("sqlite_")]

        for table in tables:
            cursor.execute(f"PRAGMA table_info({table})")
            columns = cursor.fetchall()
            for col in columns:
                col_name = col[1].lower()
                col_type = col[2].upper()
                assert "blob" not in col_type, f"Forbidden BLOB column found: {table}.{col_name}"
                assert "image" not in col_name, f"Forbidden image column name: {table}.{col_name}"
                assert "video" not in col_name, f"Forbidden video column name: {table}.{col_name}"
                assert "frame" not in col_name or col_name.endswith("_frames"), (
                    f"Forbidden raw frame column: {table}.{col_name}"
                )
    finally:
        conn.close()



def test_end_nonexistent_session(temp_db):
    """Verify safe handling when ending an invalid or missing session ID."""
    logger = SessionLogger(db_path=temp_db)
    result = logger.end_session("non-existent-uuid", good_frames=0, total_frames=0)
    assert result is None
