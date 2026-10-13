"""
Phase 1 Environment & Dependency Verification Test Suite.

Verifies:
1. Python runtime >= 3.10
2. Core dependencies: PySide6, OpenCV, MediaPipe, psutil, SQLite3
3. Project package imports across src/
"""
import sys
import sqlite3
import pytest


def test_python_version():
    """Verify Python runtime version is at least 3.10."""
    assert sys.version_info >= (3, 10), f"Python version must be >= 3.10, got {sys.version}"


def test_pyside6_import():
    """Verify PySide6 is installed, QtCore and QtWidgets can be initialized."""
    import PySide6
    from PySide6 import QtCore, QtWidgets
    assert hasattr(PySide6, "__version__"), "PySide6 version attribute missing"
    assert QtCore.qVersion(), "PySide6 QtCore failed to report Qt version"
    # Ensure QtWidgets can instantiate without crash
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    assert app is not None


def test_opencv_import():
    """Verify OpenCV (cv2) is installed and operational."""
    import cv2
    assert hasattr(cv2, "__version__"), "OpenCV version attribute missing"
    # Basic sanity check: create a blank 10x10 matrix
    import numpy as np
    mat = np.zeros((10, 10, 3), dtype=np.uint8)
    gray = cv2.cvtColor(mat, cv2.COLOR_BGR2GRAY)
    assert gray.shape == (10, 10), "OpenCV matrix transformation failed"


def test_mediapipe_import():
    """Verify MediaPipe is installed and PoseLandmarker is accessible."""
    import mediapipe as mp
    from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions
    assert hasattr(mp, "__version__"), "MediaPipe version attribute missing"
    assert PoseLandmarker is not None
    assert PoseLandmarkerOptions is not None


def test_psutil_and_power_api():
    """Verify psutil is installed and battery sensor API is accessible."""
    import psutil
    assert hasattr(psutil, "__version__"), "psutil version attribute missing"
    # Calling sensors_battery should either return battery object or None (desktop AC)
    battery = psutil.sensors_battery()
    # It must not raise an unhandled exception
    assert battery is None or hasattr(battery, "power_plugged")


def test_sqlite3_in_memory():
    """Verify SQLite3 is operational."""
    conn = sqlite3.connect(":memory:")
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE test_posture (id INTEGER PRIMARY KEY, state TEXT)")
    cursor.execute("INSERT INTO test_posture (state) VALUES ('good')")
    cursor.execute("SELECT state FROM test_posture WHERE id = 1")
    row = cursor.fetchone()
    conn.close()
    assert row is not None and row[0] == "good"


def test_src_modules_import():
    """Verify internal package layout is importable."""
    import src
    import src.core
    import src.vision
    import src.system
    import src.ui

    assert hasattr(src, "__version__")
