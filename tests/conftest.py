"""
Pytest configuration for Smart Posture and Ergonomics Tracker.
Ensures Windows 11 system libraries take precedence over third-party PATH collisions.
"""
import sys
import os

if sys.platform == "win32":
    import ctypes
    system32 = os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32")
    system32_icu = os.path.join(system32, "icuuc.dll")
    if os.path.exists(system32_icu):
        try:
            ctypes.CDLL(system32_icu)
        except Exception:
            pass

import pytest
from PySide6 import QtWidgets


@pytest.fixture(scope="session")
def qapp() -> QtWidgets.QApplication:
    """Ensure a shared QApplication instance exists for UI tests."""
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    return app
