"""
Smart Posture and Ergonomics Tracker
Main application desktop entry point.
"""
import sys
import os

# Windows 11 DLL isolation: Ensure System32 system libraries take precedence
# over legacy third-party DLLs (e.g. Anaconda PATH entries).
if sys.platform == "win32":
    import ctypes
    system32_icu = os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32", "icuuc.dll")
    if os.path.exists(system32_icu):
        try:
            ctypes.CDLL(system32_icu)
        except Exception:
            pass

import logging
from PySide6 import QtWidgets, QtCore, QtGui

from src.ui.dashboard import MainWindow

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("smart_posture_tracker")


def main() -> int:
    logger.info("Starting Smart Posture and Ergonomics Tracker...")
    # Enable high DPI scaling
    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName("Smart Posture Tracker")
    app.setOrganizationName("ErgonomicsLab")

    window = MainWindow()
    window.show()
    logger.info("Dashboard window displayed.")
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
