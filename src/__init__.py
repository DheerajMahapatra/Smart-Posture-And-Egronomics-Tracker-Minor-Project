"""
Smart Posture and Ergonomics Tracker source package.
"""
import sys
import os

__version__ = "0.1.0"

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
