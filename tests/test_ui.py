"""
Unit and functional tests for Phase 4 desktop dashboard and settings UI.
"""
import os
import tempfile
import pytest

from PySide6 import QtCore, QtWidgets, QtGui

from src.core.posture_engine import PostureState, PostureAssessment
from src.ui.settings import AppSettings, SettingsDialog
from src.ui.dashboard import MainWindow
from src.ui.worker import PostureWorker


@pytest.fixture(scope="session")
def qapp():
    """Ensure QApplication exists for the entire test session."""
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    return app


# ============================================================================
# AppSettings Tests
# ============================================================================

def test_app_settings_defaults_and_roundtrip():
    """Verify settings defaults and JSON persistence roundtrip."""
    settings = AppSettings()
    assert settings.reminder_interval_sec == 300
    assert settings.persistence_threshold_sec == 20
    assert settings.max_warnings == 3
    assert settings.emergency_shortcut == "Esc"

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test_settings.json")
        settings.reminder_interval_sec = 600
        settings.max_warnings = 4
        settings.save(path)

        loaded = AppSettings.load(path)
        assert loaded.reminder_interval_sec == 600
        assert loaded.max_warnings == 4
        assert loaded.emergency_shortcut == "Esc"


# ============================================================================
# SettingsDialog Tests
# ============================================================================

def test_settings_dialog_initialization(qapp):
    """Verify SettingsDialog builds without error and reflects settings."""
    settings = AppSettings(reminder_interval_sec=180, max_warnings=2)
    dialog = SettingsDialog(settings)
    assert dialog.spin_interval.value() == 3
    assert dialog.spin_warnings.value() == 2
    dialog.close()


# ============================================================================
# MainWindow Tests
# ============================================================================

def test_main_window_initialization(qapp):
    """Verify MainWindow initializes all cards, status badges, and controls."""
    settings = AppSettings(demo_mode=True)
    win = MainWindow(settings)
    assert "Smart Posture" in win.windowTitle()
    assert win.btn_toggle_monitor.text() == "▶ Start Monitoring"
    assert "IDLE" in win.posture_pill.text()
    assert win.lbl_score_val.text() == "100%"
    assert win.lbl_time_val.text() == "00:00:00"
    win.close()


def test_main_window_start_stop_monitoring(qapp):
    """Verify starting and stopping monitoring in simulation/demo mode."""
    settings = AppSettings(demo_mode=True, target_fps=10)
    win = MainWindow(settings)

    # Start monitoring
    win._start_monitoring()
    assert win.is_monitoring is True
    assert "Stop Monitoring" in win.btn_toggle_monitor.text()
    assert win.worker is not None
    assert win.worker.isRunning()

    # Process events briefly
    qapp.processEvents()

    # Stop monitoring
    win._stop_monitoring()
    assert win.is_monitoring is False
    assert "Start Monitoring" in win.btn_toggle_monitor.text()
    assert win.worker is None
    win.close()


def test_main_window_assessment_updates_ui(qapp):
    """Verify that an incoming PostureAssessment correctly updates UI elements."""
    win = MainWindow(AppSettings(demo_mode=True))

    assessment_slouch = PostureAssessment(
        state=PostureState.SLOUCHING,
        is_good=False,
        primary_issue="Slouching detected",
        confidence=0.92,
        slouch_compression_ratio=0.75,
        shoulder_tilt_deg=1.0,
        forward_head_delta=0.02,
        eye_proximity_ratio=1.0,
        lateral_lean_deg=2.0,
    )

    win._on_assessment_ready(assessment_slouch)
    assert "SLOUCHING" in win.posture_pill.text()
    assert "Slouching detected" in win.posture_desc.text()
    assert win.bar_conf.value() == 92
    assert win.poor_posture_incidents == 1

    # Now feed a good posture assessment
    assessment_good = PostureAssessment(
        state=PostureState.GOOD,
        is_good=True,
        primary_issue="Optimal upright posture",
        confidence=0.95,
        slouch_compression_ratio=1.0,
        shoulder_tilt_deg=0.5,
        forward_head_delta=0.01,
        eye_proximity_ratio=1.0,
        lateral_lean_deg=1.0,
    )

    win._on_assessment_ready(assessment_good)
    assert "OPTIMAL" in win.posture_pill.text()
    win.close()


def test_main_window_force_test_overlay(qapp):
    """Verify manual test trigger of full-screen blur overlay."""
    settings = AppSettings(demo_mode=True)
    win = MainWindow(settings)
    win._force_test_overlay()
    assert win.overlay.isVisible()
    assert "Test Mode" in win.overlay.lbl_issue.text()
    win.overlay.hide()
    win.close()

