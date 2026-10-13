"""
Unit and integration tests for Full-Screen Soft-Blur Translucent Overlay.

Tests cover:
- Topmost, frameless, and non-activating window flags
- Translucent background configuration
- UI card elements: "Fix your posture" warning, supporting hint, detected issue
- Holding progress state and progress bar calculations
- Transition to vibrant green "Posture corrected!" state
- Emergency manual dismissal via method and UI button
- Emergency keyboard shortcuts: Escape and Ctrl+Alt+P
- Windows DWM Acrylic backdrop safety
- MainWindow integration with reminder escalation events
"""
import pytest
from PySide6 import QtCore, QtGui, QtWidgets

from src.core.posture_engine import PostureState
from src.core.reminder_manager import ReminderEvent, ReminderState
from src.ui.dashboard import MainWindow
from src.ui.overlay import PostureOverlay, apply_windows_acrylic_blur
from src.ui.settings import AppSettings


class TestPostureOverlay:
    """Test suite for the full-screen overlay component."""

    def test_overlay_window_flags_and_attributes(self, qapp: QtWidgets.QApplication) -> None:
        overlay = PostureOverlay()
        flags = overlay.windowFlags()

        # Must be Frameless, Topmost, and Tool
        assert bool(flags & QtCore.Qt.WindowType.FramelessWindowHint) is True
        assert bool(flags & QtCore.Qt.WindowType.WindowStaysOnTopHint) is True
        assert bool(flags & QtCore.Qt.WindowType.Tool) is True

        # Attributes for alpha compositing and non-activating display
        assert overlay.testAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground) is True
        assert overlay.testAttribute(QtCore.Qt.WidgetAttribute.WA_ShowWithoutActivating) is True

        # Default typography
        assert overlay.lbl_title.text() == "Fix your posture"
        assert "Sit upright" in overlay.lbl_hint.text()
        assert overlay.btn_dismiss.text() == "Dismiss"

    def test_overlay_show_and_warning_state(self, qapp: QtWidgets.QApplication) -> None:
        overlay = PostureOverlay()
        overlay.show_overlay(issue="Severe forward craning")

        assert overlay.lbl_title.text() == "Fix your posture"
        assert "Severe forward craning" in overlay.lbl_issue.text()
        assert overlay.geometry().width() > 0
        assert overlay.geometry().height() > 0
        overlay.hide()

    def test_overlay_holding_state_updates_progress(self, qapp: QtWidgets.QApplication) -> None:
        overlay = PostureOverlay()
        overlay.show()
        overlay.set_warning_state("Slouching")
        assert overlay.progress_hold.isVisible() is False

        # User begins correcting posture (held 1.5s of 3.0s)
        overlay.set_holding_state(current_hold_sec=1.5, target_hold_sec=3.0)
        assert overlay.progress_hold.isVisible() is True
        assert overlay.progress_hold.value() == 50
        assert "1.5s / 3.0s" in overlay.lbl_hold_status.text()

        # Held 3.0s of 3.0s (100%)
        overlay.set_holding_state(current_hold_sec=3.0, target_hold_sec=3.0)
        assert overlay.progress_hold.value() == 100
        overlay.hide()

    def test_overlay_confirm_and_dismiss_turns_green(self, qapp: QtWidgets.QApplication) -> None:
        overlay = PostureOverlay()
        overlay.show_overlay("Slouching")
        overlay.confirm_and_dismiss()

        # Label turns to green "Posture corrected!"
        assert overlay.lbl_title.text() == "Posture corrected!"
        assert "#4CC674" in overlay.lbl_title.styleSheet()
        assert overlay._is_dismissing is True
        overlay.hide()

    def test_emergency_dismiss_triggers_signal(self, qapp: QtWidgets.QApplication) -> None:
        overlay = PostureOverlay()
        dismiss_reasons = []
        overlay.dismissed.connect(lambda r: dismiss_reasons.append(r))

        overlay.emergency_dismiss()
        assert "emergency" in dismiss_reasons
        assert overlay._is_dismissing is True
        overlay.hide()

    def test_emergency_dismiss_via_escape_key(self, qapp: QtWidgets.QApplication) -> None:
        overlay = PostureOverlay()
        dismiss_reasons = []
        overlay.dismissed.connect(lambda r: dismiss_reasons.append(r))

        # Synthesize Escape key press event
        key_event = QtGui.QKeyEvent(
            QtCore.QEvent.Type.KeyPress,
            QtCore.Qt.Key.Key_Escape,
            QtCore.Qt.KeyboardModifier.NoModifier,
        )
        overlay.keyPressEvent(key_event)

        assert "emergency" in dismiss_reasons
        overlay.hide()

    def test_emergency_dismiss_via_ctrl_alt_p(self, qapp: QtWidgets.QApplication) -> None:
        overlay = PostureOverlay()
        dismiss_reasons = []
        overlay.dismissed.connect(lambda r: dismiss_reasons.append(r))

        # Synthesize Ctrl+Alt+P key press event
        key_event = QtGui.QKeyEvent(
            QtCore.QEvent.Type.KeyPress,
            QtCore.Qt.Key.Key_P,
            QtCore.Qt.KeyboardModifier.ControlModifier | QtCore.Qt.KeyboardModifier.AltModifier,
        )
        overlay.keyPressEvent(key_event)

        assert "emergency" in dismiss_reasons
        overlay.hide()

    def test_windows_acrylic_blur_api_safety(self, qapp: QtWidgets.QApplication) -> None:
        overlay = PostureOverlay()
        # Even on null or test window ID, must never raise an unhandled exception
        result = apply_windows_acrylic_blur(int(overlay.winId()))
        assert isinstance(result, bool)

    def test_dashboard_overlay_integration(self, qapp: QtWidgets.QApplication) -> None:
        settings = AppSettings(demo_mode=True, max_warnings=3)
        window = MainWindow(settings)

        # 1. Simulate escalation event triggering overlay
        ev_trigger = ReminderEvent(
            state=ReminderState.OVERLAY_ACTIVE,
            warning_level=3,
            message="Full-screen overlay triggered.",
            issue_description="Slouching / hunched neck",
            should_alert=True,
            should_show_overlay=True,
            should_dismiss_overlay=False,
            sustained_poor_duration=900.0,
            correction_hold_duration=0.0,
        )
        window._on_reminder_event(ev_trigger)
        assert window.overlay.isVisible() is True
        assert "Slouching" in window.overlay.lbl_issue.text()

        # 2. Simulate user correcting posture behind overlay
        ev_hold = ReminderEvent(
            state=ReminderState.OVERLAY_ACTIVE,
            warning_level=3,
            message="Holding correct posture...",
            issue_description="Slouching / hunched neck",
            should_alert=False,
            should_show_overlay=False,
            should_dismiss_overlay=False,
            sustained_poor_duration=900.0,
            correction_hold_duration=2.0,
        )
        window._on_reminder_event(ev_hold)
        assert window.overlay.progress_hold.isVisible() is True

        # 3. Simulate correction confirmed dismissal
        ev_confirm = ReminderEvent(
            state=ReminderState.CORRECTION_CONFIRMED,
            warning_level=0,
            message="Posture corrected! Dismissing overlay.",
            issue_description="",
            should_alert=False,
            should_show_overlay=False,
            should_dismiss_overlay=True,
            sustained_poor_duration=0.0,
            correction_hold_duration=3.0,
        )
        window._on_reminder_event(ev_confirm)
        assert window.overlay.lbl_title.text() == "Posture corrected!"

        # Clean shutdown
        window.close()
