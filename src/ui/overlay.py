"""
Full-Screen Soft-Blur Translucent Overlay for Smart Posture and Ergonomics Tracker.

Critical UI Requirement (Section 3):
- Appears when user has ignored posture warnings.
- Topmost, frameless, translucent overlay covering all screens without minimizing,
  resizing, replacing, or switching away from the active application.
- Uses Windows 11 DWM SystemBackdrop Acrylic blur when available; gracefully falls
  back to soft translucent dimming scrim.
- Displays prominent centered warning: "Fix your posture".
- Displays supporting hint: "Sit upright and move a little farther from the screen."
- Live visual transition to vibrant green "Posture corrected!" when held upright.
- Automatic fade dismissal upon verified 3-second recovery.
- Reliable emergency keyboard dismissal via Escape or Ctrl+Alt+P.
"""
import ctypes
from ctypes import wintypes
import logging
import sys
from typing import Optional

from PySide6 import QtCore, QtGui, QtWidgets

logger = logging.getLogger(__name__)

# Windows 11 DWM Backdrop constants
DWMWA_SYSTEMBACKDROP_TYPE = 38
DWMSBT_AUTO = 0
DWMSBT_NONE = 1
DWMSBT_MAINWINDOW = 2         # Mica
DWMSBT_TRANSIENTWINDOW = 3   # Acrylic blur
DWMSBT_TABBEDWINDOW = 4      # Mica Alt


def apply_windows_acrylic_blur(hwnd: int) -> bool:
    """
    Attempt to enable Windows 11 hardware-accelerated Acrylic desktop blur
    behind the window using the Desktop Window Manager (DWM) API.

    Returns True if DWM Acrylic was successfully enabled, False if falling back
    to pure software translucent dimming.
    """
    if sys.platform != "win32":
        return False

    try:
        dwmapi = ctypes.WinDLL("dwmapi.dll")
        DwmSetWindowAttribute = dwmapi.DwmSetWindowAttribute
        DwmSetWindowAttribute.argtypes = [
            wintypes.HWND,
            wintypes.DWORD,
            wintypes.LPCVOID,
            wintypes.DWORD,
        ]
        DwmSetWindowAttribute.restype = ctypes.HRESULT

        backdrop_type = ctypes.c_int(DWMSBT_TRANSIENTWINDOW)
        hr = DwmSetWindowAttribute(
            wintypes.HWND(hwnd),
            wintypes.DWORD(DWMWA_SYSTEMBACKDROP_TYPE),
            ctypes.byref(backdrop_type),
            ctypes.sizeof(backdrop_type),
        )
        if hr == 0:
            logger.info("Windows 11 DWM Acrylic backdrop enabled successfully.")
            return True
        else:
            logger.info("DWM Acrylic returned HRESULT 0x%08X; using translucent dimming fallback.", hr & 0xFFFFFFFF)
            return False
    except Exception as e:
        logger.warning("Could not apply DWM Acrylic backdrop (%s); using translucent dimming fallback.", e)
        return False


class PostureOverlay(QtWidgets.QWidget):
    """
    Full-screen topmost translucent overlay window.
    Non-disruptive, non-activating, covers active workspace with soft backdrop.
    """

    dismissed = QtCore.Signal(str)  # Emits reason: 'corrected', 'emergency', 'closed'

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.is_true_blur_active = False
        self._is_dismissing = False
        self._auto_dismiss_timer: Optional[QtCore.QTimer] = None

        # Configure window flags: Topmost, Frameless, Tool (prevents focus stealing)
        self.setWindowFlags(
            QtCore.Qt.WindowType.FramelessWindowHint
            | QtCore.Qt.WindowType.WindowStaysOnTopHint
            | QtCore.Qt.WindowType.Tool
        )

        # Non-activating and translucent background attributes
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose, False)

        # Enable mouse tracking and keyboard focus when clicked
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)

        self._setup_ui()
        self._init_animations()

    def _setup_ui(self) -> None:
        """Construct the centered warning/recovery card interface."""
        root_layout = QtWidgets.QVBoxLayout(self)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)

        # Centered Card Frame
        self.card = QtWidgets.QFrame()
        self.card.setObjectName("overlayCard")
        self.card.setFixedWidth(560)
        self.card.setStyleSheet("""
            QFrame#overlayCard {
                background-color: rgba(24, 26, 34, 0.92);
                border: 1px solid rgba(255, 255, 255, 0.14);
                border-radius: 20px;
                padding: 24px 32px;
            }
        """)

        card_layout = QtWidgets.QVBoxLayout(self.card)
        card_layout.setSpacing(14)
        card_layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)

        # 1. State Badge Pill
        self.badge = QtWidgets.QLabel("⚠️  POSTURE ALERT")
        self.badge.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.badge.setStyleSheet("""
            background-color: #382A10;
            color: #F5A623;
            border: 1px solid #5A421A;
            border-radius: 12px;
            padding: 4px 16px;
            font-size: 13px;
            font-weight: 700;
            letter-spacing: 0.5px;
        """)
        card_layout.addWidget(self.badge)

        # 2. Main Title: "Fix your posture"
        self.lbl_title = QtWidgets.QLabel("Fix your posture")
        self.lbl_title.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.lbl_title.setStyleSheet("""
            color: #FFFFFF;
            font-size: 32px;
            font-weight: 800;
            margin-top: 4px;
        """)
        card_layout.addWidget(self.lbl_title)

        # 3. Supporting Hint: "Sit upright and move a little farther from the screen."
        self.lbl_hint = QtWidgets.QLabel("Sit upright and move a little farther from the screen.")
        self.lbl_hint.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.lbl_hint.setWordWrap(True)
        self.lbl_hint.setStyleSheet("""
            color: #D0D0D0;
            font-size: 16px;
            font-weight: 500;
            line-height: 1.4;
        """)
        card_layout.addWidget(self.lbl_hint)

        # 4. Detected Issue Specifics
        self.lbl_issue = QtWidgets.QLabel("Detected: Slouching / hunched neck")
        self.lbl_issue.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.lbl_issue.setStyleSheet("""
            color: #FF7B54;
            font-size: 13px;
            font-style: italic;
        """)
        card_layout.addWidget(self.lbl_issue)

        # 5. Recovery Hold Progress Bar
        self.progress_hold = QtWidgets.QProgressBar()
        self.progress_hold.setRange(0, 100)
        self.progress_hold.setValue(0)
        self.progress_hold.setTextVisible(False)
        self.progress_hold.setFixedHeight(8)
        self.progress_hold.setStyleSheet("""
            QProgressBar {
                background-color: #1F222E;
                border: none;
                border-radius: 4px;
            }
            QProgressBar::chunk {
                background-color: #4CC674;
                border-radius: 4px;
            }
        """)
        self.progress_hold.setVisible(False)
        card_layout.addWidget(self.progress_hold)

        self.lbl_hold_status = QtWidgets.QLabel("")
        self.lbl_hold_status.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.lbl_hold_status.setStyleSheet("color: #4CC674; font-size: 13px; font-weight: 600;")
        self.lbl_hold_status.setVisible(False)
        card_layout.addWidget(self.lbl_hold_status)

        # Divider line
        divider = QtWidgets.QFrame()
        divider.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        divider.setStyleSheet("color: rgba(255, 255, 255, 0.08);")
        card_layout.addWidget(divider)

        # 6. Bottom Controls & Emergency Shortcut Guidance
        footer_layout = QtWidgets.QHBoxLayout()
        footer_layout.setContentsMargins(4, 4, 4, 0)

        lbl_shortcut = QtWidgets.QLabel("Press Esc or Ctrl+Alt+P to dismiss")
        lbl_shortcut.setStyleSheet("color: #7A7A7A; font-size: 12px;")
        footer_layout.addWidget(lbl_shortcut)

        footer_layout.addStretch()

        self.btn_dismiss = QtWidgets.QPushButton("Dismiss")
        self.btn_dismiss.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.btn_dismiss.setStyleSheet("""
            QPushButton {
                background-color: #2D303E;
                color: #CCC;
                border: 1px solid #3E4254;
                border-radius: 6px;
                padding: 6px 16px;
                font-size: 12px;
                font-weight: 500;
            }
            QPushButton:hover {
                background-color: #3E4254;
                color: #FFF;
            }
        """)
        self.btn_dismiss.clicked.connect(self.emergency_dismiss)
        footer_layout.addWidget(self.btn_dismiss)

        card_layout.addLayout(footer_layout)
        root_layout.addWidget(self.card)

    def _init_animations(self) -> None:
        """Setup smooth opacity fade animations."""
        self.anim_fade = QtCore.QPropertyAnimation(self, b"windowOpacity")
        self.anim_fade.setDuration(350)
        self.anim_fade.finished.connect(self._on_fade_finished)

    def _update_geometry(self) -> None:
        """Span the overlay across all visible displays (virtual desktop)."""
        screens = QtGui.QGuiApplication.screens()
        if not screens:
            return
        virtual_geom = QtCore.QRect()
        for s in screens:
            virtual_geom = virtual_geom.united(s.geometry())
        self.setGeometry(virtual_geom)

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        """
        Draw a full-screen semi-transparent dark vignette scrim behind the card.
        If Windows DWM Acrylic is active, this adds gentle dimming.
        If DWM Acrylic is not supported, this provides the guaranteed soft translucent-dimming fallback.
        """
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)

        # Deep translucent scrim: rgba(10, 12, 18, 205) (~80% opacity)
        scrim_color = QtGui.QColor(10, 12, 18, 205)
        painter.fillRect(self.rect(), scrim_color)

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        """Handle emergency dismissal shortcuts (Esc or Ctrl+Alt+P)."""
        if event.key() == QtCore.Qt.Key.Key_Escape:
            logger.info("Overlay dismissed via Escape key shortcut.")
            self.emergency_dismiss()
            event.accept()
            return

        is_ctrl_alt = (
            event.modifiers()
            & (QtCore.Qt.KeyboardModifier.ControlModifier | QtCore.Qt.KeyboardModifier.AltModifier)
        ) == (QtCore.Qt.KeyboardModifier.ControlModifier | QtCore.Qt.KeyboardModifier.AltModifier)

        if is_ctrl_alt and event.key() == QtCore.Qt.Key.Key_P:
            logger.info("Overlay dismissed via Ctrl+Alt+P shortcut.")
            self.emergency_dismiss()
            event.accept()
            return

        super().keyPressEvent(event)

    def show_overlay(self, issue: str = "Slouching / hunched neck") -> None:
        """
        Activate and display the overlay over the current desktop.
        Does not steal focus or minimize running applications.
        """
        self._is_dismissing = False
        self._update_geometry()
        self.set_warning_state(issue)

        # Show without stealing active window focus
        self.show()
        if int(self.winId()) != 0:
            self.is_true_blur_active = apply_windows_acrylic_blur(int(self.winId()))

        # Animate smooth fade-in
        self.anim_fade.stop()
        self.anim_fade.setStartValue(0.0)
        self.anim_fade.setEndValue(1.0)
        self.anim_fade.start()

        logger.info("Posture overlay activated over desktop. True blur active: %s", self.is_true_blur_active)

    def set_warning_state(self, issue: str) -> None:
        """Display the prominent Warning state."""
        self.badge.setText("⚠️  POSTURE ALERT")
        self.badge.setStyleSheet("""
            background-color: #382A10; color: #F5A623; border: 1px solid #5A421A;
            border-radius: 12px; padding: 4px 16px; font-size: 13px; font-weight: 700;
        """)
        self.lbl_title.setText("Fix your posture")
        self.lbl_title.setStyleSheet("color: #FFFFFF; font-size: 32px; font-weight: 800; margin-top: 4px;")
        self.lbl_hint.setText("Sit upright and move a little farther from the screen.")
        self.lbl_issue.setText(f"Detected: {issue}")
        self.lbl_issue.setStyleSheet("color: #FF7B54; font-size: 13px; font-style: italic;")
        self.lbl_issue.setVisible(True)

        self.progress_hold.setVisible(False)
        self.lbl_hold_status.setVisible(False)

    def set_holding_state(self, current_hold_sec: float, target_hold_sec: float = 3.0) -> None:
        """Update progress bar when user begins correcting posture."""
        pct = int(min(100.0, (current_hold_sec / max(0.1, target_hold_sec)) * 100))
        self.progress_hold.setVisible(True)
        self.progress_hold.setValue(pct)

        self.lbl_hold_status.setVisible(True)
        self.lbl_hold_status.setText(f"Keep holding upright posture... ({current_hold_sec:.1f}s / {target_hold_sec:.1f}s)")

        self.lbl_title.setText("Hold your posture...")
        self.badge.setText("🔄  CORRECTING POSTURE")
        self.badge.setStyleSheet("""
            background-color: #1A3222; color: #4CC674; border: 1px solid #295436;
            border-radius: 12px; padding: 4px 16px; font-size: 13px; font-weight: 700;
        """)

    def confirm_and_dismiss(self) -> None:
        """
        Transition card to green 'Posture corrected!' state,
        then automatically fade away after a short confirmation pause.
        """
        if self._is_dismissing:
            return

        self._is_dismissing = True
        self.badge.setText("✅  POSTURE RESTORED")
        self.badge.setStyleSheet("""
            background-color: #104C28; color: #4CC674; border: 1px solid #1C7C41;
            border-radius: 12px; padding: 4px 16px; font-size: 13px; font-weight: 700;
        """)
        self.lbl_title.setText("Posture corrected!")
        self.lbl_title.setStyleSheet("color: #4CC674; font-size: 32px; font-weight: 800; margin-top: 4px;")
        self.lbl_hint.setText("Great job! Returning to your workspace.")
        self.lbl_issue.setVisible(False)
        self.progress_hold.setVisible(False)
        self.lbl_hold_status.setVisible(False)

        # Trigger short 1.2-second confirmation hold before fading out
        if self._auto_dismiss_timer is None:
            self._auto_dismiss_timer = QtCore.QTimer(self)
            self._auto_dismiss_timer.setSingleShot(True)
            self._auto_dismiss_timer.timeout.connect(self._fade_out_and_close)
        self._auto_dismiss_timer.start(1200)

    def emergency_dismiss(self) -> None:
        """Instant dismissal via user click or keyboard shortcut."""
        if self._is_dismissing:
            return
        self._is_dismissing = True
        self.dismissed.emit("emergency")
        self._fade_out_and_close()

    def _fade_out_and_close(self) -> None:
        """Animate smooth fade-out and hide."""
        self.anim_fade.stop()
        self.anim_fade.setStartValue(self.windowOpacity())
        self.anim_fade.setEndValue(0.0)
        self.anim_fade.start()

    def _on_fade_finished(self) -> None:
        """Handle end of animation."""
        if self._is_dismissing or self.windowOpacity() <= 0.05:
            self.hide()
            self.dismissed.emit("corrected")
