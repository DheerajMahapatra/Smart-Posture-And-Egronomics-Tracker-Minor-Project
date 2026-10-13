"""
Application Settings & Configuration Dialog for Smart Posture Tracker.
"""
from dataclasses import dataclass, asdict
import json
import logging
from pathlib import Path
from typing import Optional

from PySide6 import QtCore, QtWidgets, QtGui

from src.vision.camera import find_available_cameras

logger = logging.getLogger(__name__)



@dataclass
class AppSettings:
    """User-configurable parameters with sensible, safe defaults."""
    reminder_interval_sec: int = 300       # 5 minutes
    persistence_threshold_sec: int = 20    # 20 seconds of sustained poor posture
    max_warnings: int = 3                  # Max warnings before overlay escalation
    correction_confirmation_sec: float = 3.0
    camera_index: int = 0
    show_camera_preview: bool = True
    demo_mode: bool = False
    target_fps: int = 3                    # 2-5 FPS default when plugged in
    emergency_shortcut: str = "Esc"
    save_event_history: bool = True
    enable_sound_alerts: bool = True
    enable_tray_notifications: bool = True # Native Windows notification tray toast
    auto_detect_camera: bool = True        # Auto-detect connected phone/USB webcam
    ip_camera_url: str = ""                # Optional URL for IP Webcam phone stream
    battery_check_interval_sec: int = 300  # 4-5 minutes (default 300s)
    burst_duration_sec: float = 3.0        # 3 seconds burst sample
    prefer_phone_webcam: bool = True       # Prioritize phone/external webcam; never fallback to laptop camera
    allow_laptop_camera_on_battery: bool = True  # If using laptop camera, allow 4-5m periodic burst on battery


    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "AppSettings":
        valid_keys = {f for f in cls.__dataclass_fields__}
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)

    def save(self, filepath: Optional[str] = None) -> None:
        path = Path(filepath or self.default_settings_path())
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info("Saved settings to %s", path)

    @classmethod
    def load(cls, filepath: Optional[str] = None) -> "AppSettings":
        path = Path(filepath or cls.default_settings_path())
        if not path.exists():
            return cls()
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return cls.from_dict(data)
        except Exception as e:
            logger.warning("Failed to load settings from %s, using defaults: %s", path, e)
            return cls()

    @staticmethod
    def default_settings_path() -> Path:
        proj_root = Path(__file__).resolve().parent.parent.parent
        return proj_root / "data" / "settings.json"


class SettingsDialog(QtWidgets.QDialog):
    """
    Windows 11 styled modal settings dialog.
    """

    settings_saved = QtCore.Signal(AppSettings)

    def __init__(self, current_settings: AppSettings, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Settings - Smart Posture Tracker")
        self.resize(520, 560)
        self.setModal(True)
        self.settings = current_settings
        self._init_ui()

    def _init_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(16)

        # Header Title
        title_lbl = QtWidgets.QLabel("Application Settings")
        title_lbl.setStyleSheet("font-size: 18px; font-weight: 600; color: #E0E0E0;")
        layout.addWidget(title_lbl)

        # Tab Widget for Organization
        tabs = QtWidgets.QTabWidget()
        tabs.setStyleSheet("""
            QTabWidget::pane { border: 1px solid #333333; border-radius: 6px; background: #1E1E1E; }
            QTabBar::tab { background: #2A2A2A; color: #BBB; padding: 8px 16px; border-top-left-radius: 4px; border-top-right-radius: 4px; }
            QTabBar::tab:selected { background: #0078D4; color: #FFF; font-weight: bold; }
        """)

        # Tab 1: Ergonomics & Reminders
        tab_ergo = QtWidgets.QWidget()
        layout_ergo = QtWidgets.QFormLayout(tab_ergo)
        layout_ergo.setContentsMargins(16, 16, 16, 16)
        layout_ergo.setSpacing(12)

        self.spin_interval = QtWidgets.QSpinBox()
        self.spin_interval.setRange(1, 60)
        self.spin_interval.setSuffix(" minutes")
        self.spin_interval.setValue(self.settings.reminder_interval_sec // 60)
        layout_ergo.addRow("Reminder Frequency:", self.spin_interval)

        self.spin_persist = QtWidgets.QSpinBox()
        self.spin_persist.setRange(5, 120)
        self.spin_persist.setSuffix(" seconds")
        self.spin_persist.setValue(self.settings.persistence_threshold_sec)
        layout_ergo.addRow("Posture Persistence Trigger:", self.spin_persist)

        self.spin_warnings = QtWidgets.QSpinBox()
        self.spin_warnings.setRange(1, 5)
        self.spin_warnings.setValue(self.settings.max_warnings)
        layout_ergo.addRow("Warnings Before Overlay:", self.spin_warnings)

        self.spin_confirm = QtWidgets.QDoubleSpinBox()
        self.spin_confirm.setRange(1.0, 10.0)
        self.spin_confirm.setSingleStep(0.5)
        self.spin_confirm.setSuffix(" seconds")
        self.spin_confirm.setValue(self.settings.correction_confirmation_sec)
        layout_ergo.addRow("Correction Confirmation Hold:", self.spin_confirm)

        self.chk_sound = QtWidgets.QCheckBox("Play audible chimes on warnings")
        self.chk_sound.setChecked(self.settings.enable_sound_alerts)
        layout_ergo.addRow("Audible Alerts:", self.chk_sound)

        self.chk_tray = QtWidgets.QCheckBox("Show Windows notification tray popups (Windows 11 Toast)")
        self.chk_tray.setChecked(self.settings.enable_tray_notifications)
        layout_ergo.addRow("Windows Tray:", self.chk_tray)

        tabs.addTab(tab_ergo, "Reminders")

        # Tab 2: Vision & Camera
        tab_vision = QtWidgets.QWidget()
        layout_vision = QtWidgets.QFormLayout(tab_vision)
        layout_vision.setContentsMargins(16, 16, 16, 16)
        layout_vision.setSpacing(12)

        # Camera selection row
        cam_row = QtWidgets.QHBoxLayout()
        self.combo_camera = QtWidgets.QComboBox()
        self._populate_cameras()
        btn_refresh_cams = QtWidgets.QPushButton("🔄 Scan Devices")
        btn_refresh_cams.setStyleSheet("background: #2D2D2D; color: #FFF; padding: 4px 8px; border-radius: 4px;")
        btn_refresh_cams.clicked.connect(self._populate_cameras)
        cam_row.addWidget(self.combo_camera)
        cam_row.addWidget(btn_refresh_cams)
        layout_vision.addRow("Webcam Device:", cam_row)

        self.chk_auto_detect = QtWidgets.QCheckBox("Auto-detect connected cameras (phone / external / built-in)")
        self.chk_auto_detect.setChecked(self.settings.auto_detect_camera)
        layout_vision.addRow("Auto-Detection:", self.chk_auto_detect)

        self.chk_prefer_phone = QtWidgets.QCheckBox("Prioritize Phone Webcam (Never fallback to laptop camera)")
        self.chk_prefer_phone.setChecked(self.settings.prefer_phone_webcam)
        layout_vision.addRow("Phone Priority:", self.chk_prefer_phone)

        self.edit_ip_cam = QtWidgets.QLineEdit(self.settings.ip_camera_url)
        self.edit_ip_cam.setPlaceholderText("e.g. http://192.168.1.50:8080/video (Optional)")
        layout_vision.addRow("Phone IP Webcam:", self.edit_ip_cam)

        self.chk_allow_laptop_battery = QtWidgets.QCheckBox("Allow laptop camera in battery mode (4-5 min burst checks)")
        self.chk_allow_laptop_battery.setChecked(self.settings.allow_laptop_camera_on_battery)
        layout_vision.addRow("Laptop on Battery:", self.chk_allow_laptop_battery)

        self.spin_battery_interval = QtWidgets.QSpinBox()
        self.spin_battery_interval.setRange(1, 15)
        self.spin_battery_interval.setSuffix(" minutes")
        self.spin_battery_interval.setValue(max(1, self.settings.battery_check_interval_sec // 60))
        layout_vision.addRow("Battery Check Interval:", self.spin_battery_interval)

        self.spin_fps = QtWidgets.QSpinBox()
        self.spin_fps.setRange(1, 15)
        self.spin_fps.setSuffix(" FPS")
        self.spin_fps.setValue(self.settings.target_fps)
        layout_vision.addRow("Plugged-In Inference Rate:", self.spin_fps)

        self.chk_preview = QtWidgets.QCheckBox("Show camera preview feed in dashboard")
        self.chk_preview.setChecked(self.settings.show_camera_preview)
        layout_vision.addRow("Camera Preview:", self.chk_preview)

        self.chk_demo = QtWidgets.QCheckBox("Demo Mode (Synthetic simulation - No webcam required)")
        self.chk_demo.setChecked(self.settings.demo_mode)
        layout_vision.addRow("Mode Override:", self.chk_demo)

        tabs.addTab(tab_vision, "Vision & Hardware")


        # Tab 3: Privacy & Safety
        tab_privacy = QtWidgets.QWidget()
        layout_priv = QtWidgets.QVBoxLayout(tab_privacy)
        layout_priv.setContentsMargins(16, 16, 16, 16)
        layout_priv.setSpacing(12)

        shortcut_layout = QtWidgets.QHBoxLayout()
        shortcut_lbl = QtWidgets.QLabel("Emergency Overlay Dismissal Key:")
        self.edit_shortcut = QtWidgets.QLineEdit(self.settings.emergency_shortcut)
        self.edit_shortcut.setMaximumWidth(100)
        shortcut_layout.addWidget(shortcut_lbl)
        shortcut_layout.addWidget(self.edit_shortcut)
        layout_priv.addLayout(shortcut_layout)

        privacy_notice = QtWidgets.QTextBrowser()
        privacy_notice.setStyleSheet("background: #181818; color: #AAA; border: 1px solid #333; font-size: 12px;")
        privacy_notice.setHtml("""
            <b>Privacy & Security Assurance</b><br>
            • All video frames are processed locally in RAM and discarded immediately.<br>
            • No raw footage or photos are saved to your disk or sent across any network.<br>
            • Only anonymous session metrics (e.g., timestamps and posture category) are logged.<br>
            • The full-screen overlay does not capture passwords or sensitive windows.<br>
            <br><i>Ergonomic habit awareness tool. Not a medical diagnostic device.</i>
        """)
        layout_priv.addWidget(privacy_notice)
        tabs.addTab(tab_privacy, "Privacy & Safety")

        layout.addWidget(tabs)

        # Buttons
        btn_layout = QtWidgets.QHBoxLayout()
        btn_reset = QtWidgets.QPushButton("Reset Defaults")
        btn_reset.clicked.connect(self._reset_defaults)
        btn_cancel = QtWidgets.QPushButton("Cancel")
        btn_cancel.clicked.connect(self.reject)
        btn_save = QtWidgets.QPushButton("Save Settings")
        btn_save.setStyleSheet("background: #0078D4; color: white; font-weight: bold; padding: 6px 16px; border-radius: 4px;")
        btn_save.clicked.connect(self._save_and_close)

        btn_layout.addWidget(btn_reset)
        btn_layout.addStretch()
        btn_layout.addWidget(btn_cancel)
        btn_layout.addWidget(btn_save)
        layout.addLayout(btn_layout)

    def _populate_cameras(self) -> None:
        """Scan available camera devices and populate dropdown."""
        current_data = self.combo_camera.currentData() if self.combo_camera.count() > 0 else self.settings.camera_index
        self.combo_camera.clear()
        self.combo_camera.addItem("Auto-Detect Active Webcam", -1)

        prefer_phone = getattr(self, "chk_prefer_phone", None)
        is_phone_pref = prefer_phone.isChecked() if prefer_phone else self.settings.prefer_phone_webcam

        available = find_available_cameras(max_to_test=6, prefer_phone=is_phone_pref)
        if not available:
            self.combo_camera.addItem("Camera 0 (Default)", 0)
        else:
            for idx in available:
                label = f"📱 Camera {idx} (External / Phone)" if idx >= 1 else f"💻 Camera {idx} (Laptop Built-in)"
                self.combo_camera.addItem(label, idx)

        # Reselect previous or current setting
        for i in range(self.combo_camera.count()):
            if self.combo_camera.itemData(i) == current_data:
                self.combo_camera.setCurrentIndex(i)
                break

    def _reset_defaults(self) -> None:
        defaults = AppSettings()
        self.spin_interval.setValue(defaults.reminder_interval_sec // 60)
        self.spin_persist.setValue(defaults.persistence_threshold_sec)
        self.spin_warnings.setValue(defaults.max_warnings)
        self.spin_confirm.setValue(defaults.correction_confirmation_sec)
        self.spin_fps.setValue(defaults.target_fps)
        self.chk_preview.setChecked(defaults.show_camera_preview)
        self.chk_demo.setChecked(defaults.demo_mode)
        self.chk_sound.setChecked(defaults.enable_sound_alerts)
        self.chk_tray.setChecked(defaults.enable_tray_notifications)
        self.chk_auto_detect.setChecked(defaults.auto_detect_camera)
        self.chk_prefer_phone.setChecked(defaults.prefer_phone_webcam)
        self.chk_allow_laptop_battery.setChecked(defaults.allow_laptop_camera_on_battery)
        self.spin_battery_interval.setValue(defaults.battery_check_interval_sec // 60)
        self.edit_ip_cam.setText(defaults.ip_camera_url)
        self.edit_shortcut.setText(defaults.emergency_shortcut)
        self._populate_cameras()

    def _save_and_close(self) -> None:
        self.settings.reminder_interval_sec = self.spin_interval.value() * 60
        self.settings.persistence_threshold_sec = self.spin_persist.value()
        self.settings.max_warnings = self.spin_warnings.value()
        self.settings.correction_confirmation_sec = self.spin_confirm.value()
        self.settings.enable_sound_alerts = self.chk_sound.isChecked()
        self.settings.enable_tray_notifications = self.chk_tray.isChecked()
        self.settings.auto_detect_camera = self.chk_auto_detect.isChecked()
        self.settings.prefer_phone_webcam = self.chk_prefer_phone.isChecked()
        self.settings.allow_laptop_camera_on_battery = self.chk_allow_laptop_battery.isChecked()
        self.settings.battery_check_interval_sec = self.spin_battery_interval.value() * 60
        self.settings.ip_camera_url = self.edit_ip_cam.text().strip()

        # Camera selection
        cam_data = self.combo_camera.currentData()
        if cam_data is not None and cam_data >= 0:
            self.settings.camera_index = cam_data
        elif self.settings.ip_camera_url:
            self.settings.camera_index = 0  # Fallback index if URL fails

        self.settings.target_fps = self.spin_fps.value()
        self.settings.show_camera_preview = self.chk_preview.isChecked()
        self.settings.demo_mode = self.chk_demo.isChecked()
        self.settings.emergency_shortcut = self.edit_shortcut.text().strip() or "Esc"

        self.settings.save()
        self.settings_saved.emit(self.settings)
        self.accept()

