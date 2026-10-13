"""
Main Desktop Dashboard for Smart Posture and Ergonomics Tracker.

Windows 11 Fluent dark-styled PySide6 desktop interface:
- Live posture status badge with color-coded feedback
- Non-blocking camera / simulation preview viewport
- Session analytics, upright percentage, and incident counter
- Calibration wizard with live progress feedback
- Hardware / simulation controls and settings modal integration
- Power state awareness (AC Continuous vs Battery Saver)
"""
from datetime import datetime, timezone
import logging
import os
import time
from typing import Optional
import psutil

from PySide6 import QtCore, QtGui, QtWidgets

from src.core.calibration import CalibrationProfile, MEDICAL_DISCLAIMER
from src.core.posture_engine import PostureAssessment, PostureState
from src.core.reminder_manager import ReminderEvent, ReminderState
from src.system.power_manager import PowerMode, PowerState
from src.system.session_logger import SessionLogger
from src.ui.settings import AppSettings, SettingsDialog
from src.ui.worker import PostureWorker
from src.ui.overlay import PostureOverlay
from src.vision.camera import find_available_cameras


logger = logging.getLogger(__name__)


class MainWindow(QtWidgets.QMainWindow):
    """
    Main Application Window for Smart Posture Tracker.
    """

    def __init__(self, settings: Optional[AppSettings] = None) -> None:
        super().__init__()
        self.settings = settings or AppSettings.load()
        self.worker: Optional[PostureWorker] = None

        # Session tracking state
        self.is_monitoring = False
        self.session_start_time: Optional[float] = None
        self.total_frames_evaluated = 0
        self.good_frames_count = 0
        self.poor_posture_incidents = 0
        self.last_state = PostureState.GOOD

        # SQLite Session Analytics Logger
        self.session_logger = SessionLogger()
        self.current_session_id: Optional[str] = None

        # Full-Screen Soft-Blur Translucent Overlay
        self.overlay = PostureOverlay()
        self.overlay.dismissed.connect(self._on_overlay_dismissed)

        self._init_window()
        self._init_ui()
        self._init_timers()
        self._setup_system_tray()
        self._update_power_state_badge()


    def _init_window(self) -> None:
        self.setWindowTitle("Smart Posture and Ergonomics Tracker")
        self.resize(1080, 720)
        self.setMinimumSize(900, 600)
        # Windows 11 dark theme stylesheet
        self.setStyleSheet("""
            QMainWindow {
                background-color: #121212;
            }
            QWidget {
                color: #E0E0E0;
                font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
            }
            QFrame.card {
                background-color: #1E1E1E;
                border: 1px solid #2D2D2D;
                border-radius: 8px;
                padding: 16px;
            }
            QLabel.section-title {
                font-size: 15px;
                font-weight: 600;
                color: #A0A0A0;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }
            QPushButton.primary-btn {
                background-color: #0078D4;
                color: #FFFFFF;
                border: none;
                border-radius: 6px;
                padding: 10px 20px;
                font-size: 14px;
                font-weight: 600;
            }
            QPushButton.primary-btn:hover {
                background-color: #106EBE;
            }
            QPushButton.secondary-btn {
                background-color: #2D2D2D;
                color: #E0E0E0;
                border: 1px solid #3E3E3E;
                border-radius: 6px;
                padding: 8px 16px;
                font-size: 13px;
                font-weight: 500;
            }
            QPushButton.secondary-btn:hover {
                background-color: #383838;
            }
            QPushButton.stop-btn {
                background-color: #C42B1C;
                color: #FFFFFF;
                border: none;
                border-radius: 6px;
                padding: 10px 20px;
                font-size: 14px;
                font-weight: 600;
            }
            QPushButton.stop-btn:hover {
                background-color: #A82315;
            }
            QProgressBar {
                border: 1px solid #333333;
                border-radius: 4px;
                background-color: #2A2A2A;
                text-align: center;
                color: #FFF;
                height: 14px;
            }
            QProgressBar::chunk {
                background-color: #0078D4;
                border-radius: 3px;
            }
            QListWidget {
                background-color: #181818;
                border: 1px solid #2D2D2D;
                border-radius: 6px;
                padding: 4px;
                font-size: 12px;
            }
        """)

    def _init_ui(self) -> None:
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QtWidgets.QVBoxLayout(central_widget)
        main_layout.setContentsMargins(20, 16, 20, 16)
        main_layout.setSpacing(14)

        # --------------------------------------------------------------------
        # 1. Header Bar
        # --------------------------------------------------------------------
        header_layout = QtWidgets.QHBoxLayout()
        header_title = QtWidgets.QLabel("Smart Posture & Ergonomics Tracker")
        header_title.setStyleSheet("font-size: 20px; font-weight: 700; color: #FFFFFF;")
        header_layout.addWidget(header_title)

        header_layout.addStretch()

        # Power badge
        self.power_badge = QtWidgets.QLabel("AC Power")
        self.power_badge.setStyleSheet("""
            background-color: #1C3322;
            color: #4CC674;
            border: 1px solid #285434;
            border-radius: 12px;
            padding: 4px 12px;
            font-size: 12px;
            font-weight: 600;
        """)
        header_layout.addWidget(self.power_badge)

        # Camera mode badge
        self.camera_badge = QtWidgets.QLabel("Camera: Idle")
        self.camera_badge.setStyleSheet("""
            background-color: #2D2D2D;
            color: #CCCCCC;
            border: 1px solid #3E3E3E;
            border-radius: 12px;
            padding: 4px 12px;
            font-size: 12px;
            font-weight: 500;
        """)
        header_layout.addWidget(self.camera_badge)

        # Scan / Reconnect camera button
        btn_scan_cam = QtWidgets.QPushButton("🔄 Scan Camera")
        btn_scan_cam.setStyleSheet("""
            background-color: #2A2A2A;
            color: #DDD;
            border: 1px solid #3A3A3A;
            border-radius: 6px;
            padding: 5px 12px;
            font-size: 12px;
        """)
        btn_scan_cam.clicked.connect(self._reconnect_camera)
        header_layout.addWidget(btn_scan_cam)

        # Settings button
        btn_settings = QtWidgets.QPushButton("⚙ Settings")
        btn_settings.setStyleSheet("""
            background-color: #2A2A2A;
            color: #DDD;
            border: 1px solid #3A3A3A;
            border-radius: 6px;
            padding: 5px 12px;
            font-size: 12px;
        """)
        btn_settings.clicked.connect(self._open_settings)
        header_layout.addWidget(btn_settings)


        main_layout.addLayout(header_layout)

        # --------------------------------------------------------------------
        # 2. Main Body Grid: Left (Status & Video) + Right (Stats & Controls)
        # --------------------------------------------------------------------
        body_layout = QtWidgets.QHBoxLayout()
        body_layout.setSpacing(16)

        # --- Left Column (60% width) ---
        left_col = QtWidgets.QVBoxLayout()
        left_col.setSpacing(14)

        # Card 1: Posture Status Card
        card_status = QtWidgets.QFrame()
        card_status.setProperty("class", "card")
        layout_status = QtWidgets.QVBoxLayout(card_status)
        layout_status.setSpacing(10)

        lbl_sec1 = QtWidgets.QLabel("Current Posture State")
        lbl_sec1.setProperty("class", "section-title")
        layout_status.addWidget(lbl_sec1)

        self.posture_pill = QtWidgets.QLabel("IDLE - MONITORING STOPPED")
        self.posture_pill.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.posture_pill.setStyleSheet("""
            background-color: #2D2D2D;
            color: #A0A0A0;
            border-radius: 8px;
            padding: 12px 24px;
            font-size: 18px;
            font-weight: 700;
            letter-spacing: 0.5px;
        """)
        layout_status.addWidget(self.posture_pill)

        self.posture_desc = QtWidgets.QLabel("Click 'Start Monitoring' to begin ergonomics tracking.")
        self.posture_desc.setStyleSheet("color: #CCCCCC; font-size: 13px;")
        layout_status.addWidget(self.posture_desc)

        # Metric bars
        metrics_grid = QtWidgets.QGridLayout()
        metrics_grid.setSpacing(8)

        metrics_grid.addWidget(QtWidgets.QLabel("Confidence:"), 0, 0)
        self.bar_conf = QtWidgets.QProgressBar()
        self.bar_conf.setRange(0, 100)
        self.bar_conf.setValue(0)
        metrics_grid.addWidget(self.bar_conf, 0, 1)

        metrics_grid.addWidget(QtWidgets.QLabel("Neck Alignment:"), 1, 0)
        self.bar_neck = QtWidgets.QProgressBar()
        self.bar_neck.setRange(0, 100)
        self.bar_neck.setValue(100)
        metrics_grid.addWidget(self.bar_neck, 1, 1)

        metrics_grid.addWidget(QtWidgets.QLabel("Eye Proximity:"), 2, 0)
        self.bar_eye = QtWidgets.QProgressBar()
        self.bar_eye.setRange(0, 100)
        self.bar_eye.setValue(50)
        metrics_grid.addWidget(self.bar_eye, 2, 1)

        layout_status.addLayout(metrics_grid)
        left_col.addWidget(card_status)

        # Card 2: Viewport / Camera Card
        card_video = QtWidgets.QFrame()
        card_video.setProperty("class", "card")
        layout_video = QtWidgets.QVBoxLayout(card_video)
        layout_video.setSpacing(8)

        lbl_sec2 = QtWidgets.QLabel("Live Visual Feed & Simulation Viewport")
        lbl_sec2.setProperty("class", "section-title")
        layout_video.addWidget(lbl_sec2)

        self.video_viewport = QtWidgets.QLabel("Camera / Simulation Preview")
        self.video_viewport.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.video_viewport.setMinimumSize(420, 240)
        self.video_viewport.setStyleSheet("""
            background-color: #161616;
            border: 1px dashed #3A3A3A;
            border-radius: 6px;
            color: #666;
            font-size: 13px;
        """)
        layout_video.addWidget(self.video_viewport)

        # Simulation Quick Selector Bar
        sim_bar = QtWidgets.QHBoxLayout()
        sim_bar.addWidget(QtWidgets.QLabel("Simulate:"))
        self.combo_sim = QtWidgets.QComboBox()
        self.combo_sim.addItems([
            "GOOD (Optimal Posture)",
            "SLOUCHING (Hunched Neck)",
            "FORWARD_HEAD (Tech Neck)",
            "SHOULDER_MISALIGNMENT (Tilted)",
            "LEANING (Sideways)",
            "TOO_CLOSE (Screen Proximity / Eye Strain)",
            "NO_PERSON (Empty Chair)",
        ])
        self.combo_sim.setStyleSheet("background-color: #2D2D2D; color: #FFF; padding: 4px;")
        self.combo_sim.currentIndexChanged.connect(self._on_sim_changed)
        sim_bar.addWidget(self.combo_sim)
        layout_video.addLayout(sim_bar)

        left_col.addWidget(card_video)
        body_layout.addLayout(left_col, stretch=6)

        # --- Right Column (40% width) ---
        right_col = QtWidgets.QVBoxLayout()
        right_col.setSpacing(14)

        # Card 3: Controls Card
        card_ctrl = QtWidgets.QFrame()
        card_ctrl.setProperty("class", "card")
        layout_ctrl = QtWidgets.QVBoxLayout(card_ctrl)
        layout_ctrl.setSpacing(10)

        lbl_sec3 = QtWidgets.QLabel("Controls & Calibration")
        lbl_sec3.setProperty("class", "section-title")
        layout_ctrl.addWidget(lbl_sec3)

        self.btn_toggle_monitor = QtWidgets.QPushButton("▶ Start Monitoring")
        self.btn_toggle_monitor.setProperty("class", "primary-btn")
        self.btn_toggle_monitor.clicked.connect(self._toggle_monitoring)
        layout_ctrl.addWidget(self.btn_toggle_monitor)

        self.btn_calibrate = QtWidgets.QPushButton("🎯 Calibrate Posture Baseline")
        self.btn_calibrate.setProperty("class", "secondary-btn")
        self.btn_calibrate.clicked.connect(self._start_calibration)
        layout_ctrl.addWidget(self.btn_calibrate)

        self.calib_progress = QtWidgets.QProgressBar()
        self.calib_progress.setRange(0, 100)
        self.calib_progress.setValue(0)
        self.calib_progress.setVisible(False)
        layout_ctrl.addWidget(self.calib_progress)

        # Force test overlay button
        self.btn_test_overlay = QtWidgets.QPushButton("🚨 Test Blur Overlay Now (F12)")
        self.btn_test_overlay.setStyleSheet("""
            background-color: #8C2114;
            color: #FFFFFF;
            border: 1px solid #B02A1A;
            border-radius: 6px;
            padding: 9px 16px;
            font-size: 13px;
            font-weight: 600;
        """)
        self.btn_test_overlay.clicked.connect(self._force_test_overlay)
        layout_ctrl.addWidget(self.btn_test_overlay)

        # On-demand sample button (especially useful on battery)
        self.btn_check_now = QtWidgets.QPushButton("📸 Sample Posture Check Now")
        self.btn_check_now.setStyleSheet("""
            background-color: #1A3E26;
            color: #4CC674;
            border: 1px solid #285434;
            border-radius: 6px;
            padding: 8px 16px;
            font-size: 13px;
            font-weight: 600;
        """)
        self.btn_check_now.clicked.connect(self._check_posture_now)
        layout_ctrl.addWidget(self.btn_check_now)

        # Global shortcuts: F12 and Ctrl+T to force-trigger overlay test
        self.shortcut_f12 = QtGui.QShortcut(QtGui.QKeySequence("F12"), self)
        self.shortcut_f12.activated.connect(self._force_test_overlay)
        self.shortcut_ctrl_t = QtGui.QShortcut(QtGui.QKeySequence("Ctrl+T"), self)
        self.shortcut_ctrl_t.activated.connect(self._force_test_overlay)

        right_col.addWidget(card_ctrl)



        # Card 4: Session Analytics Card
        card_stats = QtWidgets.QFrame()
        card_stats.setProperty("class", "card")
        layout_stats = QtWidgets.QGridLayout(card_stats)
        layout_stats.setSpacing(12)

        lbl_sec4 = QtWidgets.QLabel("Session Statistics")
        lbl_sec4.setProperty("class", "section-title")
        layout_stats.addWidget(lbl_sec4, 0, 0, 1, 2)

        self.lbl_time_val = QtWidgets.QLabel("00:00:00")
        self.lbl_time_val.setStyleSheet("font-size: 22px; font-weight: 700; color: #FFFFFF;")
        layout_stats.addWidget(QtWidgets.QLabel("Duration:"), 1, 0)
        layout_stats.addWidget(self.lbl_time_val, 1, 1)

        self.lbl_score_val = QtWidgets.QLabel("100%")
        self.lbl_score_val.setStyleSheet("font-size: 22px; font-weight: 700; color: #4CC674;")
        layout_stats.addWidget(QtWidgets.QLabel("Upright Ratio:"), 2, 0)
        layout_stats.addWidget(self.lbl_score_val, 2, 1)

        self.lbl_warn_val = QtWidgets.QLabel(f"0 / {self.settings.max_warnings}")
        self.lbl_warn_val.setStyleSheet("font-size: 22px; font-weight: 700; color: #E0E0E0;")
        layout_stats.addWidget(QtWidgets.QLabel("Warning Level:"), 3, 0)
        layout_stats.addWidget(self.lbl_warn_val, 3, 1)

        right_col.addWidget(card_stats)

        # Card 5: Recent Posture Events List
        card_events = QtWidgets.QFrame()
        card_events.setProperty("class", "card")
        layout_events = QtWidgets.QVBoxLayout(card_events)
        layout_events.setSpacing(8)

        lbl_sec5 = QtWidgets.QLabel("Recent Posture Events")
        lbl_sec5.setProperty("class", "section-title")
        layout_events.addWidget(lbl_sec5)

        self.list_events = QtWidgets.QListWidget()
        self.list_events.addItem(f"[{datetime.now().strftime('%H:%M:%S')}] Application started.")
        layout_events.addWidget(self.list_events)

        right_col.addWidget(card_events)
        body_layout.addLayout(right_col, stretch=4)

        main_layout.addLayout(body_layout)

        # --------------------------------------------------------------------
        # 3. Footer / Medical Disclaimer
        # --------------------------------------------------------------------
        footer_lbl = QtWidgets.QLabel(MEDICAL_DISCLAIMER)
        footer_lbl.setStyleSheet("color: #707070; font-size: 11px;")
        footer_lbl.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(footer_lbl)

    def _init_timers(self) -> None:
        """Initialize background UI refresh timers."""
        # Session duration timer
        self.timer_session = QtCore.QTimer(self)
        self.timer_session.setInterval(1000)
        self.timer_session.timeout.connect(self._update_session_timer)

        # Power state probe timer (every 10 seconds)
        self.timer_power = QtCore.QTimer(self)
        self.timer_power.setInterval(10000)
        self.timer_power.timeout.connect(self._update_power_state_badge)
        self.timer_power.start()

    def _update_power_state_badge(self) -> None:
        """Query battery status and update power mode badge."""
        battery = psutil.sensors_battery()
        if battery is None or battery.power_plugged:
            self.power_badge.setText("⚡ AC Power (Continuous)")
            self.power_badge.setStyleSheet("""
                background-color: #1C3322; color: #4CC674; border: 1px solid #285434;
                border-radius: 12px; padding: 4px 12px; font-size: 12px; font-weight: 600;
            """)
        else:
            pct = int(battery.percent)
            self.power_badge.setText(f"🔋 Battery {pct}% (Burst Mode)")
            self.power_badge.setStyleSheet("""
                background-color: #382A10; color: #F5A623; border: 1px solid #5A421A;
                border-radius: 12px; padding: 4px 12px; font-size: 12px; font-weight: 600;
            """)

    def _update_session_timer(self) -> None:
        if not self.is_monitoring or not self.session_start_time:
            return
        elapsed = int(time.time() - self.session_start_time)
        hrs = elapsed // 3600
        mins = (elapsed % 3600) // 60
        secs = elapsed % 60
        self.lbl_time_val.setText(f"{hrs:02d}:{mins:02d}:{secs:02d}")

    def _toggle_monitoring(self) -> None:
        """Toggle monitoring on or off."""
        if self.is_monitoring:
            self._stop_monitoring()
        else:
            self._start_monitoring()

    def _start_monitoring(self) -> None:
        """Instantiate and launch worker thread."""
        self.is_monitoring = True
        self.session_start_time = time.time()
        self.total_frames_evaluated = 0
        self.good_frames_count = 0
        self.timer_session.start()

        self.btn_toggle_monitor.setText("⏹ Stop Monitoring")
        self.btn_toggle_monitor.setStyleSheet("""
            background-color: #C42B1C; color: #FFF; border: none; border-radius: 6px;
            padding: 10px 20px; font-size: 14px; font-weight: 600;
        """)

        # Start worker thread
        self.worker = PostureWorker(self.settings)
        self.worker.assessment_ready.connect(self._on_assessment_ready)
        self.worker.reminder_event.connect(self._on_reminder_event)
        self.worker.power_state_changed.connect(self._on_power_state_changed)
        self.worker.frame_ready.connect(self._on_frame_ready)
        self.worker.calibration_progress.connect(self._on_calibration_progress)
        self.worker.calibration_finished.connect(self._on_calibration_finished)
        self.worker.camera_mode_changed.connect(self._on_camera_mode_changed)
        self.worker.status_changed.connect(self._on_status_changed)
        self.worker.start()

        # Start SQLite session
        if self.settings.save_event_history:
            self.current_session_id = self.session_logger.start_session()
            self.session_logger.log_event(self.current_session_id, "SESSION_START", "Monitoring session started.")

        self._add_event("Monitoring session started.")

    def _stop_monitoring(self) -> None:
        """Safely terminate worker thread and update UI."""
        self.is_monitoring = False
        self.timer_session.stop()

        # Finalize SQLite session
        if self.current_session_id and self.settings.save_event_history:
            self.session_logger.end_session(
                session_id=self.current_session_id,
                good_frames=self.good_frames_count,
                total_frames=self.total_frames_evaluated,
                warning_count=self.poor_posture_incidents,
            )
            self.current_session_id = None

        self.btn_toggle_monitor.setText("▶ Start Monitoring")
        self.btn_toggle_monitor.setStyleSheet("""
            background-color: #0078D4; color: #FFF; border: none; border-radius: 6px;
            padding: 10px 20px; font-size: 14px; font-weight: 600;
        """)

        if self.worker:
            self.worker.stop()
            self.worker = None

        if self.overlay and self.overlay.isVisible():
            self.overlay.hide()

        self.lbl_warn_val.setText(f"0 / {self.settings.max_warnings}")
        self.lbl_warn_val.setStyleSheet("font-size: 22px; font-weight: 700; color: #E0E0E0;")
        self.posture_pill.setText("IDLE - MONITORING STOPPED")
        self.posture_pill.setStyleSheet("""
            background-color: #2D2D2D; color: #A0A0A0; border-radius: 8px;
            padding: 12px 24px; font-size: 18px; font-weight: 700;
        """)
        self.posture_desc.setText("Monitoring is stopped. Click 'Start Monitoring' to resume.")
        self.camera_badge.setText("Camera: Idle")
        self.camera_badge.setStyleSheet("background-color: #2D2D2D; color: #CCC; border-radius: 12px; padding: 4px 12px;")

        self._add_event("Monitoring session stopped.")

    def _start_calibration(self) -> None:
        """Trigger calibration."""
        if not self.is_monitoring:
            self._start_monitoring()

        self.calib_progress.setVisible(True)
        self.calib_progress.setValue(0)
        self.btn_calibrate.setEnabled(False)
        if self.worker:
            self.worker.start_calibration()
        self._add_event("Calibration started. Sit upright.")

    def _on_calibration_progress(self, fraction: float, count: int) -> None:
        self.calib_progress.setValue(int(fraction * 100))
        self.posture_desc.setText(f"Calibrating posture... ({count}/30 samples captured)")

    def _on_calibration_finished(self, profile: CalibrationProfile) -> None:
        self.calib_progress.setVisible(False)
        self.btn_calibrate.setEnabled(True)
        self.posture_desc.setText("Personal calibration saved! Tracking calibrated posture.")
        self._add_event("Personal baseline calibration complete.")

        # Persist calibration profile in SQLite
        if self.settings.save_event_history:
            self.session_logger.save_calibration(profile.to_dict())
            if self.current_session_id:
                self.session_logger.log_event(
                    self.current_session_id,
                    "CALIBRATION_SAVED",
                    "Personal posture baseline calibrated and saved.",
                )

    def _on_assessment_ready(self, assessment: PostureAssessment) -> None:
        """Update live posture UI badges and stats from worker inference."""
        self.total_frames_evaluated += 1
        if assessment.is_good:
            self.good_frames_count += 1

        # Check for incident transitions
        if not assessment.is_good and assessment.state != PostureState.NO_PERSON:
            if self.last_state == PostureState.GOOD:
                self.poor_posture_incidents += 1
                self._add_event(f"Issue: {assessment.primary_issue}")

        self.last_state = assessment.state

        # Update Upright Score
        if self.total_frames_evaluated > 0:
            score = int((self.good_frames_count / self.total_frames_evaluated) * 100)
            self.lbl_score_val.setText(f"{score}%")
            if score >= 80:
                self.lbl_score_val.setStyleSheet("font-size: 22px; font-weight: 700; color: #4CC674;")
            else:
                self.lbl_score_val.setStyleSheet("font-size: 22px; font-weight: 700; color: #F5A623;")

        # Update Posture Badge
        state = assessment.state
        if state == PostureState.GOOD:
            self.posture_pill.setText("OPTIMAL UPRIGHT POSTURE")
            self.posture_pill.setStyleSheet("""
                background-color: #107C41; color: #FFFFFF; border-radius: 8px;
                padding: 12px 24px; font-size: 18px; font-weight: 700;
            """)
        elif state == PostureState.NO_PERSON:
            self.posture_pill.setText("NO PERSON DETECTED")
            self.posture_pill.setStyleSheet("""
                background-color: #383838; color: #CCCCCC; border-radius: 8px;
                padding: 12px 24px; font-size: 18px; font-weight: 700;
            """)
        else:
            # Poor posture detected
            self.posture_pill.setText(f"ATTENTION: {state.value}")
            self.posture_pill.setStyleSheet("""
                background-color: #D83B01; color: #FFFFFF; border-radius: 8px;
                padding: 12px 24px; font-size: 18px; font-weight: 700;
            """)

        self.posture_desc.setText(assessment.primary_issue)

        # Update Progress Meters
        self.bar_conf.setValue(int(assessment.confidence * 100))
        neck_pct = int(min(100, max(0, assessment.slouch_compression_ratio * 100)))
        self.bar_neck.setValue(neck_pct)
        eye_pct = int(min(100, max(0, (assessment.eye_proximity_ratio / 1.5) * 100)))
        self.bar_eye.setValue(eye_pct)

    def _on_reminder_event(self, event: ReminderEvent) -> None:
        """Handle reminder escalation events from worker thread."""
        self.lbl_warn_val.setText(f"{event.warning_level} / {self.settings.max_warnings}")
        if event.warning_level > 0:
            self.lbl_warn_val.setStyleSheet("font-size: 22px; font-weight: 700; color: #D83B01;")
        else:
            self.lbl_warn_val.setStyleSheet("font-size: 22px; font-weight: 700; color: #4CC674;")

        if event.should_alert:
            self._add_event(f"⚠️ {event.message}")
            if self.current_session_id and self.settings.save_event_history:
                self.session_logger.log_event(
                    self.current_session_id,
                    f"WARNING_{event.warning_level}",
                    event.message,
                    metrics={"warning_level": event.warning_level, "issue": event.issue_description},
                )
            if self.settings.enable_tray_notifications:
                self._show_tray_message(
                    f"Posture Warning ({event.warning_level}/{self.settings.max_warnings})",
                    event.message,
                    QtWidgets.QSystemTrayIcon.MessageIcon.Warning,
                )
            if self.settings.enable_sound_alerts:
                try:
                    import winsound
                    winsound.MessageBeep(winsound.MB_ICONEXCLAMATION)
                except Exception:
                    pass

        # Handle full-screen overlay activation
        if event.should_show_overlay:
            self._add_event("🚨 OVERLAY ESCALATION: 3 warnings ignored. Full-screen overlay triggered.")
            if self.current_session_id and self.settings.save_event_history:
                self.session_logger.log_event(
                    self.current_session_id,
                    "OVERLAY_TRIGGERED",
                    "Full-screen overlay escalation displayed.",
                    metrics={"issue": event.issue_description},
                )
            if self.settings.enable_tray_notifications:
                self._show_tray_message(
                    "Posture Escalation",
                    "3 warnings ignored! Full-screen posture overlay active.",
                    QtWidgets.QSystemTrayIcon.MessageIcon.Critical,
                )
            self.overlay.show_overlay(issue=event.issue_description or "Sustained poor posture")

        # Update live recovery state on the overlay if visible
        if self.overlay and self.overlay.isVisible():
            if event.correction_hold_duration > 0.0:
                self.overlay.set_holding_state(
                    event.correction_hold_duration,
                    target_hold_sec=self.settings.correction_confirmation_sec,
                )
            elif not event.should_dismiss_overlay and event.state != ReminderState.CORRECTION_CONFIRMED:
                self.overlay.set_warning_state(issue=event.issue_description or "Sustained poor posture")

        # Dismiss overlay when posture correction is sustained and confirmed
        if event.should_dismiss_overlay and self.overlay and self.overlay.isVisible():
            self.overlay.confirm_and_dismiss()

        if event.state == ReminderState.CORRECTION_CONFIRMED:
            self._add_event("✅ Posture corrected! Warning escalation reset.")
            if self.current_session_id and self.settings.save_event_history:
                self.session_logger.log_event(
                    self.current_session_id,
                    "CORRECTION_CONFIRMED",
                    "Posture successfully corrected and verified.",
                )
            if self.settings.enable_tray_notifications:
                self._show_tray_message(
                    "Posture Corrected!",
                    "Great upright posture! Warning sequence has been reset.",
                    QtWidgets.QSystemTrayIcon.MessageIcon.Information,
                )
            if self.settings.enable_sound_alerts:
                try:
                    import winsound
                    winsound.MessageBeep(winsound.MB_OK)
                except Exception:
                    pass


    def _on_overlay_dismissed(self, reason: str) -> None:
        """Handle manual emergency dismissal or recovery dismissal of overlay."""
        if reason == "emergency":
            if self.worker:
                self.worker.reminder_mgr.emergency_dismiss()
            self._add_event("Overlay dismissed via emergency user override.")
            self.lbl_warn_val.setText(f"0 / {self.settings.max_warnings}")
            self.lbl_warn_val.setStyleSheet("font-size: 22px; font-weight: 700; color: #4CC674;")
            if self.current_session_id and self.settings.save_event_history:
                self.session_logger.log_event(
                    self.current_session_id,
                    "EMERGENCY_DISMISSAL",
                    "Overlay dismissed via keyboard shortcut override.",
                )
        else:
            self._add_event("Overlay dismissed: Posture successfully corrected.")
            if self.current_session_id and self.settings.save_event_history:
                self.session_logger.log_event(
                    self.current_session_id,
                    "OVERLAY_DISMISSED",
                    "Overlay dismissed following confirmed posture correction.",
                )


    def _on_power_state_changed(self, power_state: PowerState) -> None:
        """Handle dynamic power transitions from the background worker."""
        if power_state.mode == PowerMode.AC_CONTINUOUS:
            self.power_badge.setText(f"⚡ AC Power ({power_state.percent}% - Continuous)")
            self.power_badge.setStyleSheet("""
                background-color: #1C3322; color: #4CC674; border: 1px solid #285434;
                border-radius: 12px; padding: 4px 12px; font-size: 12px; font-weight: 600;
            """)
            self._add_event(f"Power: Connected to AC ({power_state.percent}%). Continuous monitoring active.")
        else:
            self.power_badge.setText(f"🔋 Battery {power_state.percent}% (5-min Burst Mode)")
            self.power_badge.setStyleSheet("""
                background-color: #382A10; color: #F5A623; border: 1px solid #5A421A;
                border-radius: 12px; padding: 4px 12px; font-size: 12px; font-weight: 600;
            """)
            self._add_event(f"Power: Running on battery ({power_state.percent}%). Switched to 5-min burst checks.")

    def _on_frame_ready(self, q_img: QtGui.QImage) -> None:
        """Render frame to viewport."""
        scaled = q_img.scaled(
            self.video_viewport.size(),
            QtCore.Qt.AspectRatioMode.KeepAspectRatio,
            QtCore.Qt.TransformationMode.SmoothTransformation,
        )
        self.video_viewport.setPixmap(QtGui.QPixmap.fromImage(scaled))

    def _on_camera_mode_changed(self, mode_str: str) -> None:
        self.camera_badge.setText(f"Mode: {mode_str}")
        if "Live" in mode_str:
            self.camera_badge.setStyleSheet("background-color: #107C41; color: #FFF; border-radius: 12px; padding: 4px 12px;")
        else:
            self.camera_badge.setStyleSheet("background-color: #0078D4; color: #FFF; border-radius: 12px; padding: 4px 12px;")

    def _on_status_changed(self, msg: str) -> None:
        self.statusBar().showMessage(msg, 5000)

    def _on_sim_changed(self, index: int) -> None:
        """Map combo box selection to simulation state."""
        states = [
            PostureState.GOOD,
            PostureState.SLOUCHING,
            PostureState.FORWARD_HEAD,
            PostureState.SHOULDER_MISALIGNMENT,
            PostureState.LEANING,
            PostureState.TOO_CLOSE,
            PostureState.NO_PERSON,
        ]
        if 0 <= index < len(states):
            target = states[index]
            if self.worker:
                self.worker.set_simulation_state(target)
            self._add_event(f"Simulation steered to {target.value}")

    def _add_event(self, text: str) -> None:
        ts = datetime.now().strftime("%H:%M:%S")
        self.list_events.insertItem(0, f"[{ts}] {text}")
        if self.list_events.count() > 50:
            self.list_events.takeItem(50)

    def _open_settings(self) -> None:
        dialog = SettingsDialog(self.settings, self)
        dialog.settings_saved.connect(self._on_settings_saved)
        dialog.exec()

    def _on_settings_saved(self, new_settings: AppSettings) -> None:
        self.settings = new_settings
        if self.worker:
            self.worker.update_settings(new_settings)
        self._add_event("Settings updated.")

    def _setup_system_tray(self) -> None:
        """Initialize native Windows 11 system notification tray icon and context menu."""
        self.tray_icon = QtWidgets.QSystemTrayIcon(self)
        icon = self.style().standardIcon(QtWidgets.QStyle.StandardPixmap.SP_ComputerIcon)
        self.tray_icon.setIcon(icon)
        self.tray_icon.setToolTip("Smart Posture and Ergonomics Tracker")

        tray_menu = QtWidgets.QMenu(self)

        act_show = QtGui.QAction("Open Dashboard", self)
        act_show.triggered.connect(self._restore_from_tray)
        tray_menu.addAction(act_show)

        act_toggle = QtGui.QAction("Start / Stop Monitoring", self)
        act_toggle.triggered.connect(self._toggle_monitoring)
        tray_menu.addAction(act_toggle)

        act_test_overlay = QtGui.QAction("🚨 Test Blur Overlay Now (F12)", self)
        act_test_overlay.triggered.connect(self._force_test_overlay)
        tray_menu.addAction(act_test_overlay)

        tray_menu.addSeparator()

        act_quit = QtGui.QAction("Exit Application", self)
        act_quit.triggered.connect(self.close)
        tray_menu.addAction(act_quit)

        self.tray_icon.setContextMenu(tray_menu)
        self.tray_icon.activated.connect(self._on_tray_activated)
        self.tray_icon.show()

    def _on_tray_activated(self, reason: QtWidgets.QSystemTrayIcon.ActivationReason) -> None:
        if reason in (
            QtWidgets.QSystemTrayIcon.ActivationReason.Trigger,
            QtWidgets.QSystemTrayIcon.ActivationReason.DoubleClick,
        ):
            self._restore_from_tray()

    def _restore_from_tray(self) -> None:
        self.showNormal()
        self.activateWindow()

    def _show_tray_message(
        self,
        title: str,
        message: str,
        icon: QtWidgets.QSystemTrayIcon.MessageIcon = QtWidgets.QSystemTrayIcon.MessageIcon.Information,
        duration_ms: int = 4000,
    ) -> None:
        """Display native Windows Notification Tray popup toast."""
        if hasattr(self, "tray_icon") and self.tray_icon and self.tray_icon.isVisible():
            self.tray_icon.showMessage(title, message, icon, duration_ms)

    def _force_test_overlay(self) -> None:
        """Immediately trigger the full-screen soft-blur overlay for testing."""
        self._add_event("🚨 MANUAL TEST: Force-triggering full-screen soft-blur overlay.")
        if self.settings.enable_tray_notifications:
            self._show_tray_message(
                "Posture Overlay Test Active",
                "Testing full-screen soft-blur overlay! Press Esc to dismiss or hold upright posture.",
                QtWidgets.QSystemTrayIcon.MessageIcon.Warning,
            )
        if self.current_session_id and self.settings.save_event_history:
            self.session_logger.log_event(
                self.current_session_id,
                "OVERLAY_TEST_TRIGGERED",
                "Full-screen overlay triggered manually for user testing.",
            )
        self.overlay.show_overlay(issue="Slouching / Forward Leaning (Test Mode)")

    def _reconnect_camera(self) -> None:
        """Scan available camera devices and reconnect live webcam."""
        available = find_available_cameras(max_to_test=6)
        if available:
            # If phone or external webcam is connected, it is usually at index 1 or above
            chosen = available[-1] if len(available) > 1 else available[0]
            self.settings.camera_index = chosen
            self._add_event(f"Detected camera devices: {available}. Selected Camera {chosen}.")
        else:
            self._add_event("Scan: No physical cameras detected. Ensure phone USB or Wi-Fi driver is active.")

        if self.is_monitoring:
            self._stop_monitoring()
            self._start_monitoring()
            if self.worker:
                self.worker.trigger_immediate_burst()
            self._add_event("Monitoring restarted with newly scanned camera.")
        else:
            self._add_event("Camera updated. Click 'Start Monitoring' to begin.")

    def _check_posture_now(self) -> None:
        """Trigger an on-demand posture check (especially handy on battery mode)."""
        if not self.is_monitoring:
            self._start_monitoring()
        if self.worker:
            self.worker.trigger_immediate_burst()
            self._add_event("Requested on-demand posture sample check.")

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        """Clean shutdown on window close."""
        self._stop_monitoring()
        if hasattr(self, "tray_icon") and self.tray_icon:
            self.tray_icon.hide()
        if hasattr(self, "overlay") and self.overlay:
            self.overlay.close()
        event.accept()


