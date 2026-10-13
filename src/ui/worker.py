"""
Background worker thread for camera capture and posture evaluation.

Keeps CPU-intensive OpenCV and MediaPipe inference completely off the PySide6 UI thread.
Provides automatic fallback to synthetic demo simulation when no physical webcam is present.
"""
import logging
import time
from typing import Optional
import cv2
import numpy as np

from PySide6 import QtCore, QtGui

from src.core.calibration import CalibrationProfile, CalibrationAccumulator
from src.core.mock_generator import MockPostureGenerator
from src.core.posture_engine import PostureEngine, PostureAssessment, PostureState
from src.core.reminder_manager import ReminderManager, ReminderEvent
from src.ui.settings import AppSettings
from src.system.power_manager import PowerManager, PowerMode, PowerState
from src.vision.camera import CameraManager, CameraUnavailableError, find_available_cameras
from src.vision.pose_detector import PoseDetector, ExtractedPose

logger = logging.getLogger(__name__)


class PostureWorker(QtCore.QThread):
    """
    Dedicated QThread executing camera capture, MediaPipe pose detection,
    and PostureEngine assessment at a controlled inference frame rate.
    """

    frame_ready = QtCore.Signal(QtGui.QImage)
    assessment_ready = QtCore.Signal(object)  # Emits PostureAssessment
    reminder_event = QtCore.Signal(object)    # Emits ReminderEvent
    power_state_changed = QtCore.Signal(object)  # Emits PowerState
    calibration_progress = QtCore.Signal(float, int)  # progress fraction, current count
    calibration_finished = QtCore.Signal(object)  # Emits CalibrationProfile
    status_changed = QtCore.Signal(str)
    camera_mode_changed = QtCore.Signal(str)  # "Live Webcam" vs "Demo Simulation"
    error_occurred = QtCore.Signal(str)

    def __init__(self, settings: AppSettings, parent: Optional[QtCore.QObject] = None) -> None:
        super().__init__(parent)
        self.settings = settings
        self._is_running = False
        self._is_calibrating = False

        # Core engines
        self.engine = PostureEngine()
        self.accumulator = CalibrationAccumulator(target_samples=30)
        self.mock_gen = MockPostureGenerator()
        self.reminder_mgr = ReminderManager(
            persistence_threshold_sec=self.settings.persistence_threshold_sec,
            reminder_interval_sec=self.settings.reminder_interval_sec,
            max_warnings=self.settings.max_warnings,
            correction_confirmation_sec=self.settings.correction_confirmation_sec,
        )
        self.power_mgr = PowerManager(
            battery_check_interval_sec=float(self.settings.battery_check_interval_sec),
            burst_duration_sec=float(self.settings.burst_duration_sec),
        )

        # Hardware handles
        self.camera: Optional[CameraManager] = None
        self.detector: Optional[PoseDetector] = None

        # Simulation override state for testing
        self.demo_sim_state = PostureState.GOOD
        self._force_burst_requested = False

    def trigger_immediate_burst(self) -> None:
        """Trigger an immediate posture sample burst on battery without waiting for timer."""
        self._force_burst_requested = True

    def start_calibration(self) -> None:
        """Trigger a fresh 30-sample calibration accumulation window."""
        self.accumulator.reset()
        self._is_calibrating = True
        self.status_changed.emit("Calibrating baseline... Please sit comfortably upright.")


    def set_simulation_state(self, state: PostureState) -> None:
        """Allow manually steering the simulation state for UI testing."""
        self.demo_sim_state = state

    def update_settings(self, new_settings: AppSettings) -> None:
        """Apply updated configuration parameters."""
        self.settings = new_settings
        self.reminder_mgr.persistence_threshold_sec = new_settings.persistence_threshold_sec
        self.reminder_mgr.reminder_interval_sec = new_settings.reminder_interval_sec
        self.reminder_mgr.max_warnings = new_settings.max_warnings
        self.reminder_mgr.correction_confirmation_sec = new_settings.correction_confirmation_sec
        self.power_mgr.battery_check_interval_sec = float(new_settings.battery_check_interval_sec)
        self.power_mgr.burst_duration_sec = float(new_settings.burst_duration_sec)

    def _emit_preview_frame(self, frame: np.ndarray) -> None:
        """Helper to convert and emit QImage preview for UI viewport."""
        if not self.settings.show_camera_preview:
            return
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        q_img = QtGui.QImage(
            rgb_frame.data,
            w,
            h,
            bytes_per_line,
            QtGui.QImage.Format_RGB888,
        ).copy()
        self.frame_ready.emit(q_img)

    def _process_frame_and_pose(
        self,
        frame: Optional[np.ndarray],
        pose: Optional[ExtractedPose],
    ) -> None:
        """Evaluate posture frame through engine and reminder escalation."""
        if self._is_calibrating:
            if pose is not None:
                accepted = self.accumulator.add_sample(pose)
                if accepted:
                    self.calibration_progress.emit(
                        self.accumulator.progress_fraction,
                        self.accumulator.sample_count,
                    )
                    if self.accumulator.is_complete:
                        self._is_calibrating = False
                        profile = self.accumulator.compute_profile()
                        self.engine.set_calibration(profile)
                        self.calibration_finished.emit(profile)
                        self.status_changed.emit("Calibration completed successfully!")
                        # If on battery, release camera now that calibration is finished
                        if self.power_mgr.get_current_power_state().mode == PowerMode.BATTERY_SAVER:
                            if self.camera and self.camera.is_open:
                                self.camera.close()
        else:
            assessment = self.engine.evaluate(pose)
            self.assessment_ready.emit(assessment)

            # Process reminder escalation engine
            rem_event = self.reminder_mgr.process_assessment(assessment)
            self.reminder_event.emit(rem_event)

        if frame is not None:
            self._emit_preview_frame(frame)

    def _initialize_camera(self) -> bool:
        """
        Probe and connect to the camera hardware (physical or phone/virtual).
        Tries primary configured index/URL; if unavailable, auto-scans indices 0..5.
        """
        source = self.settings.ip_camera_url if self.settings.ip_camera_url else self.settings.camera_index
        target_fps = self.settings.target_fps

        # 1. Try primary source
        try:
            cam = CameraManager(camera_index=source, fps=target_fps)
            cam.open()
            self.camera = cam
            logger.info("Successfully connected to primary camera source %s", source)
            return True
        except Exception as e:
            logger.warning("Primary camera source %s failed: %s", source, e)

        # 2. Auto-detect across other devices (e.g. phone at index 1 or 2)
        if self.settings.auto_detect_camera:
            available = find_available_cameras(max_to_test=6)
            logger.info("Auto-detecting cameras. Detected: %s", available)
            for cand_idx in available:
                if cand_idx != source:
                    try:
                        logger.info("Probing auto-detected camera index %d...", cand_idx)
                        candidate_cam = CameraManager(camera_index=cand_idx, fps=target_fps)
                        candidate_cam.open()
                        self.camera = candidate_cam
                        self.settings.camera_index = cand_idx
                        logger.info("Auto-detected and connected to camera index %d (Phone/External)!", cand_idx)
                        return True
                    except Exception as err:
                        logger.debug("Candidate %d failed: %s", cand_idx, err)
                        continue
        return False

    def _execute_battery_burst(self, use_demo: bool) -> None:
        """
        Execute a short 3-second posture sample on battery,
        evaluates posture locally in RAM, then immediately releases webcam hardware.
        """
        logger.info("Executing battery burst check (duration: %.1fs)", self.power_mgr.burst_duration_sec)
        burst_end = time.time() + self.power_mgr.burst_duration_sec

        if not use_demo:
            try:
                if not self.camera or not self.camera.is_open:
                    if self.camera:
                        try:
                            self.camera.open()
                        except Exception:
                            self._initialize_camera()
                    else:
                        self._initialize_camera()

                while time.time() < burst_end and self._is_running:
                    if self.camera and self.camera.is_open:
                        frame = self.camera.read_frame()
                        pose = self.detector.detect(frame) if (frame is not None and self.detector) else None
                        self._process_frame_and_pose(frame, pose)
                    time.sleep(1.0 / max(1, self.settings.target_fps))
            except Exception as e:
                logger.error("Error during battery burst capture: %s", e)
            finally:
                # MANDATORY: Release camera hardware immediately after burst!
                if self.camera and self.camera.is_open:
                    self.camera.close()
                    logger.debug("Camera hardware released following battery burst check.")
        else:
            # Synthetic simulation burst
            while time.time() < burst_end and self._is_running:
                pose = self.mock_gen.generate_pose(self.demo_sim_state)
                frame = self.mock_gen.generate_frame(self.demo_sim_state)
                self._process_frame_and_pose(frame, pose)
                time.sleep(1.0 / max(1, self.settings.target_fps))

    def run(self) -> None:
        """Thread execution loop with adaptive AC continuous / Battery burst switching."""
        self._is_running = True
        use_demo = self.settings.demo_mode

        # Check initial power state
        current_power = self.power_mgr.get_current_power_state()
        self.power_state_changed.emit(current_power)
        last_power_poll_time = time.time()
        last_battery_burst_time = 0.0

        # Try initializing hardware detector if not in forced demo mode
        if not use_demo:
            try:
                self.detector = PoseDetector()
                success = self._initialize_camera()
                if not success:
                    raise CameraUnavailableError("No usable camera found during initial probe.")

                cam_desc = f"Camera {self.settings.camera_index}" if not self.settings.ip_camera_url else "Phone IP Camera"

                if current_power.mode == PowerMode.AC_CONTINUOUS:
                    self.camera_mode_changed.emit(f"{cam_desc} (Live Continuous)")
                    self.status_changed.emit(f"Monitoring active ({cam_desc})")
                else:
                    # Battery mode: run initial verification burst immediately so user sees video right away!
                    self.camera_mode_changed.emit(f"{cam_desc} (Battery Saver)")
                    self.status_changed.emit(f"Battery Mode: Sampling initial 3s check with {cam_desc}...")
                    self._execute_battery_burst(use_demo=False)
                    last_battery_burst_time = time.time()
                    self.status_changed.emit(
                        f"Battery Saver Active ({current_power.percent}%): Initial check done with {cam_desc}. Camera sleeping to save power."
                    )
            except (CameraUnavailableError, Exception) as e:
                logger.warning("Camera unavailable (%s). Activating Demo Simulation mode.", e)
                use_demo = True
                if self.camera:
                    self.camera.close()
                self.camera = None
                self.camera_mode_changed.emit("Demo Simulation (No Camera)")
                self.status_changed.emit("Monitoring active (Demo Simulation Mode)")
        else:
            self.camera_mode_changed.emit("Demo Simulation")
            self.status_changed.emit("Monitoring active (Demo Simulation Mode)")



        interval = 1.0 / max(1, self.settings.target_fps)

        while self._is_running:
            now = time.time()

            # Poll for AC / Battery power transitions every 2 seconds
            if now - last_power_poll_time >= 2.0:
                last_power_poll_time = now
                transition = self.power_mgr.poll()
                if transition is not None:
                    current_power = transition
                    self.power_state_changed.emit(transition)
                    if transition.mode == PowerMode.BATTERY_SAVER:
                        self.status_changed.emit(
                            f"Switched to Battery Saver ({transition.percent}%): Periodic 5-min burst checks."
                        )
                        # Release camera on switch to battery
                        if self.camera and self.camera.is_open:
                            self.camera.close()
                    else:
                        self.status_changed.emit(
                            f"Connected to AC Power ({transition.percent}%): Continuous 3-5 FPS monitoring resumed."
                        )
                        # Re-open camera on switch to AC
                        if not use_demo and self.camera and not self.camera.is_open:
                            try:
                                self.camera.open()
                            except Exception as e:
                                logger.warning("Could not re-open camera on AC transition: %s", e)

            # ----------------------------------------------------------------
            # Mode A: Battery Saver Mode (Periodic 5-minute sampling burst)
            # ----------------------------------------------------------------
            if current_power.mode == PowerMode.BATTERY_SAVER and not self._is_calibrating:
                time_since_burst = now - last_battery_burst_time
                if not self._force_burst_requested and time_since_burst < self.power_mgr.battery_check_interval_sec:
                    # Idle sleep between burst checks; camera remains closed
                    time.sleep(0.25)
                    continue

                self._force_burst_requested = False
                # Run scheduled or on-demand 3-second posture sample burst
                self.status_changed.emit("Battery Mode: Sampling 3-second posture burst...")
                self._execute_battery_burst(use_demo)
                last_battery_burst_time = time.time()
                mins = int(self.power_mgr.battery_check_interval_sec // 60)
                self.status_changed.emit(
                    f"Battery Mode: Posture checked. Camera sleeping to save power (Next check in {mins}m or click 'Check Now')."
                )
                continue

            # ----------------------------------------------------------------
            # Mode B: AC Power Mode (or Active Calibration)
            # ----------------------------------------------------------------
            loop_start = time.time()
            frame: Optional[np.ndarray] = None
            pose: Optional[ExtractedPose] = None

            if not use_demo and self.camera:
                if not self.camera.is_open:
                    try:
                        self.camera.open()
                    except Exception:
                        pass
                if self.camera.is_open:
                    try:
                        frame = self.camera.read_frame()
                        if frame is not None and self.detector:
                            pose = self.detector.detect(frame)
                    except Exception as e:
                        logger.error("Error during camera read/detect: %s", e)
                        self.error_occurred.emit(str(e))
                        use_demo = True
            else:
                pose = self.mock_gen.generate_pose(self.demo_sim_state)
                frame = self.mock_gen.generate_frame(self.demo_sim_state)


            self._process_frame_and_pose(frame, pose)

            # Pace continuous inference rate
            elapsed = time.time() - loop_start
            sleep_time = max(0.005, interval - elapsed)
            time.sleep(sleep_time)

        # Cleanup on exit
        if self.camera:
            self.camera.close()
            self.camera = None
        if self.detector:
            self.detector.close()
            self.detector = None
        logger.info("PostureWorker thread terminated cleanly.")

    def stop(self) -> None:
        """Signal thread to cease execution and await termination."""
        self._is_running = False
        self.reminder_mgr.reset()
        self.wait(2000)
