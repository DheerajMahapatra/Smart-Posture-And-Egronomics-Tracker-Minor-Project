"""
Webcam capture management module for Smart Posture and Ergonomics Tracker.

Provides thread-safe, privacy-respecting webcam access:
- Frames are analyzed locally in volatile memory and NEVER written to disk by default.
- Supports continuous streaming and transient burst capture (3-5s) with automatic camera release.
- Gracefully handles missing, busy, or disconnected webcams without crashing.
"""
import time
import logging
from typing import Generator, List, Optional, Tuple, Union
import cv2
import numpy as np

logger = logging.getLogger(__name__)


class CameraError(Exception):
    """Base exception for camera-related errors."""
    pass


class CameraUnavailableError(CameraError):
    """Raised when the requested camera index cannot be opened or accessed."""
    pass


class CameraManager:
    """
    Manages OpenCV VideoCapture lifecycle for continuous or burst posture analysis.
    Ensures camera hardware is released promptly when not actively in use.
    Supports physical webcams, virtual phone cameras (DroidCam, Iriun, Camo),
    and network IP webcam streams.
    """

    def __init__(
        self,
        camera_index: Union[int, str] = 0,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
    ) -> None:
        self.camera_index = camera_index
        self.target_width = width
        self.target_height = height
        self.target_fps = fps
        self._cap: Optional[cv2.VideoCapture] = None
        self._is_open = False

    @property
    def is_open(self) -> bool:
        """Check if camera device is currently opened."""
        return self._is_open and self._cap is not None and self._cap.isOpened()

    def open(self) -> None:
        """
        Open the camera device and configure resolution.
        Raises CameraUnavailableError if the hardware device is not accessible.
        """
        if self.is_open:
            return

        logger.debug("Opening webcam source %s", self.camera_index)

        # Handle IP Camera / network streams (http:// or rtsp://)
        if isinstance(self.camera_index, str) and (
            self.camera_index.startswith("http://")
            or self.camera_index.startswith("https://")
            or self.camera_index.startswith("rtsp://")
        ):
            self._cap = cv2.VideoCapture(self.camera_index)
        else:
            # Device Index (physical or virtual phone webcam)
            try:
                idx = int(self.camera_index)
            except (ValueError, TypeError):
                idx = 0

            # On Windows, try CAP_DSHOW first (optimal for virtual phone webcams like DroidCam/Iriun)
            cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
            if not cap or not cap.isOpened():
                if cap:
                    cap.release()
                # Fallback to CAP_ANY
                cap = cv2.VideoCapture(idx, cv2.CAP_ANY)

            self._cap = cap

        if not self._cap or not self._cap.isOpened():
            self._cap = None
            self._is_open = False
            raise CameraUnavailableError(
                f"Webcam at index/source {self.camera_index} is unavailable, disconnected, or in use by another application."
            )

        # Apply desired lightweight capture resolution
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.target_width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.target_height)
        self._cap.set(cv2.CAP_PROP_FPS, self.target_fps)
        self._is_open = True
        logger.info(
            "Webcam %s opened successfully (%dx%d)",
            str(self.camera_index),
            self.target_width,
            self.target_height,
        )


    def close(self) -> None:
        """Release the camera device immediately."""
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception as e:
                logger.warning("Error releasing camera %d: %s", self.camera_index, e)
            finally:
                self._cap = None
                self._is_open = False
                logger.debug("Webcam index %d released.", self.camera_index)

    def read_frame(self) -> Optional[np.ndarray]:
        """
        Capture a single BGR frame in memory.
        Returns None if frame read fails or camera is not open.
        """
        if not self.is_open or self._cap is None:
            return None

        ret, frame = self._cap.read()
        if not ret or frame is None:
            logger.warning("Failed to grab frame from camera %d", self.camera_index)
            return None
        return frame

    def capture_burst(
        self,
        duration_seconds: float = 3.0,
        target_fps: float = 4.0,
        warmup_frames: int = 5,
    ) -> List[np.ndarray]:
        """
        Opens camera, discards initial auto-exposure warmup frames, captures a burst
        of frames in-memory over the given duration, and immediately releases the camera.
        Ideal for battery-saving mode.
        """
        frames: List[np.ndarray] = []
        interval = 1.0 / max(target_fps, 1.0)
        start_time = time.time()
        was_already_open = self.is_open

        try:
            if not was_already_open:
                self.open()

            # Discard warmup frames for camera auto-focus/auto-exposure stabilization
            for _ in range(warmup_frames):
                self.read_frame()

            burst_start = time.time()
            next_sample_time = burst_start

            while (time.time() - burst_start) < duration_seconds:
                now = time.time()
                if now >= next_sample_time:
                    frame = self.read_frame()
                    if frame is not None:
                        frames.append(frame.copy())
                    next_sample_time = now + interval
                time.sleep(0.01)  # Yield CPU briefly

            logger.info(
                "Burst capture completed: %d frames acquired over %.2fs",
                len(frames),
                time.time() - burst_start,
            )
            return frames
        finally:
            # If camera was opened solely for this burst, release it immediately
            if not was_already_open:
                self.close()

    def __enter__(self) -> "CameraManager":
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


def is_phone_or_external_source(source: Union[int, str]) -> bool:
    """
    Determine if a camera source is an external/phone webcam rather than the internal laptop webcam.
    - IP streams (http://, https://, rtsp://) are phone/network webcams.
    - Device indices >= 1 are external USB webcams or virtual phone webcam drivers (DroidCam, Iriun, Camo).
    - Device index 0 is typically the internal laptop integrated webcam.
    """
    if isinstance(source, str):
        s_lower = source.strip().lower()
        if (
            s_lower.startswith("http://")
            or s_lower.startswith("https://")
            or s_lower.startswith("rtsp://")
        ):
            return True
        try:
            return int(source) >= 1
        except (ValueError, TypeError):
            return False
    return isinstance(source, int) and source >= 1


def get_camera_source_description(source: Union[int, str]) -> str:
    """Return user-friendly label for camera source."""
    if is_phone_or_external_source(source):
        if isinstance(source, str) and (source.startswith("http://") or source.startswith("rtsp://")):
            return "📱 Phone IP Webcam"
        return f"📱 Phone/External Webcam (Index {source})"
    return f"💻 Laptop Internal Camera (Index {source})"


def find_available_cameras(max_to_test: int = 6, prefer_phone: bool = False) -> List[int]:
    """
    Probe device indices to detect connected webcams (physical or virtual phone webcams).
    Checks both DirectShow (CAP_DSHOW) and default backends.
    If prefer_phone is True, external/phone indices (>= 1) are placed before index 0.
    """
    available: List[int] = []
    for idx in range(max_to_test):
        # Test DSHOW first (standard on Windows for phone apps like DroidCam/Iriun)
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if cap is not None and cap.isOpened():
            available.append(idx)
            cap.release()
            continue
        # Fallback test with CAP_ANY
        cap = cv2.VideoCapture(idx, cv2.CAP_ANY)
        if cap is not None and cap.isOpened():
            available.append(idx)
            cap.release()

    if prefer_phone and available:
        # Sort so that external/phone cameras (index >= 1) appear first
        available.sort(key=lambda x: (0 if x >= 1 else 1, x))

    return available

