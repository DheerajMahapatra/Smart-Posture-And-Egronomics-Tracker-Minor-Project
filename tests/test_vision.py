"""
Unit and integration tests for Phase 2 vision pipeline:
- CameraManager lifecycle, burst capture, error handling
- PoseDetector MediaPipe initialization, landmark calculations, and frame inference
"""
import math
import os
from unittest.mock import MagicMock, patch
import numpy as np
import pytest

from src.vision.camera import (
    CameraManager,
    CameraUnavailableError,
    find_available_cameras,
)
from src.vision.pose_detector import (
    PoseDetector,
    ExtractedPose,
    LandmarkPoint,
)


# ============================================================================
# CameraManager Tests
# ============================================================================

def test_camera_manager_initialization():
    """Verify camera configuration defaults and property setters."""
    cam = CameraManager(camera_index=0, width=640, height=480, fps=15)
    assert cam.camera_index == 0
    assert cam.target_width == 640
    assert cam.target_height == 480
    assert cam.target_fps == 15
    assert not cam.is_open


def test_camera_unavailable_error():
    """Verify that an invalid/non-existent camera index raises CameraUnavailableError."""
    # Using an out-of-range index like 9999
    cam = CameraManager(camera_index=9999)
    with pytest.raises(CameraUnavailableError):
        cam.open()
    assert not cam.is_open


def test_camera_read_frame_when_closed():
    """Reading from a closed camera must safely return None without throwing."""
    cam = CameraManager(camera_index=0)
    assert cam.read_frame() is None


def test_camera_mocked_read_and_burst():
    """Test frame reading and burst capture using a mocked OpenCV capture device."""
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.read.return_value = (True, dummy_frame)

    with patch("cv2.VideoCapture", return_value=mock_cap):
        cam = CameraManager(camera_index=0, width=640, height=480)
        with cam:
            assert cam.is_open
            frame = cam.read_frame()
            assert frame is not None
            assert frame.shape == (480, 640, 3)

        assert not cam.is_open
        mock_cap.release.assert_called()

        # Test burst capture automatically manages open/close
        burst_frames = cam.capture_burst(duration_seconds=0.2, target_fps=10, warmup_frames=2)
        assert len(burst_frames) > 0
        assert all(f.shape == (480, 640, 3) for f in burst_frames)
        assert not cam.is_open


# ============================================================================
# PoseDetector & Geometry Calculation Tests
# ============================================================================

def test_pose_detector_initialization_and_close():
    """Verify PoseDetector can load the local model bundle and cleanly close."""
    detector = PoseDetector()
    assert detector.model_path is not None
    assert os.path.exists(detector.model_path)
    # Ensure close does not error
    detector.close()
    assert detector._landmarker is None


def test_pose_detector_blank_frame_returns_none():
    """A blank black image should return None (no person detected)."""
    with PoseDetector() as detector:
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detector.detect(blank)
        assert result is None


def test_pose_detector_none_frame():
    """Passing None or empty array returns None safely."""
    with PoseDetector() as detector:
        assert detector.detect(None) is None
        assert detector.detect(np.array([])) is None


def test_extracted_pose_metrics_calculation():
    """
    Test geometric calculations on ExtractedPose:
    midpoints, shoulder width (camera proximity indicator), and shoulder angle.
    """
    # Create synthetic landmarks
    # Left shoulder at (0.4, 0.5), Right shoulder at (0.6, 0.5) -> perfectly horizontal
    left_sh = LandmarkPoint(x=0.4, y=0.5, z=0.0, visibility=0.95)
    right_sh = LandmarkPoint(x=0.6, y=0.5, z=0.0, visibility=0.95)
    left_ear = LandmarkPoint(x=0.45, y=0.3, z=0.0, visibility=0.9)
    right_ear = LandmarkPoint(x=0.55, y=0.3, z=0.0, visibility=0.9)
    left_eye = LandmarkPoint(x=0.47, y=0.28, z=0.0, visibility=0.92)
    right_eye = LandmarkPoint(x=0.53, y=0.28, z=0.0, visibility=0.92)
    nose = LandmarkPoint(x=0.5, y=0.32, z=0.0, visibility=0.98)

    pose = ExtractedPose(
        nose=nose,
        left_eye=left_eye,
        right_eye=right_eye,
        left_ear=left_ear,
        right_ear=right_ear,
        left_shoulder=left_sh,
        right_shoulder=right_sh,
    )

    # Shoulder midpoint should be exactly (0.5, 0.5, 0.0)
    mid_sh = pose.shoulder_midpoint
    assert math.isclose(mid_sh[0], 0.5, abs_tol=1e-5)
    assert math.isclose(mid_sh[1], 0.5, abs_tol=1e-5)

    # Ear midpoint should be (0.5, 0.3, 0.0)
    mid_ear = pose.ear_midpoint
    assert math.isclose(mid_ear[0], 0.5, abs_tol=1e-5)
    assert math.isclose(mid_ear[1], 0.3, abs_tol=1e-5)

    # Shoulder width should be 0.2
    assert math.isclose(pose.shoulder_width, 0.2, abs_tol=1e-5)

    # Shoulder angle should be 0.0 degrees (horizontal)
    assert math.isclose(pose.shoulder_angle_degrees, 0.0, abs_tol=1e-4)

    # Mean confidence should be > 0.9
    assert pose.mean_confidence > 0.9


def test_shoulder_angle_tilt():
    """Verify shoulder angle calculation when shoulders are misaligned."""
    # Right shoulder higher up on screen (lower y in screen space: 0.45 vs 0.55)
    left_sh = LandmarkPoint(x=0.4, y=0.55, z=0.0)
    right_sh = LandmarkPoint(x=0.6, y=0.45, z=0.0)
    dummy_point = LandmarkPoint(x=0.5, y=0.3, z=0.0)

    pose = ExtractedPose(
        nose=dummy_point,
        left_eye=dummy_point,
        right_eye=dummy_point,
        left_ear=dummy_point,
        right_ear=dummy_point,
        left_shoulder=left_sh,
        right_shoulder=right_sh,
    )

    # dy = -0.1, dx = 0.2 -> angle = atan2(-0.1, 0.2) in degrees ~ -26.565 degrees
    angle = pose.shoulder_angle_degrees
    assert angle < 0.0
    assert math.isclose(angle, math.degrees(math.atan2(-0.1, 0.2)), abs_tol=1e-3)
