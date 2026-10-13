"""
Pretrained MediaPipe Pose Landmark Detection pipeline for Smart Posture Tracker.

Uses the lightweight PoseLandmarker model (pose_landmarker_lite.task) via the
modern MediaPipe Tasks API. Processes frames entirely in volatile memory and
extracts standardized anatomical coordinates for upper-body posture metrics.
"""
from dataclasses import dataclass
import logging
import math
import os
from pathlib import Path
from typing import Any, List, Optional, Tuple
import urllib.request
import cv2
import numpy as np

import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    PoseLandmarker,
    PoseLandmarkerOptions,
    RunningMode,
)

logger = logging.getLogger(__name__)

# Default remote model URL for lightweight pose estimation
DEFAULT_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
)

# Standard MediaPipe landmark indices
LANDMARK_NOSE = 0
LANDMARK_LEFT_EYE_INNER = 1
LANDMARK_LEFT_EYE = 2
LANDMARK_LEFT_EYE_OUTER = 3
LANDMARK_RIGHT_EYE_INNER = 4
LANDMARK_RIGHT_EYE = 5
LANDMARK_RIGHT_EYE_OUTER = 6
LANDMARK_LEFT_EAR = 7
LANDMARK_RIGHT_EAR = 8
LANDMARK_MOUTH_LEFT = 9
LANDMARK_MOUTH_RIGHT = 10
LANDMARK_LEFT_SHOULDER = 11
LANDMARK_RIGHT_SHOULDER = 12
LANDMARK_LEFT_ELBOW = 13
LANDMARK_RIGHT_ELBOW = 14
LANDMARK_LEFT_WRIST = 15
LANDMARK_RIGHT_WRIST = 16
LANDMARK_LEFT_HIP = 23
LANDMARK_RIGHT_HIP = 24


@dataclass(frozen=True)
class LandmarkPoint:
    """Represents a single 3D anatomical landmark."""
    x: float
    y: float
    z: float
    visibility: float = 1.0
    presence: float = 1.0


@dataclass
class ExtractedPose:
    """
    Standardized upper-body landmarks and derived ergonomic anchor points.
    All coordinates (x, y, z) are normalized [0.0, 1.0] relative to frame dimensions.
    """
    nose: LandmarkPoint
    left_eye: LandmarkPoint
    right_eye: LandmarkPoint
    left_ear: LandmarkPoint
    right_ear: LandmarkPoint
    left_shoulder: LandmarkPoint
    right_shoulder: LandmarkPoint
    left_hip: Optional[LandmarkPoint] = None
    right_hip: Optional[LandmarkPoint] = None
    all_landmarks: Optional[List[LandmarkPoint]] = None

    @property
    def shoulder_midpoint(self) -> Tuple[float, float, float]:
        """Calculates 3D midpoint between left and right shoulders."""
        return (
            (self.left_shoulder.x + self.right_shoulder.x) / 2.0,
            (self.left_shoulder.y + self.right_shoulder.y) / 2.0,
            (self.left_shoulder.z + self.right_shoulder.z) / 2.0,
        )

    @property
    def ear_midpoint(self) -> Tuple[float, float, float]:
        """Calculates 3D midpoint between left and right ears."""
        return (
            (self.left_ear.x + self.right_ear.x) / 2.0,
            (self.left_ear.y + self.right_ear.y) / 2.0,
            (self.left_ear.z + self.right_ear.z) / 2.0,
        )

    @property
    def eye_midpoint(self) -> Tuple[float, float, float]:
        """Calculates 3D midpoint between left and right eyes."""
        return (
            (self.left_eye.x + self.right_eye.x) / 2.0,
            (self.left_eye.y + self.right_eye.y) / 2.0,
            (self.left_eye.z + self.right_eye.z) / 2.0,
        )

    @property
    def shoulder_width(self) -> float:
        """
        Calculates 2D Euclidean distance between shoulders in normalized coordinates.
        Functions as a scale-invariant metric for camera proximity estimation.
        """
        dx = self.right_shoulder.x - self.left_shoulder.x
        dy = self.right_shoulder.y - self.left_shoulder.y
        return math.sqrt(dx * dx + dy * dy)

    @property
    def shoulder_angle_degrees(self) -> float:
        """
        Angle of shoulder line relative to horizontal in degrees.
        Zero degrees indicates perfectly level shoulders; sign indicates tilt direction.
        """
        dx = self.right_shoulder.x - self.left_shoulder.x
        dy = self.right_shoulder.y - self.left_shoulder.y
        return math.degrees(math.atan2(dy, dx))

    @property
    def mean_confidence(self) -> float:
        """Average visibility score across key upper-body landmarks."""
        points = [
            self.nose,
            self.left_eye,
            self.right_eye,
            self.left_ear,
            self.right_ear,
            self.left_shoulder,
            self.right_shoulder,
        ]
        return sum(p.visibility for p in points) / len(points)


class PoseDetector:
    """
    Encapsulates MediaPipe Tasks PoseLandmarker for local in-memory frame inference.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.model_path = self._resolve_model_path(model_path)
        self._landmarker: Optional[PoseLandmarker] = None
        self._initialize_landmarker()

    def _resolve_model_path(self, model_path: Optional[str]) -> str:
        """Locates or downloads the lightweight pose landmark model."""
        if model_path and os.path.exists(model_path):
            return str(Path(model_path).resolve())

        # Check default project models directory
        proj_root = Path(__file__).resolve().parent.parent.parent
        default_model = proj_root / "models" / "pose_landmarker_lite.task"
        if default_model.exists():
            return str(default_model)

        # Attempt automatic download into models/
        default_model.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Downloading MediaPipe Lite Pose model to %s...", default_model)
        try:
            urllib.request.urlretrieve(DEFAULT_MODEL_URL, str(default_model))
            logger.info("Pose model downloaded successfully (%d bytes).", default_model.stat().st_size)
            return str(default_model)
        except Exception as e:
            logger.error("Failed to download pose landmarker model: %s", e)
            raise FileNotFoundError(
                f"Pose model bundle not found at {default_model} and download failed: {e}"
            )

    def _initialize_landmarker(self) -> None:
        """Builds the PoseLandmarker instance with CPU delegate."""
        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=self.model_path),
            running_mode=RunningMode.IMAGE,
            min_pose_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
            num_poses=1,
        )
        self._landmarker = PoseLandmarker.create_from_options(options)
        logger.info("PoseLandmarker initialized using model: %s", self.model_path)

    def detect(self, bgr_frame: np.ndarray) -> Optional[ExtractedPose]:
        """
        Performs pose estimation on an in-memory BGR numpy array frame.
        Returns an ExtractedPose instance, or None if no person is detected
        or confidence is below threshold.
        """
        if self._landmarker is None or bgr_frame is None or bgr_frame.size == 0:
            return None

        # Convert OpenCV BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        result = self._landmarker.detect(mp_image)
        if not result.pose_landmarks or len(result.pose_landmarks) == 0:
            return None

        landmarks = result.pose_landmarks[0]
        if len(landmarks) < 25:
            return None

        def to_point(idx: int) -> LandmarkPoint:
            lm = landmarks[idx]
            vis = getattr(lm, "visibility", 1.0)
            pres = getattr(lm, "presence", 1.0)
            # Default to 1.0 if not populated by lite model
            vis = 1.0 if vis is None else float(vis)
            pres = 1.0 if pres is None else float(pres)
            return LandmarkPoint(
                x=float(lm.x),
                y=float(lm.y),
                z=float(lm.z),
                visibility=vis,
                presence=pres,
            )

        pose = ExtractedPose(
            nose=to_point(LANDMARK_NOSE),
            left_eye=to_point(LANDMARK_LEFT_EYE),
            right_eye=to_point(LANDMARK_RIGHT_EYE),
            left_ear=to_point(LANDMARK_LEFT_EAR),
            right_ear=to_point(LANDMARK_RIGHT_EAR),
            left_shoulder=to_point(LANDMARK_LEFT_SHOULDER),
            right_shoulder=to_point(LANDMARK_RIGHT_SHOULDER),
            left_hip=to_point(LANDMARK_LEFT_HIP),
            right_hip=to_point(LANDMARK_RIGHT_HIP),
            all_landmarks=[to_point(i) for i in range(len(landmarks))],
        )

        # Enforce minimum detection confidence
        if pose.mean_confidence < self.min_detection_confidence:
            logger.debug("Detected pose discarded due to low confidence: %.2f", pose.mean_confidence)
            return None

        return pose

    def close(self) -> None:
        """Release underlying MediaPipe / TensorFlow Lite resources."""
        if self._landmarker is not None:
            try:
                self._landmarker.close()
            except Exception as e:
                logger.warning("Error closing PoseLandmarker: %s", e)
            finally:
                self._landmarker = None

    def __enter__(self) -> "PoseDetector":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
