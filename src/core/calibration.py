"""
Calibration module for Smart Posture and Ergonomics Tracker.

Provides personalized posture baselines:
- Records individual upright posture metrics (ear-to-shoulder vertical distance,
  shoulder width, inter-ocular eye distance, and shoulder tilt).
- Calibrates over a continuous multi-frame window for stability.
- Serializes and deserializes calibration profiles for persistent session use.
- Strictly ergonomic tracking; never provides medical diagnostics.
"""
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import json
import logging
import math
import os
from pathlib import Path
from typing import List, Optional
import numpy as np

from src.vision.pose_detector import ExtractedPose

logger = logging.getLogger(__name__)

MEDICAL_DISCLAIMER = (
    "Ergonomic Notice: This application provides automated workplace ergonomic posture tracking "
    "and habit awareness only. It is not a medical device and does not diagnose, treat, or prevent "
    "any spinal, orthopedic, or medical conditions."
)


@dataclass
class CalibrationProfile:
    """
    Personalized upright ergonomic baseline coordinates and distances.
    All distance metrics are in normalized camera frame coordinates.
    """
    ear_to_shoulder_dist: float       # Vertical distance: ear_midpoint.y to shoulder_midpoint.y
    shoulder_width: float             # 2D distance between shoulders (scale proxy)
    eye_distance: float               # 2D distance between eyes (eye-screen proximity proxy)
    shoulder_tilt_angle: float        # Baseline shoulder tilt in degrees
    head_forward_delta: float         # 2D horizontal ear-shoulder offset (craning proxy)
    confidence: float                 # Average confidence during calibration
    created_at: str                   # ISO timestamp string
    disclaimer: str = MEDICAL_DISCLAIMER

    def to_dict(self) -> dict:
        """Convert profile to serializable dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "CalibrationProfile":
        """Instantiate profile from dictionary."""
        # Ensure disclaimer is preserved
        data.setdefault("disclaimer", MEDICAL_DISCLAIMER)
        return cls(**data)

    def save(self, filepath: str) -> None:
        """Save calibration profile to a JSON file."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info("Saved calibration profile to %s", filepath)

    @classmethod
    def load(cls, filepath: str) -> "CalibrationProfile":
        """Load calibration profile from a JSON file."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

    @classmethod
    def default_ergonomic_baseline(cls) -> "CalibrationProfile":
        """
        Sensible statistical default baseline based on typical desk webcam framing
        (used when personal calibration has not yet been performed).
        """
        return cls(
            ear_to_shoulder_dist=0.20,
            shoulder_width=0.35,
            eye_distance=0.07,
            shoulder_tilt_angle=0.0,
            head_forward_delta=0.03,
            confidence=0.90,
            created_at=datetime.now(timezone.utc).isoformat(),
        )


class CalibrationAccumulator:
    """
    Gathers a stream of high-confidence poses over 3-5 seconds to compute
    a stable, jitter-free personal baseline profile.
    """

    def __init__(self, target_samples: int = 30, min_confidence: float = 0.6) -> None:
        self.target_samples = target_samples
        self.min_confidence = min_confidence
        self._samples: List[ExtractedPose] = []

    def reset(self) -> None:
        """Clear current accumulator samples."""
        self._samples.clear()

    @property
    def sample_count(self) -> int:
        """Current number of collected valid samples."""
        return len(self._samples)

    @property
    def progress_fraction(self) -> float:
        """Calibration progress fraction [0.0, 1.0]."""
        return min(1.0, len(self._samples) / max(1, self.target_samples))

    @property
    def is_complete(self) -> bool:
        """Whether sufficient samples have been gathered."""
        return len(self._samples) >= self.target_samples

    def add_sample(self, pose: Optional[ExtractedPose]) -> bool:
        """
        Add a candidate pose to the calibration sample pool.
        Returns True if the sample was accepted, False if rejected or invalid.
        """
        if pose is None:
            return False

        if pose.mean_confidence < self.min_confidence:
            logger.debug("Rejected calibration frame due to low confidence: %.2f", pose.mean_confidence)
            return False

        self._samples.append(pose)
        return True

    def compute_profile(self) -> CalibrationProfile:
        """
        Compute mean baseline metrics from collected samples.
        Raises ValueError if insufficient samples were collected.
        """
        if not self._samples:
            raise ValueError("Cannot compute calibration profile: no samples collected.")

        ear_shoulder_dists = []
        shoulder_widths = []
        eye_distances = []
        shoulder_angles = []
        head_deltas = []
        confidences = []

        for p in self._samples:
            # 1. Vertical ear to shoulder distance (ear is higher on screen -> smaller y)
            ear_mid = p.ear_midpoint
            sh_mid = p.shoulder_midpoint
            vert_dist = max(0.01, sh_mid[1] - ear_mid[1])
            ear_shoulder_dists.append(vert_dist)

            # 2. Shoulder width
            shoulder_widths.append(p.shoulder_width)

            # 3. Inter-ocular eye distance
            dx_eye = p.right_eye.x - p.left_eye.x
            dy_eye = p.right_eye.y - p.left_eye.y
            eye_dist = math.sqrt(dx_eye * dx_eye + dy_eye * dy_eye)
            eye_distances.append(eye_dist)

            # 4. Shoulder tilt
            shoulder_angles.append(p.shoulder_angle_degrees)

            # 5. Head forward offset (horizontal or anterior ear displacement)
            head_delta = abs(ear_mid[0] - sh_mid[0])
            head_deltas.append(head_delta)

            confidences.append(p.mean_confidence)

        profile = CalibrationProfile(
            ear_to_shoulder_dist=float(np.median(ear_shoulder_dists)),
            shoulder_width=float(np.median(shoulder_widths)),
            eye_distance=float(np.median(eye_distances)),
            shoulder_tilt_angle=float(np.median(shoulder_angles)),
            head_forward_delta=float(np.median(head_deltas)),
            confidence=float(np.mean(confidences)),
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        logger.info(
            "Computed personal calibration: ear-sh=%.3f, sh-w=%.3f, eye-dist=%.3f (samples=%d)",
            profile.ear_to_shoulder_dist,
            profile.shoulder_width,
            profile.eye_distance,
            len(self._samples),
        )
        return profile
