"""
Mock Posture & Landmark Generator for Smart Posture and Ergonomics Tracker.

Enables full application testing and UI development without a physical webcam:
- Generates synthetic ExtractedPose objects for all posture states (Good, Slouch,
  Forward-Head, Shoulder Tilt, Leaning, Too Close / Eye Strain, Too Far).
- Generates rendered 640x480 BGR frames with simulated posture figures for visual UI feeds.
"""
import math
import time
from typing import Optional, Tuple
import cv2
import numpy as np

from src.core.posture_engine import PostureState
from src.vision.pose_detector import ExtractedPose, LandmarkPoint


class MockPostureGenerator:
    """
    Produces deterministic or continuous animated synthetic posture landmarks and frames.
    """

    def __init__(self, baseline_width: float = 0.35, baseline_eye_dist: float = 0.07) -> None:
        self.baseline_width = baseline_width
        self.baseline_eye_dist = baseline_eye_dist
        self._start_time = time.time()

    def generate_pose(
        self,
        state: PostureState = PostureState.GOOD,
        confidence: float = 0.95,
        noise: float = 0.0,
    ) -> Optional[ExtractedPose]:
        """
        Generate a synthetic ExtractedPose corresponding to the requested PostureState.
        """
        if state == PostureState.NO_PERSON:
            return None

        # Center reference point (chest/shoulders)
        center_x = 0.50
        sh_y = 0.55
        sh_w = self.baseline_width
        eye_w = self.baseline_eye_dist

        # Default upright offsets
        ear_y = 0.35  # delta = 0.20
        sh_tilt_deg = 0.0
        ear_offset_x = 0.0
        head_z = 0.0

        if state == PostureState.GOOD:
            pass  # Use standard upright baselines

        elif state == PostureState.SLOUCHING:
            # Compress neck: ears move down toward shoulders
            ear_y = 0.43  # delta shrinks from 0.20 to 0.12 (compressed by 40%)

        elif state == PostureState.FORWARD_HEAD:
            # Head and ears crane forward toward camera (negative Z in MediaPipe)
            head_z = -0.12
            ear_offset_x = 0.0  # Centered horizontally, craning anteriorly

        elif state == PostureState.SHOULDER_MISALIGNMENT:
            # Left shoulder higher than right (tilted by 12 degrees)
            sh_tilt_deg = 12.0

        elif state == PostureState.LEANING:
            # Entire head shifts sideways relative to shoulders (lateral lean)
            center_x = 0.50
            ear_offset_x = 0.10

        elif state == PostureState.TOO_CLOSE:
            # Face moves close to monitor -> eye distance expands by 35%
            eye_w = self.baseline_eye_dist * 1.40
            sh_w = self.baseline_width * 1.30

        elif state == PostureState.TOO_FAR:
            # Slumped far back -> eye distance shrinks
            eye_w = self.baseline_eye_dist * 0.60
            sh_w = self.baseline_width * 0.70

        # Calculate shoulder coordinates with tilt
        tilt_rad = math.radians(sh_tilt_deg)
        half_w = sh_w / 2.0
        sh_dx = half_w * math.cos(tilt_rad)
        sh_dy = half_w * math.sin(tilt_rad)

        left_sh = LandmarkPoint(x=center_x - sh_dx, y=sh_y + sh_dy, z=0.0, visibility=confidence)
        right_sh = LandmarkPoint(x=center_x + sh_dx, y=sh_y - sh_dy, z=0.0, visibility=confidence)

        # Head / face coordinates
        head_cx = center_x + ear_offset_x
        half_eye = eye_w / 2.0
        left_eye = LandmarkPoint(x=head_cx - half_eye, y=ear_y - 0.02, z=head_z, visibility=confidence)
        right_eye = LandmarkPoint(x=head_cx + half_eye, y=ear_y - 0.02, z=head_z, visibility=confidence)

        half_ear = half_eye * 1.8
        left_ear = LandmarkPoint(x=head_cx - half_ear, y=ear_y, z=head_z, visibility=confidence)
        right_ear = LandmarkPoint(x=head_cx + half_ear, y=ear_y, z=head_z, visibility=confidence)

        nose = LandmarkPoint(x=head_cx, y=ear_y + 0.02, z=head_z - 0.02, visibility=confidence)

        return ExtractedPose(
            nose=nose,
            left_eye=left_eye,
            right_eye=right_eye,
            left_ear=left_ear,
            right_ear=right_ear,
            left_shoulder=left_sh,
            right_shoulder=right_sh,
        )

    def generate_frame(
        self,
        state: PostureState = PostureState.GOOD,
        width: int = 640,
        height: int = 480,
    ) -> np.ndarray:
        """
        Render a 640x480 BGR frame visualizing the simulated posture skeleton.
        """
        frame = np.full((height, width, 3), 32, dtype=np.uint8)  # Dark theme background

        pose = self.generate_pose(state)
        if pose is None:
            cv2.putText(
                frame,
                "SIMULATION: No Person in Frame",
                (30, height // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 200),
                2,
            )
            return frame

        def pt(lm: LandmarkPoint) -> Tuple[int, int]:
            return (int(lm.x * width), int(lm.y * height))

        # Color based on state
        color = (80, 220, 80) if state == PostureState.GOOD else (50, 140, 255)

        # Draw shoulder line
        cv2.line(frame, pt(pose.left_shoulder), pt(pose.right_shoulder), color, 4)

        # Draw neck & head axis
        sh_mid = (int(pose.shoulder_midpoint[0] * width), int(pose.shoulder_midpoint[1] * height))
        ear_mid = (int(pose.ear_midpoint[0] * width), int(pose.ear_midpoint[1] * height))
        cv2.line(frame, sh_mid, ear_mid, color, 3)

        # Draw eyes and head circle
        cv2.circle(frame, pt(pose.nose), 8, (255, 255, 255), -1)
        cv2.circle(frame, pt(pose.left_eye), 5, (255, 200, 100), -1)
        cv2.circle(frame, pt(pose.right_eye), 5, (255, 200, 100), -1)
        cv2.circle(frame, pt(pose.left_shoulder), 9, color, -1)
        cv2.circle(frame, pt(pose.right_shoulder), 9, color, -1)

        # Label status
        cv2.putText(
            frame,
            f"SIMULATED: {state.value}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2,
        )
        return frame
