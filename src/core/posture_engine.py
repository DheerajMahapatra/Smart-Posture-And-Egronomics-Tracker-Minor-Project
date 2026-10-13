"""
Posture Evaluation Engine for Smart Posture and Ergonomics Tracker.

Evaluates upper-body ergonomic posture in real time:
- Detects slouching (vertical ear-to-shoulder compression)
- Detects forward-head posture (craning: anterior ear displacement)
- Detects shoulder misalignment (shoulder line tilt angle)
- Detects lateral leaning (lateral tilt of the head-torso vertical axis)
- Detects screen proximity (eye-to-screen distance / inter-ocular expansion)
- Applies Exponential Moving Average (EMA) temporal smoothing
- Applies hysteresis thresholds to prevent state flickering
- Strictly non-medical: habit and ergonomic tracking only.
"""
from dataclasses import dataclass
from enum import Enum
import logging
import math
from typing import Dict, List, Optional, Tuple

from src.core.calibration import CalibrationProfile, MEDICAL_DISCLAIMER
from src.vision.pose_detector import ExtractedPose

logger = logging.getLogger(__name__)


class PostureState(str, Enum):
    """Discrete ergonomic posture states."""
    GOOD = "GOOD"
    SLOUCHING = "SLOUCHING"
    FORWARD_HEAD = "FORWARD_HEAD"
    SHOULDER_MISALIGNMENT = "SHOULDER_MISALIGNMENT"
    LEANING = "LEANING"
    TOO_CLOSE = "TOO_CLOSE"
    TOO_FAR = "TOO_FAR"
    NO_PERSON = "NO_PERSON"


@dataclass
class PostureAssessment:
    """Detailed evaluation result for a single processed frame."""
    state: PostureState
    is_good: bool
    primary_issue: str
    confidence: float
    # Continuous normalized metrics
    slouch_compression_ratio: float   # Current / baseline vertical span (< 0.82 implies slouch)
    shoulder_tilt_deg: float          # Deviation from horizontal (degrees)
    forward_head_delta: float         # Horizontal ear-to-shoulder offset
    eye_proximity_ratio: float        # Current / baseline eye distance (> 1.25 implies too close)
    lateral_lean_deg: float           # Lateral deviation of neck/spine axis (degrees)
    disclaimer: str = MEDICAL_DISCLAIMER


class PostureEngine:
    """
    Evaluates ExtractedPose landmarks against personal calibration baselines
    using temporal exponential moving average (EMA) smoothing and hysteresis.
    """

    def __init__(
        self,
        calibration: Optional[CalibrationProfile] = None,
        smoothing_alpha: float = 0.30,
        # Slouching thresholds (ear-to-shoulder height compression relative to baseline)
        slouch_enter_ratio: float = 0.82,  # Enter SLOUCHING when compressed below 82% of baseline
        slouch_exit_ratio: float = 0.90,   # Recover to GOOD when reaching 90% of baseline
        # Forward-head thresholds (ear forward horizontal displacement)
        forward_head_enter_delta: float = 0.08,
        forward_head_exit_delta: float = 0.05,
        # Shoulder misalignment thresholds (degrees)
        shoulder_tilt_enter_deg: float = 7.0,
        shoulder_tilt_exit_deg: float = 4.0,
        # Lateral lean thresholds (degrees from vertical)
        lean_enter_deg: float = 8.5,
        lean_exit_deg: float = 5.0,
        # Eye-to-screen proximity thresholds (inter-ocular distance ratio vs baseline)
        eye_close_enter_ratio: float = 1.28,  # > 28% wider eye distance -> too close to screen
        eye_close_exit_ratio: float = 1.15,
        eye_far_enter_ratio: float = 0.65,    # < 65% of baseline eye distance -> too far back
        eye_far_exit_ratio: float = 0.75,
        min_confidence: float = 0.50,
    ) -> None:
        self.calibration = calibration or CalibrationProfile.default_ergonomic_baseline()
        self.alpha = smoothing_alpha

        # Hysteresis threshold definitions
        self.slouch_enter_ratio = slouch_enter_ratio
        self.slouch_exit_ratio = slouch_exit_ratio
        self.forward_head_enter_delta = forward_head_enter_delta
        self.forward_head_exit_delta = forward_head_exit_delta
        self.shoulder_tilt_enter_deg = shoulder_tilt_enter_deg
        self.shoulder_tilt_exit_deg = shoulder_tilt_exit_deg
        self.lean_enter_deg = lean_enter_deg
        self.lean_exit_deg = lean_exit_deg
        self.eye_close_enter_ratio = eye_close_enter_ratio
        self.eye_close_exit_ratio = eye_close_exit_ratio
        self.eye_far_enter_ratio = eye_far_enter_ratio
        self.eye_far_exit_ratio = eye_far_exit_ratio
        self.min_confidence = min_confidence

        # Smoothed state variables
        self._smoothed_ear_shoulder_dist: Optional[float] = None
        self._smoothed_shoulder_tilt: Optional[float] = None
        self._smoothed_forward_delta: Optional[float] = None
        self._smoothed_eye_distance: Optional[float] = None
        self._smoothed_lean_deg: Optional[float] = None

        # Current hysteresis state
        self._current_state = PostureState.GOOD

    def set_calibration(self, profile: CalibrationProfile) -> None:
        """Update active personal baseline calibration profile."""
        self.calibration = profile
        logger.info("PostureEngine updated with personal calibration profile.")

    def reset_smoothing(self) -> None:
        """Reset internal EMA filters and state."""
        self._smoothed_ear_shoulder_dist = None
        self._smoothed_shoulder_tilt = None
        self._smoothed_forward_delta = None
        self._smoothed_eye_distance = None
        self._smoothed_lean_deg = None
        self._current_state = PostureState.GOOD

    def _ema(self, new_val: float, smoothed_val: Optional[float]) -> float:
        """Apply Exponential Moving Average smoothing."""
        if smoothed_val is None:
            return new_val
        return self.alpha * new_val + (1.0 - self.alpha) * smoothed_val

    def evaluate(self, pose: Optional[ExtractedPose]) -> PostureAssessment:
        """
        Evaluate a single frame's pose against calibrated baselines.
        Returns a strongly typed PostureAssessment.
        """
        if pose is None or pose.mean_confidence < self.min_confidence:
            self._current_state = PostureState.NO_PERSON
            return PostureAssessment(
                state=PostureState.NO_PERSON,
                is_good=False,
                primary_issue="No user detected in camera frame",
                confidence=pose.mean_confidence if pose else 0.0,
                slouch_compression_ratio=1.0,
                shoulder_tilt_deg=0.0,
                forward_head_delta=0.0,
                eye_proximity_ratio=1.0,
                lateral_lean_deg=0.0,
            )

        # --------------------------------------------------------------------
        # 1. Compute raw geometric metrics
        # --------------------------------------------------------------------
        ear_mid = pose.ear_midpoint
        sh_mid = pose.shoulder_midpoint

        # Vertical ear-to-shoulder span (ear.y < shoulder.y in screen space)
        raw_ear_sh_dist = max(0.01, sh_mid[1] - ear_mid[1])

        # Shoulder tilt relative to horizontal (abs degrees)
        raw_sh_tilt = abs(pose.shoulder_angle_degrees - self.calibration.shoulder_tilt_angle)

        # Inter-ocular eye distance (proximity proxy)
        dx_eye = pose.right_eye.x - pose.left_eye.x
        dy_eye = pose.right_eye.y - pose.left_eye.y
        raw_eye_dist = math.sqrt(dx_eye * dx_eye + dy_eye * dy_eye)

        # Lateral leaning: angle of spine/neck vector from vertical (degrees)
        dx_spine = ear_mid[0] - sh_mid[0]
        dy_spine = sh_mid[1] - ear_mid[1]  # positive upward
        raw_lean_deg = abs(math.degrees(math.atan2(dx_spine, max(1e-5, dy_spine))))

        # Forward head craning offset:
        # Measures 3D anterior head craning (in MediaPipe, smaller Z is closer to camera: sh_mid.z - ear_mid.z > 0)
        # For 2D craning, only count horizontal offset if user is not leaning laterally
        z_craning = max(0.0, sh_mid[2] - ear_mid[2])
        x_craning = abs(dx_spine) if raw_lean_deg < self.lean_exit_deg else 0.0
        raw_forward_delta = max(z_craning, x_craning)

        # --------------------------------------------------------------------
        # 2. Apply Exponential Moving Average (EMA) smoothing
        # --------------------------------------------------------------------
        self._smoothed_ear_shoulder_dist = self._ema(raw_ear_sh_dist, self._smoothed_ear_shoulder_dist)
        self._smoothed_shoulder_tilt = self._ema(raw_sh_tilt, self._smoothed_shoulder_tilt)
        self._smoothed_forward_delta = self._ema(raw_forward_delta, self._smoothed_forward_delta)
        self._smoothed_eye_distance = self._ema(raw_eye_dist, self._smoothed_eye_distance)
        self._smoothed_lean_deg = self._ema(raw_lean_deg, self._smoothed_lean_deg)

        # --------------------------------------------------------------------
        # 3. Calculate Normalized Ratios vs Baseline
        # --------------------------------------------------------------------
        base_ear_sh = max(1e-4, self.calibration.ear_to_shoulder_dist)
        base_eye = max(1e-4, self.calibration.eye_distance)

        slouch_ratio = self._smoothed_ear_shoulder_dist / base_ear_sh
        eye_ratio = self._smoothed_eye_distance / base_eye
        sh_tilt = self._smoothed_shoulder_tilt
        fwd_delta = self._smoothed_forward_delta
        lean_deg = self._smoothed_lean_deg

        # --------------------------------------------------------------------
        # 4. Hysteresis State Machine
        # --------------------------------------------------------------------
        candidate_issues: List[Tuple[PostureState, str, float]] = []

        # Check Too Close (Eye Proximity)
        is_too_close = (
            eye_ratio > self.eye_close_enter_ratio
            if self._current_state != PostureState.TOO_CLOSE
            else eye_ratio > self.eye_close_exit_ratio
        )
        if is_too_close:
            candidate_issues.append((PostureState.TOO_CLOSE, "Too close to screen (eye strain risk)", eye_ratio))

        # Check Slouching
        is_slouching = (
            slouch_ratio < self.slouch_enter_ratio
            if self._current_state != PostureState.SLOUCHING
            else slouch_ratio < self.slouch_exit_ratio
        )
        if is_slouching:
            candidate_issues.append((PostureState.SLOUCHING, "Slouching / hunched posture detected", 1.0 - slouch_ratio))

        # Check Forward-Head
        is_forward_head = (
            fwd_delta > self.forward_head_enter_delta
            if self._current_state != PostureState.FORWARD_HEAD
            else fwd_delta > self.forward_head_exit_delta
        )
        if is_forward_head:
            candidate_issues.append((PostureState.FORWARD_HEAD, "Forward-head posture (tech neck)", fwd_delta))

        # Check Shoulder Misalignment
        is_shoulder_misaligned = (
            sh_tilt > self.shoulder_tilt_enter_deg
            if self._current_state != PostureState.SHOULDER_MISALIGNMENT
            else sh_tilt > self.shoulder_tilt_exit_deg
        )
        if is_shoulder_misaligned:
            candidate_issues.append((PostureState.SHOULDER_MISALIGNMENT, "Shoulders uneven / tilted", sh_tilt))

        # Check Lateral Leaning
        is_leaning = (
            lean_deg > self.lean_enter_deg
            if self._current_state != PostureState.LEANING
            else lean_deg > self.lean_exit_deg
        )
        if is_leaning:
            candidate_issues.append((PostureState.LEANING, "Leaning excessively to one side", lean_deg))

        # Check Too Far
        is_too_far = (
            eye_ratio < self.eye_far_enter_ratio
            if self._current_state != PostureState.TOO_FAR
            else eye_ratio < self.eye_far_exit_ratio
        )
        if is_too_far:
            candidate_issues.append((PostureState.TOO_FAR, "Too far from screen / camera", 1.0 - eye_ratio))

        # Prioritize issue: if existing issue is still active, maintain it for hysteresis stability;
        # otherwise pick highest severity candidate
        if not candidate_issues:
            self._current_state = PostureState.GOOD
            primary_issue = "Optimal upright posture"
        else:
            # If current active state is among active issues, keep it
            active_match = next((item for item in candidate_issues if item[0] == self._current_state), None)
            if active_match:
                primary_issue = active_match[1]
            else:
                self._current_state = candidate_issues[0][0]
                primary_issue = candidate_issues[0][1]

        return PostureAssessment(
            state=self._current_state,
            is_good=(self._current_state == PostureState.GOOD),
            primary_issue=primary_issue,
            confidence=pose.mean_confidence,
            slouch_compression_ratio=slouch_ratio,
            shoulder_tilt_deg=sh_tilt,
            forward_head_delta=fwd_delta,
            eye_proximity_ratio=eye_ratio,
            lateral_lean_deg=lean_deg,
        )
