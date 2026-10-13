"""
Unit and algorithmic tests for Phase 3:
- CalibrationProfile and CalibrationAccumulator
- PostureEngine detection of Slouching, Forward Head, Shoulder Tilt, Leaning, and Eye Proximity
- Exponential Moving Average (EMA) smoothing and Hysteresis threshold stability
- MockPostureGenerator synthetic frame and pose generation
"""
import math
import os
import tempfile
import numpy as np
import pytest

from src.core.calibration import (
    CalibrationProfile,
    CalibrationAccumulator,
    MEDICAL_DISCLAIMER,
)
from src.core.posture_engine import (
    PostureEngine,
    PostureState,
    PostureAssessment,
)
from src.core.mock_generator import MockPostureGenerator
from src.vision.pose_detector import ExtractedPose, LandmarkPoint


# ============================================================================
# Calibration Tests
# ============================================================================

def test_calibration_profile_defaults_and_serialization():
    """Verify default baseline profile properties and JSON serialization roundtrip."""
    profile = CalibrationProfile.default_ergonomic_baseline()
    assert profile.ear_to_shoulder_dist > 0.0
    assert profile.shoulder_width > 0.0
    assert profile.eye_distance > 0.0
    assert MEDICAL_DISCLAIMER in profile.disclaimer

    d = profile.to_dict()
    restored = CalibrationProfile.from_dict(d)
    assert restored.ear_to_shoulder_dist == profile.ear_to_shoulder_dist
    assert restored.eye_distance == profile.eye_distance
    assert restored.disclaimer == profile.disclaimer


def test_calibration_profile_save_and_load():
    """Test saving profile to file and loading it back."""
    profile = CalibrationProfile.default_ergonomic_baseline()
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test_calib.json")
        profile.save(path)
        assert os.path.exists(path)

        loaded = CalibrationProfile.load(path)
        assert loaded.ear_to_shoulder_dist == profile.ear_to_shoulder_dist
        assert loaded.shoulder_width == profile.shoulder_width


def test_calibration_accumulator():
    """Test multi-frame accumulation, rejection of low-confidence frames, and profile computation."""
    acc = CalibrationAccumulator(target_samples=5, min_confidence=0.6)
    gen = MockPostureGenerator()

    # Reject None
    assert not acc.add_sample(None)

    # Reject low-confidence pose
    low_conf_pose = gen.generate_pose(PostureState.GOOD, confidence=0.3)
    assert not acc.add_sample(low_conf_pose)
    assert acc.sample_count == 0

    # Add 5 valid good poses
    for _ in range(5):
        pose = gen.generate_pose(PostureState.GOOD, confidence=0.95)
        accepted = acc.add_sample(pose)
        assert accepted

    assert acc.is_complete
    assert acc.progress_fraction == 1.0

    profile = acc.compute_profile()
    assert profile.confidence >= 0.90
    assert profile.ear_to_shoulder_dist > 0.15
    assert profile.eye_distance > 0.05


# ============================================================================
# PostureEngine Metric & State Tests
# ============================================================================

@pytest.fixture
def calibrated_engine():
    """Fixture providing a fresh PostureEngine calibrated with standard baselines."""
    calib = CalibrationProfile(
        ear_to_shoulder_dist=0.20,
        shoulder_width=0.35,
        eye_distance=0.07,
        shoulder_tilt_angle=0.0,
        head_forward_delta=0.0,
        confidence=0.95,
        created_at="2026-10-12T00:00:00Z",
    )
    # Using alpha=1.0 for unit tests to test instantaneous response before testing smoothing
    return PostureEngine(calibration=calib, smoothing_alpha=1.0)


def test_posture_engine_good_posture(calibrated_engine):
    """Calibrated upright pose should evaluate to PostureState.GOOD."""
    gen = MockPostureGenerator()
    pose = gen.generate_pose(PostureState.GOOD)

    assessment = calibrated_engine.evaluate(pose)
    assert assessment.state == PostureState.GOOD
    assert assessment.is_good is True
    assert "Optimal" in assessment.primary_issue


def test_posture_engine_detects_slouching(calibrated_engine):
    """When ear-to-shoulder vertical distance compresses below threshold, detect SLOUCHING."""
    gen = MockPostureGenerator()
    slouch_pose = gen.generate_pose(PostureState.SLOUCHING)

    assessment = calibrated_engine.evaluate(slouch_pose)
    assert assessment.state == PostureState.SLOUCHING
    assert assessment.is_good is False
    assert assessment.slouch_compression_ratio < 0.82
    assert "Slouching" in assessment.primary_issue


def test_posture_engine_detects_forward_head(calibrated_engine):
    """When ears shift forward/anteriorly, detect FORWARD_HEAD."""
    gen = MockPostureGenerator()
    fwd_pose = gen.generate_pose(PostureState.FORWARD_HEAD)

    assessment = calibrated_engine.evaluate(fwd_pose)
    assert assessment.state == PostureState.FORWARD_HEAD
    assert assessment.is_good is False
    assert "Forward-head" in assessment.primary_issue


def test_posture_engine_detects_shoulder_misalignment(calibrated_engine):
    """When shoulders tilt beyond angle threshold, detect SHOULDER_MISALIGNMENT."""
    gen = MockPostureGenerator()
    tilt_pose = gen.generate_pose(PostureState.SHOULDER_MISALIGNMENT)

    assessment = calibrated_engine.evaluate(tilt_pose)
    assert assessment.state == PostureState.SHOULDER_MISALIGNMENT
    assert assessment.is_good is False
    assert assessment.shoulder_tilt_deg > 7.0
    assert "uneven" in assessment.primary_issue.lower()


def test_posture_engine_detects_lateral_leaning(calibrated_engine):
    """When head/torso tilts sideways, detect LEANING."""
    gen = MockPostureGenerator()
    lean_pose = gen.generate_pose(PostureState.LEANING)

    assessment = calibrated_engine.evaluate(lean_pose)
    assert assessment.state == PostureState.LEANING
    assert assessment.is_good is False
    assert assessment.lateral_lean_deg > 8.0
    assert "Leaning" in assessment.primary_issue


def test_posture_engine_detects_eye_too_close_proximity(calibrated_engine):
    """When eyes move very close to the screen, detect TOO_CLOSE (eye strain)."""
    gen = MockPostureGenerator()
    close_pose = gen.generate_pose(PostureState.TOO_CLOSE)

    assessment = calibrated_engine.evaluate(close_pose)
    assert assessment.state == PostureState.TOO_CLOSE
    assert assessment.is_good is False
    assert assessment.eye_proximity_ratio > 1.28
    assert "close" in assessment.primary_issue.lower()


def test_posture_engine_detects_too_far(calibrated_engine):
    """When user is too far from camera, detect TOO_FAR."""
    gen = MockPostureGenerator()
    far_pose = gen.generate_pose(PostureState.TOO_FAR)

    assessment = calibrated_engine.evaluate(far_pose)
    assert assessment.state == PostureState.TOO_FAR
    assert assessment.is_good is False
    assert assessment.eye_proximity_ratio < 0.65


def test_posture_engine_no_person(calibrated_engine):
    """Empty or None frame must yield NO_PERSON."""
    assessment = calibrated_engine.evaluate(None)
    assert assessment.state == PostureState.NO_PERSON
    assert assessment.is_good is False


# ============================================================================
# Hysteresis and Smoothing Tests
# ============================================================================

def test_hysteresis_prevents_state_flapping():
    """
    Test hysteresis:
    - Slouch enters when compression ratio < 0.82
    - Once in SLOUCHING, recovering to 0.85 (between enter 0.82 and exit 0.90) stays in SLOUCHING.
    - Only recovering past 0.90 returns to GOOD.
    """
    calib = CalibrationProfile(
        ear_to_shoulder_dist=0.20,
        shoulder_width=0.35,
        eye_distance=0.07,
        shoulder_tilt_angle=0.0,
        head_forward_delta=0.0,
        confidence=0.95,
        created_at="2026-10-12T00:00:00Z",
    )
    engine = PostureEngine(
        calibration=calib,
        smoothing_alpha=1.0,
        slouch_enter_ratio=0.82,
        slouch_exit_ratio=0.90,
    )

    def make_pose_with_neck_height(h: float):
        sh_mid_y = 0.55
        ear_mid_y = sh_mid_y - h
        left_sh = LandmarkPoint(x=0.35, y=sh_mid_y, z=0.0)
        right_sh = LandmarkPoint(x=0.65, y=sh_mid_y, z=0.0)
        left_ear = LandmarkPoint(x=0.45, y=ear_mid_y, z=0.0)
        right_ear = LandmarkPoint(x=0.55, y=ear_mid_y, z=0.0)
        nose = LandmarkPoint(x=0.50, y=ear_mid_y, z=0.0)
        left_eye = LandmarkPoint(x=0.465, y=ear_mid_y, z=0.0)
        right_eye = LandmarkPoint(x=0.535, y=ear_mid_y, z=0.0)
        return ExtractedPose(
            nose=nose,
            left_eye=left_eye,
            right_eye=right_eye,
            left_ear=left_ear,
            right_ear=right_ear,
            left_shoulder=left_sh,
            right_shoulder=right_sh,
        )

    # 1. Baseline good pose (height = 0.20 -> ratio = 1.0)
    res1 = engine.evaluate(make_pose_with_neck_height(0.20))
    assert res1.state == PostureState.GOOD

    # 2. Borderline posture (height = 0.17 -> ratio = 0.85). Since enter threshold is 0.82, stays GOOD.
    res2 = engine.evaluate(make_pose_with_neck_height(0.17))
    assert res2.state == PostureState.GOOD

    # 3. Severe slouch (height = 0.15 -> ratio = 0.75 < 0.82). Transitions to SLOUCHING!
    res3 = engine.evaluate(make_pose_with_neck_height(0.15))
    assert res3.state == PostureState.SLOUCHING

    # 4. Partial recovery (height = 0.17 -> ratio = 0.85). In hysteresis deadband!
    # Because exit threshold is 0.90, it MUST STAY in SLOUCHING to prevent flickering!
    res4 = engine.evaluate(make_pose_with_neck_height(0.17))
    assert res4.state == PostureState.SLOUCHING

    # 5. Full recovery (height = 0.19 -> ratio = 0.95 > 0.90). Re-enters GOOD!
    res5 = engine.evaluate(make_pose_with_neck_height(0.19))
    assert res5.state == PostureState.GOOD


def test_exponential_smoothing_stabilizes_jitter():
    """Verify that with smoothing_alpha=0.3, a single sudden frame spike does not immediately flip state."""
    calib = CalibrationProfile.default_ergonomic_baseline()
    engine = PostureEngine(calibration=calib, smoothing_alpha=0.30)
    gen = MockPostureGenerator()

    # Feed 10 good frames to establish baseline
    for _ in range(10):
        engine.evaluate(gen.generate_pose(PostureState.GOOD))

    # A single sudden glitch slouch frame arrives
    single_glitch = gen.generate_pose(PostureState.SLOUCHING)
    res = engine.evaluate(single_glitch)

    # Because EMA alpha is 0.30, the single frame does not instantly plunge below enter threshold
    assert res.state == PostureState.GOOD


def test_mock_generator_renders_frames():
    """Verify MockPostureGenerator renders valid 640x480 BGR frames."""
    gen = MockPostureGenerator()
    frame = gen.generate_frame(PostureState.GOOD)
    assert frame.shape == (480, 640, 3)
    assert frame.dtype == np.uint8

    slouch_frame = gen.generate_frame(PostureState.SLOUCHING)
    assert slouch_frame.shape == (480, 640, 3)
