"""
Core posture evaluation, calibration, and reminder modules.
"""
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
from src.core.reminder_manager import (
    ReminderManager,
    ReminderState,
    ReminderEvent,
)

__all__ = [
    "CalibrationProfile",
    "CalibrationAccumulator",
    "MEDICAL_DISCLAIMER",
    "PostureEngine",
    "PostureState",
    "PostureAssessment",
    "MockPostureGenerator",
    "ReminderManager",
    "ReminderState",
    "ReminderEvent",
]
