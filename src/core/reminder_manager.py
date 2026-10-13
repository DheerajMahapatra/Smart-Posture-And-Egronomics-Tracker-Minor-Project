"""
Reminder & Escalation Engine for Smart Posture and Ergonomics Tracker.

Implements a deterministic state machine managing posture notifications:
- Tracks continuous duration of poor posture against a persistence threshold.
- Rate-limits reminders to at most once per configured interval (default 5 minutes).
- Escalates through a 3-warning progression before triggering the full-screen overlay.
- Requires sustained posture correction for a confirmation period before resetting.
- Completely decoupled from camera capture and UI threads.
"""
from dataclasses import dataclass
from enum import Enum
import logging
import time
from typing import Callable, List, Optional

from src.core.posture_engine import PostureAssessment, PostureState

logger = logging.getLogger(__name__)


class ReminderState(str, Enum):
    """Formal states of the reminder escalation state machine."""
    MONITORING = "MONITORING"
    POOR_POSTURE_DETECTED = "POOR_POSTURE_DETECTED"
    WARNING_1 = "WARNING_1"
    WARNING_2 = "WARNING_2"
    WARNING_3 = "WARNING_3"
    OVERLAY_ACTIVE = "OVERLAY_ACTIVE"
    CORRECTION_CONFIRMED = "CORRECTION_CONFIRMED"
    COOLDOWN_RESET = "COOLDOWN_RESET"


@dataclass
class ReminderEvent:
    """Event payload emitted on posture reminder evaluations."""
    state: ReminderState
    warning_level: int                  # 0 = none, 1-3 = warnings, 4 = overlay
    message: str                        # Human-readable reminder text
    issue_description: str              # e.g., "Slouching / hunched posture"
    should_alert: bool                  # True if a new audible/visual alert should fire
    should_show_overlay: bool           # True when full-screen overlay must open
    should_dismiss_overlay: bool        # True when overlay should close
    sustained_poor_duration: float      # Seconds poor posture has persisted
    correction_hold_duration: float     # Seconds good posture has been held


class ReminderManager:
    """
    Manages posture alert thresholds, rate-limiting, and 3-stage warning escalation.
    """

    def __init__(
        self,
        persistence_threshold_sec: float = 20.0,
        reminder_interval_sec: float = 300.0,  # 5 minutes
        max_warnings: int = 3,
        correction_confirmation_sec: float = 3.0,
        on_event: Optional[Callable[[ReminderEvent], None]] = None,
    ) -> None:
        self.persistence_threshold_sec = persistence_threshold_sec
        self.reminder_interval_sec = reminder_interval_sec
        self.max_warnings = max_warnings
        self.correction_confirmation_sec = correction_confirmation_sec
        self.on_event = on_event

        # State machine variables
        self.state = ReminderState.MONITORING
        self.warning_count = 0

        # Timing trackers
        self.poor_posture_start_time: Optional[float] = None
        self.last_warning_time: Optional[float] = None
        self.good_posture_start_time: Optional[float] = None
        self.last_issue = ""

    def reset(self) -> None:
        """Reset state machine and all timer counters to idle monitoring."""
        self.state = ReminderState.MONITORING
        self.warning_count = 0
        self.poor_posture_start_time = None
        self.last_warning_time = None
        self.good_posture_start_time = None
        self.last_issue = ""
        logger.debug("ReminderManager reset to initial MONITORING state.")

    def emergency_dismiss(self) -> ReminderEvent:
        """Emergency dismissal of active overlay."""
        was_overlay = (self.state == ReminderState.OVERLAY_ACTIVE)
        self.reset()
        event = ReminderEvent(
            state=ReminderState.MONITORING,
            warning_level=0,
            message="Overlay dismissed by user.",
            issue_description="",
            should_alert=False,
            should_show_overlay=False,
            should_dismiss_overlay=was_overlay,
            sustained_poor_duration=0.0,
            correction_hold_duration=0.0,
        )
        if self.on_event:
            self.on_event(event)
        return event

    def process_assessment(
        self,
        assessment: PostureAssessment,
        now: Optional[float] = None,
    ) -> ReminderEvent:
        """
        Evaluate an incoming posture assessment through the state machine.
        Optional `now` timestamp allows precise deterministic unit testing.
        """
        current_time = time.time() if now is None else now

        # Case 1: No person detected in camera frame
        if assessment.state == PostureState.NO_PERSON:
            # Pause timers without escalating
            self.good_posture_start_time = None
            return ReminderEvent(
                state=self.state,
                warning_level=self.warning_count,
                message="User away / no face detected.",
                issue_description="No user detected",
                should_alert=False,
                should_show_overlay=False,
                should_dismiss_overlay=False,
                sustained_poor_duration=0.0,
                correction_hold_duration=0.0,
            )

        # Update last known issue description
        if not assessment.is_good:
            self.last_issue = assessment.primary_issue

        # --------------------------------------------------------------------
        # State: MONITORING (Optimal posture, no active warnings)
        # --------------------------------------------------------------------
        if self.state == ReminderState.MONITORING:
            if assessment.is_good:
                self.poor_posture_start_time = None
                return self._make_event(
                    message="Optimal posture maintained.",
                    should_alert=False,
                )
            else:
                # Poor posture just started
                self.state = ReminderState.POOR_POSTURE_DETECTED
                self.poor_posture_start_time = current_time
                self.good_posture_start_time = None
                logger.debug("Poor posture detected: %s. Starting persistence timer.", self.last_issue)
                return self._make_event(
                    message=f"Poor posture detected: {self.last_issue}.",
                    should_alert=False,
                )

        # --------------------------------------------------------------------
        # State: POOR_POSTURE_DETECTED (Monitoring persistence threshold)
        # --------------------------------------------------------------------
        if self.state == ReminderState.POOR_POSTURE_DETECTED:
            if assessment.is_good:
                # User self-corrected quickly before persistence threshold
                self.state = ReminderState.MONITORING
                self.poor_posture_start_time = None
                logger.debug("Brief posture deviation cleared before persistence threshold.")
                return self._make_event(
                    message="Transient deviation cleared.",
                    should_alert=False,
                )
            else:
                start = self.poor_posture_start_time if self.poor_posture_start_time is not None else current_time
                elapsed = current_time - start
                if elapsed >= self.persistence_threshold_sec:
                    # Sustained poor posture threshold reached -> Warning 1!
                    self.warning_count = 1
                    self.state = ReminderState.WARNING_1
                    self.last_warning_time = current_time
                    logger.info("Warning 1 triggered: Sustained poor posture (%.1fs)", elapsed)
                    return self._make_event(
                        message=f"Posture Reminder (1/{self.max_warnings}): Please sit upright.",
                        should_alert=True,
                        sustained_poor=elapsed,
                    )
                else:
                    return self._make_event(
                        message=f"Monitoring poor posture duration ({elapsed:.0f}s/{self.persistence_threshold_sec:.0f}s)",
                        should_alert=False,
                        sustained_poor=elapsed,
                    )

        # --------------------------------------------------------------------
        # States: WARNING_1, WARNING_2, WARNING_3
        # --------------------------------------------------------------------
        if self.state in (ReminderState.WARNING_1, ReminderState.WARNING_2, ReminderState.WARNING_3):
            if assessment.is_good:
                # User is attempting to correct posture
                if self.good_posture_start_time is None:
                    self.good_posture_start_time = current_time

                hold_time = current_time - self.good_posture_start_time
                if hold_time >= self.correction_confirmation_sec:
                    # Sustained correction confirmed!
                    self.state = ReminderState.CORRECTION_CONFIRMED
                    self.warning_count = 0
                    self.poor_posture_start_time = None
                    self.last_warning_time = None
                    self.good_posture_start_time = None
                    logger.info("Posture correction confirmed (held %.1fs). Warnings reset.", hold_time)
                    event = self._make_event(
                        message="Posture corrected! Great job.",
                        should_alert=False,
                        correction_hold=hold_time,
                    )
                    # Return to clean monitoring
                    self.state = ReminderState.MONITORING
                    return event
                else:
                    return self._make_event(
                        message=f"Holding correct posture ({hold_time:.1f}s/{self.correction_confirmation_sec:.1f}s)...",
                        should_alert=False,
                        correction_hold=hold_time,
                    )
            else:
                # Poor posture persists
                self.good_posture_start_time = None
                lw = self.last_warning_time if self.last_warning_time is not None else current_time
                time_since_last_warn = current_time - lw
                ps = self.poor_posture_start_time if self.poor_posture_start_time is not None else current_time
                total_poor = current_time - ps

                if time_since_last_warn >= self.reminder_interval_sec:
                    # Interval expired while poor posture continues -> Escalate!
                    if self.warning_count < self.max_warnings:
                        self.warning_count += 1
                        self.state = ReminderState(f"WARNING_{self.warning_count}")
                        self.last_warning_time = current_time
                        logger.info("Warning %d triggered after %.1fs interval", self.warning_count, time_since_last_warn)
                        return self._make_event(
                            message=f"Posture Warning ({self.warning_count}/{self.max_warnings}): Please correct your posture.",
                            should_alert=True,
                            sustained_poor=total_poor,
                        )
                    else:
                        # Escalation limit reached -> Trigger full-screen overlay!
                        self.state = ReminderState.OVERLAY_ACTIVE
                        logger.warning("Max warnings exceeded! Escalating to Full-Screen Overlay.")
                        return self._make_event(
                            message="Maximum warnings reached. Full-screen overlay triggered.",
                            should_alert=True,
                            should_show_overlay=True,
                            sustained_poor=total_poor,
                        )
                else:
                    return self._make_event(
                        message=f"Warning active. Next reminder in {int(self.reminder_interval_sec - time_since_last_warn)}s if posture is not corrected.",
                        should_alert=False,
                        sustained_poor=total_poor,
                    )

        # --------------------------------------------------------------------
        # State: OVERLAY_ACTIVE (Full-screen overlay is showing)
        # --------------------------------------------------------------------
        if self.state == ReminderState.OVERLAY_ACTIVE:
            if assessment.is_good:
                # User is correcting behind the overlay!
                if self.good_posture_start_time is None:
                    self.good_posture_start_time = current_time

                hold_time = current_time - self.good_posture_start_time
                if hold_time >= self.correction_confirmation_sec:
                    # Sustained correction confirmed! Dismiss overlay and reset
                    self.state = ReminderState.CORRECTION_CONFIRMED
                    self.warning_count = 0
                    self.poor_posture_start_time = None
                    self.last_warning_time = None
                    self.good_posture_start_time = None
                    logger.info("Overlay: Sustained correction confirmed! Dismissing overlay.")
                    event = self._make_event(
                        message="Posture corrected! Dismissing overlay.",
                        should_alert=False,
                        should_dismiss_overlay=True,
                        correction_hold=hold_time,
                    )
                    self.state = ReminderState.MONITORING
                    return event
                else:
                    return self._make_event(
                        message=f"Posture corrected! Keep holding... ({hold_time:.1f}s/{self.correction_confirmation_sec:.1f}s)",
                        should_alert=False,
                        correction_hold=hold_time,
                    )
            else:
                self.good_posture_start_time = None
                ps = self.poor_posture_start_time if self.poor_posture_start_time is not None else current_time
                total_poor = current_time - ps
                return self._make_event(
                    message="Fix your posture to dismiss the overlay.",
                    should_alert=False,
                    sustained_poor=total_poor,
                )

        return self._make_event(message="Monitoring posture.", should_alert=False)

    def _make_event(
        self,
        message: str,
        should_alert: bool,
        should_show_overlay: bool = False,
        should_dismiss_overlay: bool = False,
        sustained_poor: float = 0.0,
        correction_hold: float = 0.0,
    ) -> ReminderEvent:
        event = ReminderEvent(
            state=self.state,
            warning_level=self.warning_count,
            message=message,
            issue_description=self.last_issue,
            should_alert=should_alert,
            should_show_overlay=should_show_overlay,
            should_dismiss_overlay=should_dismiss_overlay,
            sustained_poor_duration=sustained_poor,
            correction_hold_duration=correction_hold,
        )
        if self.on_event and should_alert:
            try:
                self.on_event(event)
            except Exception as e:
                logger.error("Error executing reminder on_event callback: %s", e)
        return event
