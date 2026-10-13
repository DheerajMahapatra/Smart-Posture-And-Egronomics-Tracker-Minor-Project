"""
Unit and regression tests for ReminderManager state machine.

Tests cover:
- Initial idle monitoring state
- Transient poor posture clearance before persistence threshold
- Sustained poor posture progression to Warning 1
- Rate-limiting intervals between sequential warnings (300s / 5min)
- Escalation progression: WARNING_1 -> WARNING_2 -> WARNING_3 -> OVERLAY_ACTIVE
- Sustained posture correction hold (3s) resetting escalation sequence
- Overlay dismissal upon confirmed upright posture
- Emergency manual dismissal override
- Handling of user absence (NO_PERSON)
- Dynamic reconfiguration of thresholds
- Event callback firing on alerts
"""
import pytest
from src.core.posture_engine import PostureAssessment, PostureState
from src.core.reminder_manager import ReminderEvent, ReminderManager, ReminderState


def _make_assessment(state: PostureState, issue: str = "") -> PostureAssessment:
    """Helper to generate mock PostureAssessment objects."""
    is_good = (state == PostureState.GOOD)
    return PostureAssessment(
        state=state,
        is_good=is_good,
        primary_issue=issue if issue else ("Optimal posture" if is_good else state.value),
        confidence=0.95,
        slouch_compression_ratio=1.0 if is_good else 0.70,
        shoulder_tilt_deg=0.0,
        forward_head_delta=0.0 if is_good else 0.12,
        eye_proximity_ratio=1.0 if is_good else 1.35,
        lateral_lean_deg=0.0,
    )


class TestReminderManager:
    """Comprehensive test suite for the posture reminder state machine."""

    def test_initial_state_monitoring(self) -> None:
        mgr = ReminderManager()
        assert mgr.state == ReminderState.MONITORING
        assert mgr.warning_count == 0

        # Good assessments maintain MONITORING with no alerts
        good = _make_assessment(PostureState.GOOD)
        event = mgr.process_assessment(good, now=100.0)
        assert event.state == ReminderState.MONITORING
        assert event.warning_level == 0
        assert event.should_alert is False
        assert event.should_show_overlay is False

    def test_transient_slouch_does_not_trigger_warning(self) -> None:
        mgr = ReminderManager(persistence_threshold_sec=20.0)
        slouch = _make_assessment(PostureState.SLOUCHING, "Hunched neck")
        good = _make_assessment(PostureState.GOOD)

        # Poor posture starts at t=100
        e1 = mgr.process_assessment(slouch, now=100.0)
        assert e1.state == ReminderState.POOR_POSTURE_DETECTED
        assert e1.should_alert is False

        # Still poor at t=110 (10s < 20s threshold)
        e2 = mgr.process_assessment(slouch, now=110.0)
        assert e2.state == ReminderState.POOR_POSTURE_DETECTED
        assert e2.should_alert is False

        # User sits upright at t=115 -> self-correction before threshold!
        e3 = mgr.process_assessment(good, now=115.0)
        assert e3.state == ReminderState.MONITORING
        assert e3.warning_level == 0
        assert e3.should_alert is False
        assert mgr.state == ReminderState.MONITORING

    def test_sustained_slouch_triggers_warning_1(self) -> None:
        mgr = ReminderManager(persistence_threshold_sec=20.0)
        slouch = _make_assessment(PostureState.SLOUCHING, "Hunched neck")

        mgr.process_assessment(slouch, now=100.0)
        # At t=119.9: just before threshold
        e_pre = mgr.process_assessment(slouch, now=119.9)
        assert e_pre.state == ReminderState.POOR_POSTURE_DETECTED
        assert e_pre.should_alert is False

        # At t=120.0: exactly 20.0s elapsed -> WARNING_1 fires!
        e_warn = mgr.process_assessment(slouch, now=120.0)
        assert e_warn.state == ReminderState.WARNING_1
        assert e_warn.warning_level == 1
        assert e_warn.should_alert is True
        assert e_warn.should_show_overlay is False
        assert "1/3" in e_warn.message

    def test_rate_limiting_between_sequential_warnings(self) -> None:
        mgr = ReminderManager(
            persistence_threshold_sec=20.0,
            reminder_interval_sec=300.0,  # 5 minutes
            max_warnings=3,
        )
        slouch = _make_assessment(PostureState.SLOUCHING, "Slouching")

        # Trigger Warning 1 at t=120
        mgr.process_assessment(slouch, now=100.0)
        mgr.process_assessment(slouch, now=120.0)
        assert mgr.state == ReminderState.WARNING_1

        # Intermediate frames at t=125, t=200, t=419 (all < 300s interval)
        for t in [125.0, 200.0, 350.0, 419.0]:
            event = mgr.process_assessment(slouch, now=t)
            assert event.state == ReminderState.WARNING_1
            assert event.should_alert is False, f"Alert prematurely fired at t={t}"
            assert event.warning_level == 1

        # At t=420 (exactly 300s after Warning 1) -> Warning 2 fires!
        e_w2 = mgr.process_assessment(slouch, now=420.0)
        assert e_w2.state == ReminderState.WARNING_2
        assert e_w2.warning_level == 2
        assert e_w2.should_alert is True
        assert "2/3" in e_w2.message

    def test_full_escalation_progression_to_overlay(self) -> None:
        mgr = ReminderManager(
            persistence_threshold_sec=20.0,
            reminder_interval_sec=300.0,
            max_warnings=3,
        )
        forward_head = _make_assessment(PostureState.FORWARD_HEAD, "Tech neck")

        # t=100: start poor posture
        mgr.process_assessment(forward_head, now=100.0)

        # t=120: Warning 1
        w1 = mgr.process_assessment(forward_head, now=120.0)
        assert w1.state == ReminderState.WARNING_1
        assert w1.should_alert is True

        # t=420: Warning 2
        w2 = mgr.process_assessment(forward_head, now=420.0)
        assert w2.state == ReminderState.WARNING_2
        assert w2.should_alert is True

        # t=720: Warning 3 (Max warning)
        w3 = mgr.process_assessment(forward_head, now=720.0)
        assert w3.state == ReminderState.WARNING_3
        assert w3.should_alert is True
        assert "3/3" in w3.message

        # Intermediate checks before interval expires -> no overlay yet
        mid = mgr.process_assessment(forward_head, now=850.0)
        assert mid.state == ReminderState.WARNING_3
        assert mid.should_show_overlay is False

        # t=1020: 300s after Warning 3 -> ESCALATES TO OVERLAY!
        overlay_event = mgr.process_assessment(forward_head, now=1020.0)
        assert overlay_event.state == ReminderState.OVERLAY_ACTIVE
        assert overlay_event.should_alert is True
        assert overlay_event.should_show_overlay is True
        assert mgr.state == ReminderState.OVERLAY_ACTIVE

        # Subsequent poor frames keep overlay active without re-triggering show_overlay
        next_event = mgr.process_assessment(forward_head, now=1030.0)
        assert next_event.state == ReminderState.OVERLAY_ACTIVE
        assert next_event.should_show_overlay is False

    def test_sustained_correction_hold_resets_warnings(self) -> None:
        mgr = ReminderManager(
            persistence_threshold_sec=20.0,
            reminder_interval_sec=300.0,
            correction_confirmation_sec=3.0,
        )
        slouch = _make_assessment(PostureState.SLOUCHING)
        good = _make_assessment(PostureState.GOOD)

        # Reach Warning 2
        mgr.process_assessment(slouch, now=100.0)
        mgr.process_assessment(slouch, now=120.0)  # W1
        mgr.process_assessment(slouch, now=420.0)  # W2
        assert mgr.state == ReminderState.WARNING_2

        # User starts sitting upright at t=450
        h1 = mgr.process_assessment(good, now=450.0)
        assert h1.correction_hold_duration == 0.0

        # Holding upright for 1.5s (< 3.0s confirmation)
        h2 = mgr.process_assessment(good, now=451.5)
        assert h2.correction_hold_duration == 1.5
        assert mgr.state == ReminderState.WARNING_2  # Not reset yet!

        # Premature slouch at t=452 breaks hold
        mgr.process_assessment(slouch, now=452.0)
        assert mgr.good_posture_start_time is None

        # User sits upright again at t=460 and holds for full 3.0s (t=463)
        mgr.process_assessment(good, now=460.0)
        mgr.process_assessment(good, now=461.5)
        confirmed = mgr.process_assessment(good, now=463.0)

        assert confirmed.state == ReminderState.CORRECTION_CONFIRMED
        assert confirmed.warning_level == 0
        assert mgr.warning_count == 0
        assert mgr.state == ReminderState.MONITORING

    def test_overlay_dismissal_on_sustained_correction(self) -> None:
        mgr = ReminderManager(
            persistence_threshold_sec=20.0,
            reminder_interval_sec=300.0,
            correction_confirmation_sec=3.0,
        )
        slouch = _make_assessment(PostureState.SLOUCHING)
        good = _make_assessment(PostureState.GOOD)

        # Escalate directly to OVERLAY_ACTIVE
        mgr.process_assessment(slouch, now=100.0)
        mgr.process_assessment(slouch, now=120.0)   # W1
        mgr.process_assessment(slouch, now=420.0)   # W2
        mgr.process_assessment(slouch, now=720.0)   # W3
        mgr.process_assessment(slouch, now=1020.0)  # Overlay
        assert mgr.state == ReminderState.OVERLAY_ACTIVE

        # User sits upright behind overlay at t=1030
        mgr.process_assessment(good, now=1030.0)
        mgr.process_assessment(good, now=1031.5)  # 1.5s hold

        # Held upright for 3.0s (t=1033.0) -> DISMISS OVERLAY!
        dismiss_ev = mgr.process_assessment(good, now=1033.0)
        assert dismiss_ev.should_dismiss_overlay is True
        assert dismiss_ev.state == ReminderState.CORRECTION_CONFIRMED
        assert mgr.warning_count == 0
        assert mgr.state == ReminderState.MONITORING

    def test_emergency_dismiss(self) -> None:
        mgr = ReminderManager()
        mgr.state = ReminderState.OVERLAY_ACTIVE
        mgr.warning_count = 3

        event = mgr.emergency_dismiss()
        assert event.should_dismiss_overlay is True
        assert event.state == ReminderState.MONITORING
        assert mgr.state == ReminderState.MONITORING
        assert mgr.warning_count == 0

    def test_no_person_pauses_without_escalating(self) -> None:
        mgr = ReminderManager(persistence_threshold_sec=20.0)
        slouch = _make_assessment(PostureState.SLOUCHING)
        no_person = _make_assessment(PostureState.NO_PERSON)

        # Slouch for 10s
        mgr.process_assessment(slouch, now=100.0)
        mgr.process_assessment(slouch, now=110.0)
        assert mgr.state == ReminderState.POOR_POSTURE_DETECTED

        # User steps away for 10 minutes
        for t in [115.0, 300.0, 700.0]:
            ev = mgr.process_assessment(no_person, now=t)
            assert ev.should_alert is False
            assert ev.should_show_overlay is False

        # State remained paused in POOR_POSTURE_DETECTED without jumping to Warning
        assert mgr.state == ReminderState.POOR_POSTURE_DETECTED
        assert mgr.warning_count == 0

    def test_on_event_callback_fires_on_alerts(self) -> None:
        alert_log = []

        def callback(event: ReminderEvent) -> None:
            alert_log.append(event)

        mgr = ReminderManager(
            persistence_threshold_sec=5.0,
            reminder_interval_sec=10.0,
            max_warnings=2,
            on_event=callback,
        )
        slouch = _make_assessment(PostureState.SLOUCHING)

        mgr.process_assessment(slouch, now=0.0)
        mgr.process_assessment(slouch, now=5.0)   # Triggers Warning 1
        mgr.process_assessment(slouch, now=15.0)  # Triggers Warning 2
        mgr.process_assessment(slouch, now=25.0)  # Triggers Overlay

        assert len(alert_log) == 3
        assert alert_log[0].state == ReminderState.WARNING_1
        assert alert_log[1].state == ReminderState.WARNING_2
        assert alert_log[2].state == ReminderState.OVERLAY_ACTIVE

    def test_custom_configurable_thresholds(self) -> None:
        mgr = ReminderManager(
            persistence_threshold_sec=8.0,
            reminder_interval_sec=60.0,
            max_warnings=2,
            correction_confirmation_sec=1.5,
        )
        assert mgr.persistence_threshold_sec == 8.0
        assert mgr.reminder_interval_sec == 60.0
        assert mgr.max_warnings == 2
        assert mgr.correction_confirmation_sec == 1.5
