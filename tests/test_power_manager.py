"""
Unit and integration tests for PowerManager and Adaptive Power Scheduling (Phase 7).

Tests cover:
- Detection of AC wall power vs Battery mode via psutil
- Safe fallback when battery sensor is unavailable (desktop PCs)
- Transition detection between plugged in and unplugged states
- Transition callback execution
- Worker integration and power-state signaling
"""
from unittest.mock import MagicMock, patch
import pytest

from src.system.power_manager import PowerManager, PowerMode, PowerState
from src.ui.settings import AppSettings
from src.ui.worker import PostureWorker


class TestPowerManager:
    """Test suite for system power monitoring and state transitions."""

    def test_power_state_when_plugged_in_ac(self) -> None:
        mock_battery = MagicMock()
        mock_battery.power_plugged = True
        mock_battery.percent = 100
        mock_battery.secsleft = -1

        with patch("psutil.sensors_battery", return_value=mock_battery):
            mgr = PowerManager()
            state = mgr.get_current_power_state()
            assert state.mode == PowerMode.AC_CONTINUOUS
            assert state.is_plugged is True
            assert state.is_battery is False
            assert state.percent == 100

    def test_power_state_when_running_on_battery(self) -> None:
        mock_battery = MagicMock()
        mock_battery.power_plugged = False
        mock_battery.percent = 68
        mock_battery.secsleft = 7200

        with patch("psutil.sensors_battery", return_value=mock_battery):
            mgr = PowerManager()
            state = mgr.get_current_power_state()
            assert state.mode == PowerMode.BATTERY_SAVER
            assert state.is_plugged is False
            assert state.is_battery is True
            assert state.percent == 68
            assert state.seconds_left == 7200

    def test_desktop_pc_fallback_when_sensors_absent(self) -> None:
        with patch("psutil.sensors_battery", return_value=None):
            mgr = PowerManager()
            state = mgr.get_current_power_state()
            assert state.mode == PowerMode.AC_CONTINUOUS
            assert state.is_plugged is True
            assert state.is_battery is False
            assert state.percent == 100

    def test_power_state_sensor_exception_safety(self) -> None:
        with patch("psutil.sensors_battery", side_effect=RuntimeError("Sensor error")):
            mgr = PowerManager()
            state = mgr.get_current_power_state()
            assert state.mode == PowerMode.AC_CONTINUOUS
            assert state.is_plugged is True

    def test_transition_detection_and_callbacks(self) -> None:
        transitions_log = []

        def callback(state: PowerState) -> None:
            transitions_log.append(state)

        # 1. Initialize on AC
        mock_ac = MagicMock(power_plugged=True, percent=95, secsleft=-1)
        mock_batt = MagicMock(power_plugged=False, percent=94, secsleft=6000)

        with patch("psutil.sensors_battery", return_value=mock_ac):
            mgr = PowerManager(on_transition=callback)
            assert mgr.current_state.mode == PowerMode.AC_CONTINUOUS
            # Same state -> poll returns None
            assert mgr.poll() is None

        # 2. Unplug charger -> Switch to Battery
        with patch("psutil.sensors_battery", return_value=mock_batt):
            t1 = mgr.poll()
            assert t1 is not None
            assert t1.mode == PowerMode.BATTERY_SAVER
            assert t1.is_plugged is False
            assert len(transitions_log) == 1
            assert transitions_log[0].mode == PowerMode.BATTERY_SAVER

            # Repetitive poll on battery without change -> no new event
            assert mgr.poll() is None
            assert len(transitions_log) == 1

        # 3. Plug charger back in -> Switch to AC
        with patch("psutil.sensors_battery", return_value=mock_ac):
            t2 = mgr.poll()
            assert t2 is not None
            assert t2.mode == PowerMode.AC_CONTINUOUS
            assert t2.is_plugged is True
            assert len(transitions_log) == 2
            assert transitions_log[1].mode == PowerMode.AC_CONTINUOUS

    def test_worker_initializes_power_manager(self) -> None:
        settings = AppSettings(
            demo_mode=True,
            battery_check_interval_sec=300,
            burst_duration_sec=3.0,
        )
        worker = PostureWorker(settings)
        assert worker.power_mgr is not None
        assert worker.power_mgr.battery_check_interval_sec == 300.0
        assert worker.power_mgr.burst_duration_sec == 3.0
