"""
Power-Aware Monitoring Module for Smart Posture and Ergonomics Tracker.

Adapts monitoring behavior based on Windows power state (AC vs Battery):
- Section 5A: Battery Mode -> Periodic 5-minute sampling with 3-5s burst capture,
  immediate hardware camera release, and ultra-low background CPU/GPU footprint.
- Section 5B: Plugged-in Mode -> Continuous 2-5 FPS monitoring with live feed.
- Section 5C: Power-state transitions -> Safe live switching without duplicate
  camera handles, resource leaks, or UI freezes.
"""
from dataclasses import dataclass
from enum import Enum
import logging
import sys
from typing import Callable, Optional
import psutil

logger = logging.getLogger(__name__)


class PowerMode(str, Enum):
    """Operational power modes."""
    AC_CONTINUOUS = "AC_CONTINUOUS"   # Laptop plugged in or desktop PC: continuous 2-5 FPS
    BATTERY_SAVER = "BATTERY_SAVER"   # Laptop running on battery: 5-min periodic burst check


@dataclass
class PowerState:
    """Current snapshot of system power and battery metrics."""
    mode: PowerMode
    is_plugged: bool
    percent: int
    seconds_left: Optional[int]
    description: str

    @property
    def is_battery(self) -> bool:
        return self.mode == PowerMode.BATTERY_SAVER


class PowerManager:
    """
    Monitors system power status and detects AC/Battery transitions.
    Provides battery percentage, charger presence, and recommended sampling schedules.
    """

    def __init__(
        self,
        on_transition: Optional[Callable[[PowerState], None]] = None,
        battery_check_interval_sec: float = 300.0,  # 5 minutes
        burst_duration_sec: float = 3.0,            # 3-5 seconds
    ) -> None:
        self.on_transition = on_transition
        self.battery_check_interval_sec = battery_check_interval_sec
        self.burst_duration_sec = burst_duration_sec

        self._last_state: Optional[PowerState] = None
        # Initialize baseline state
        self._last_state = self.get_current_power_state()

    def get_current_power_state(self) -> PowerState:
        """Query system battery status via psutil with safe desktop fallbacks."""
        try:
            battery = psutil.sensors_battery()
        except Exception as e:
            logger.warning("Could not query battery sensors (%s); assuming AC wall power.", e)
            battery = None

        if battery is None:
            # Desktop PC or hardware without battery sensors
            return PowerState(
                mode=PowerMode.AC_CONTINUOUS,
                is_plugged=True,
                percent=100,
                seconds_left=None,
                description="AC Wall Power (Continuous 3-5 FPS)",
            )

        is_plugged = bool(battery.power_plugged) if battery.power_plugged is not None else True
        percent = int(battery.percent) if battery.percent is not None else 100
        seconds_left = int(battery.secsleft) if (battery.secsleft is not None and battery.secsleft > 0) else None

        if is_plugged:
            mode = PowerMode.AC_CONTINUOUS
            desc = f"AC Wall Power ({percent}% - Continuous 3-5 FPS)"
        else:
            mode = PowerMode.BATTERY_SAVER
            desc = f"Battery Saver ({percent}% - Periodic 5-Min Burst Mode)"

        return PowerState(
            mode=mode,
            is_plugged=is_plugged,
            percent=percent,
            seconds_left=seconds_left,
            description=desc,
        )

    def poll(self) -> Optional[PowerState]:
        """
        Check for power transitions.
        If a transition between AC and Battery occurs, triggers callback and returns new state.
        """
        current = self.get_current_power_state()
        if self._last_state is None or self._last_state.mode != current.mode or self._last_state.is_plugged != current.is_plugged:
            logger.info(
                "Power state transition detected: %s -> %s (Battery: %d%%, Plugged: %s)",
                self._last_state.mode if self._last_state else "None",
                current.mode,
                current.percent,
                current.is_plugged,
            )
            self._last_state = current
            if self.on_transition:
                try:
                    self.on_transition(current)
                except Exception as e:
                    logger.error("Error executing power transition callback: %s", e)
            return current
        return None

    @property
    def current_state(self) -> PowerState:
        """Return the most recently recorded or freshly polled power state."""
        if self._last_state is None:
            self._last_state = self.get_current_power_state()
        return self._last_state

    @property
    def is_battery(self) -> bool:
        """True if currently operating on battery power."""
        return self.current_state.mode == PowerMode.BATTERY_SAVER
