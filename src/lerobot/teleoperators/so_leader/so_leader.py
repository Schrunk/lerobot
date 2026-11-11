# !/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import time
from queue import Empty, Queue
from typing import TYPE_CHECKING, Any

from lerobot.motors import Motor, MotorNormMode, SerialMotorsBus
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected
from lerobot.utils.import_utils import _pynput_available
from lerobot.utils.keyboard_input import pynput_can_capture

from ..teleoperator import Teleoperator
from ..utils import TeleopEvents
from .config_so_leader import SOLeaderTeleopConfig

PYNPUT_AVAILABLE = _pynput_available
if TYPE_CHECKING:
    from pynput import keyboard
elif PYNPUT_AVAILABLE:
    try:
        from pynput import keyboard
    except Exception as e:  # installed but not importable here, e.g. no reachable X display
        PYNPUT_AVAILABLE = False
        logging.info("Could not import pynput keyboard backend: %s", e)

logger = logging.getLogger(__name__)

# Episode control keys, read by `get_teleop_events()` (e.g. in `lerobot.rl.gym_manipulator`).
_EPISODE_KEYS_HELP = """SO leader episode controls:
  s   : end episode with SUCCESS
  q   : end episode with FAILURE
  r   : discard and re-record the episode
  Esc : end the episode and stop recording"""

_HOMING_POSITION_DIAGRAM = r"""
       ╭─────┬────────────────────┬──────╮ ◉╲═════╗   ← moveable claw
       │     │      forearm       │  ▤▤  │╤══╲════╝
       ╰┬───┬┴────────────────────┴──────┴┴═══════╝
        │   │                                  ███
        │   │                                  ███  ← handle
        │   │                                  ███
        │   │   upper arm
        │   │
        │   │
        │   │
   ╭────┴───┴────╮
   │   base ◉    │
   ╰──┬───────┬──╯
  ╭───┴───────┴───╮
  │▓▓▓ C-clamp ▓▓▓│
  ╰───────────────╯
 ═══════════════════════════════════════════════════
                table edge
"""


class SOLeader(Teleoperator):
    """Generic SO leader base for SO-100/101/10X teleoperators."""

    config_class = SOLeaderTeleopConfig
    name = "so_leader"

    def __init__(self, config: SOLeaderTeleopConfig):
        super().__init__(config)
        self.config = config
        norm_mode_body = MotorNormMode.DEGREES if config.use_degrees else MotorNormMode.RANGE_M100_100
        self.bus = SerialMotorsBus(
            port=self.config.port,
            motors={
                "shoulder_pan": Motor(1, "sts3215", norm_mode_body),
                "shoulder_lift": Motor(2, "sts3215", norm_mode_body),
                "elbow_flex": Motor(3, "sts3215", norm_mode_body),
                "wrist_flex": Motor(4, "sts3215", norm_mode_body),
                "wrist_roll": Motor(5, "sts3215", norm_mode_body),
                "gripper": Motor(6, "sts3215", MotorNormMode.RANGE_0_100),
            },
            calibration=self.calibration,
        )
        # Key presses from the pynput listener thread, consumed in `get_teleop_events()`.
        self._key_queue: Queue[str] = Queue()
        self._key_listener: Any = None
        # Set once Esc was pressed; lets the control loop stop after the current episode.
        self.stop_requested = False

    @property
    def action_features(self) -> dict[str, type]:
        return {f"{motor}.pos": float for motor in self.bus.motors}

    @property
    def feedback_features(self) -> dict[str, type]:
        return self.action_features

    @property
    def is_connected(self) -> bool:
        return self.bus.is_connected

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        self.bus.connect()
        if not self.is_calibrated and calibrate:
            logger.info(
                "Mismatch between calibration values in the motor and the calibration file or no calibration file found"
            )
            self.calibrate()

        self.configure()
        self._start_key_listener()
        logger.info(f"{self} connected.")

    def _start_key_listener(self) -> None:
        """Listen for the episode control keys (s, q, r, Esc) on the local keyboard."""
        self.stop_requested = False
        if not (PYNPUT_AVAILABLE and pynput_can_capture()):
            logger.warning(
                "Keyboard episode controls (s/q/r/Esc) are unavailable: pynput can only capture keys on "
                "an X11 session (Linux), a Windows desktop or macOS with Input Monitoring granted."
            )
            return
        self._key_listener = keyboard.Listener(on_press=self._on_key_press)
        self._key_listener.start()
        print(_EPISODE_KEYS_HELP)

    def _on_key_press(self, key: Any) -> None:
        if key == keyboard.Key.esc:
            self._key_queue.put("esc")
            return
        char = getattr(key, "char", None)
        if char in ("s", "q", "r"):
            self._key_queue.put(char)

    def get_teleop_events(self) -> dict[TeleopEvents, bool]:
        """Return the episode control events pressed on the keyboard since the last call.

        - ``s``: success (episode ends with reward 1)
        - ``q``: failure (episode ends without success)
        - ``r``: re-record (episode ends and is discarded)
        - ``Esc``: episode ends without success and ``stop_requested`` is set

        The leader arm itself never reports an intervention.
        """
        terminate_episode = False
        success = False
        rerecord_episode = False

        while True:
            try:
                key = self._key_queue.get_nowait()
            except Empty:
                break
            if key == "s":
                terminate_episode = True
                success = True
            elif key == "q":
                terminate_episode = True
            elif key == "r":
                terminate_episode = True
                rerecord_episode = True
            elif key == "esc":
                logger.info("Esc pressed, stopping after this episode.")
                terminate_episode = True
                self.stop_requested = True

        return {
            TeleopEvents.IS_INTERVENTION: False,
            TeleopEvents.TERMINATE_EPISODE: terminate_episode,
            TeleopEvents.SUCCESS: success,
            TeleopEvents.RERECORD_EPISODE: rerecord_episode,
        }

    @property
    def is_calibrated(self) -> bool:
        return self.bus.is_calibrated

    def calibrate(self) -> None:
        if self._keep_calibration_file(self.bus):
            return

        print(_HOMING_POSITION_DIAGRAM)
        print("Video walkthrough: https://huggingface.co/docs/lerobot/main/en/so101#calibration-video")
        self.calibration = self.bus.record_calibration("position", full_turn_motors=["wrist_roll"])
        self.bus.write_calibration(self.calibration)
        self._save_calibration()
        print("Calibration saved to", self.calibration_fpath)

    def configure(self) -> None:
        self.bus.disable_torque()
        self.bus.configure_motors()
        self.bus.set_operating_mode("position")

    def enable_torque(self) -> None:
        self.bus.enable_torque()

    def disable_torque(self) -> None:
        self.bus.disable_torque()

    def setup_motors(self) -> None:
        self.bus.setup_motors()

    @check_if_not_connected
    def get_action(self) -> dict[str, float]:
        start = time.perf_counter()
        action = self.bus.sync_read("Present_Position", num_retry=self.config.num_read_retries)
        action = {f"{motor}.pos": val for motor, val in action.items()}
        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read action: {dt_ms:.1f}ms")
        return action

    @check_if_not_connected
    def send_feedback(self, feedback: dict[str, float]) -> None:
        goals = {k.removesuffix(".pos"): v for k, v in feedback.items() if k.endswith(".pos")}
        if goals:
            self.bus.sync_write("Goal_Position", goals)

    @check_if_not_connected
    def disconnect(self) -> None:
        if self._key_listener is not None:
            self._key_listener.stop()
            self._key_listener = None
        self.bus.disconnect()
        logger.info(f"{self} disconnected.")


SO100Leader = SOLeader
SO101Leader = SOLeader
