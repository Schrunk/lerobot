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

"""Read the load, current and temperature of a robot's serial-bus motors for live display."""

import logging
import time

from lerobot.motors import SerialMotorsBus

logger = logging.getLogger(__name__)

# quantity -> (register, unit)
TELEMETRY_REGISTERS: dict[str, tuple[str, str]] = {
    "load": ("Present_Load", "%"),
    "current": ("Present_Current", "mA"),
    "temperature": ("Present_Temperature", "°C"),
}

# Raw register value * scale = value in the unit above. Models not listed are shown raw.
_SCALES: dict[str, dict[str, float]] = {
    # Present_Load is in 0.1 % of the torque limit, Present_Current in steps of 6.5 mA.
    "sts3215": {"load": 0.1, "current": 6.5},
}


def find_motor_buses(robot: object) -> dict[str, SerialMotorsBus]:
    """Return the robot's motor buses keyed by the prefix its observation keys use.

    A single arm (`robot.bus`) gets the prefix "", a bimanual robot (`robot.left_arm.bus`,
    `robot.right_arm.bus`) gets "left_" and "right_".
    """
    bus = getattr(robot, "bus", None)
    if isinstance(bus, SerialMotorsBus):
        return {"": bus}
    buses = {}
    for side in ("left", "right"):
        arm_bus = getattr(getattr(robot, f"{side}_arm", None), "bus", None)
        if isinstance(arm_bus, SerialMotorsBus):
            buses[f"{side}_"] = arm_bus
    return buses


class MotorTelemetry:
    """Reads load, current and temperature of every motor, at most `rate_hz` times per second.

    Between two reads, `read()` returns the last values, so it can be called every control step
    without slowing the loop down more than `rate_hz` allows.
    """

    def __init__(self, robot: object, rate_hz: float = 10.0):
        self.buses = find_motor_buses(robot)
        self.period = 1.0 / rate_hz if rate_hz > 0 else 0.0
        self._last_read = -float("inf")
        self._values: dict[str, dict[str, float]] = {}
        # Per bus, per quantity: the motors that have the register.
        self._motors: dict[str, dict[str, list[str]]] = {
            prefix: {
                q: [m for m in bus.motors if bus._has_register(m, reg)]
                for q, (reg, _) in TELEMETRY_REGISTERS.items()
            }
            for prefix, bus in self.buses.items()
        }
        if not self.buses:
            logger.warning("Motor telemetry: the robot has no serial motor bus, nothing to show.")

    @property
    def available(self) -> bool:
        return bool(self.buses)

    def read(self) -> dict[str, dict[str, float]]:
        """Return `{motor_key: {"load": %, "current": mA, "temperature": °C}}`.

        `motor_key` is the motor name with the robot's prefix, e.g. "shoulder_pan" or
        "left_shoulder_pan", matching the action keys without ".pos".
        """
        now = time.perf_counter()
        if now - self._last_read < self.period:
            return self._values
        self._last_read = now

        values: dict[str, dict[str, float]] = {}
        for prefix, bus in self.buses.items():
            for q, (reg, _) in TELEMETRY_REGISTERS.items():
                motors = self._motors[prefix][q]
                if not motors:
                    continue
                try:
                    raw = bus.sync_read(reg, motors, normalize=False)
                except (ConnectionError, RuntimeError) as e:
                    # A missed telemetry read must not stop the teleoperation.
                    logger.debug(f"Motor telemetry: failed to read {reg}: {e}")
                    continue
                for m, v in raw.items():
                    scale = _SCALES.get(bus.motors[m].model, {}).get(q, 1.0)
                    values.setdefault(f"{prefix}{m}", {})[q] = float(v) * scale
        self._values = values
        return values

    @staticmethod
    def as_observation(values: dict[str, dict[str, float]]) -> dict[str, float]:
        """Flatten to observation-style keys, e.g. {"shoulder_pan.load": 12.3}."""
        return {f"{motor}.{q}": v for motor, qs in values.items() for q, v in qs.items()}
