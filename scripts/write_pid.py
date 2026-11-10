#!/usr/bin/env python
"""Read and set the PID gains, startup force and torque limit of the STS3215 motors of an SO-101 arm.

Examples:
    # Show the current values of all motors
    uv run python scripts/write_pid.py --port /dev/ttyACM1

    # Set P/I/D of all motors
    uv run python scripts/write_pid.py --port /dev/ttyACM1 --p 16 --i 0 --d 32

    # Set only P of two motors
    uv run python scripts/write_pid.py --port /dev/ttyACM1 --motors shoulder_lift elbow_flex --p 24

    # Raise the startup force and the torque limit of all motors
    uv run python scripts/write_pid.py --port /dev/ttyACM1 --startup-force 30 --max-torque 1000

    # Different gains per motor (MOTOR=P,I,D; leave a field empty to keep its value)
    uv run python scripts/write_pid.py --port /dev/ttyACM1 --set shoulder_lift=24,0,32 --set gripper=12,,

Registers:
    P, I, D        0..255   Position loop gains (default in LeRobot: 16, 0, 32).
    startup-force  0..1000  Minimum_Startup_Force: minimum output to overcome static friction.
    max-torque     0..1000  Max_Torque_Limit in 0.1 % of the stall torque (1000 = 100 %).
                            The live limit (Torque_Limit) is written too, so it applies at once.

Note: `lerobot-record`, `lerobot-teleoperate` etc. call `robot.configure()` on connect, which
overwrites P/I/D with the config values and sets the gripper's Max_Torque_Limit to 500. To keep
your gains there, also pass `--robot.position_p_coefficient=... --robot.position_i_coefficient=...
--robot.position_d_coefficient=...`. Minimum_Startup_Force and the torque limit of the other
motors are not touched by LeRobot.
"""

import argparse

from lerobot.motors import Motor, MotorNormMode, SerialMotorsBus

MOTORS = {
    "shoulder_pan": 1,
    "shoulder_lift": 2,
    "elbow_flex": 3,
    "wrist_flex": 4,
    "wrist_roll": 5,
    "gripper": 6,
}
# key -> (register, max value, column header)
REGISTERS = {
    "p": ("P_Coefficient", 255, "P"),
    "i": ("I_Coefficient", 255, "I"),
    "d": ("D_Coefficient", 255, "D"),
    "startup_force": ("Minimum_Startup_Force", 1000, "Startup"),
    "max_torque": ("Max_Torque_Limit", 1000, "MaxTorque"),
}
PID_KEYS = ("p", "i", "d")


def bounded(key: str):
    max_value = REGISTERS[key][1]

    def parse(value: str) -> int:
        v = int(value)
        if not 0 <= v <= max_value:
            raise argparse.ArgumentTypeError(f"{key} must be in 0..{max_value}, got {v}")
        return v

    return parse


def parse_set(spec: str) -> tuple[str, dict[str, int]]:
    """Parse 'MOTOR=P,I,D' where any of P, I, D may be empty."""
    try:
        motor, values = spec.split("=", 1)
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"expected MOTOR=P,I,D, got '{spec}'") from e
    fields = values.split(",")
    if motor not in MOTORS:
        raise argparse.ArgumentTypeError(f"unknown motor '{motor}', choose from {list(MOTORS)}")
    if len(fields) != 3:
        raise argparse.ArgumentTypeError(f"expected 3 comma-separated values in '{spec}'")
    gains = {k: bounded(k)(f) for k, f in zip(PID_KEYS, fields, strict=True) if f.strip() != ""}
    return motor, gains


def read_values(bus: SerialMotorsBus, motors: list[str]) -> dict[str, dict[str, int]]:
    return {m: {k: int(bus.read(reg, m)) for k, (reg, _, _) in REGISTERS.items()} for m in motors}


def print_values(title: str, values: dict[str, dict[str, int]]) -> None:
    print(f"\n{title}")
    print(f"  {'motor':<15}" + "".join(f"{header:>11}" for _, _, header in REGISTERS.values()))
    for motor, v in values.items():
        print(f"  {motor:<15}" + "".join(f"{v[k]:>11}" for k in REGISTERS))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", required=True, help="Serial port of the arm, e.g. /dev/ttyACM1")
    parser.add_argument(
        "--motors", nargs="+", choices=list(MOTORS), default=list(MOTORS), help="Motors the options below apply to"
    )
    parser.add_argument("--p", type=bounded("p"), help="P coefficient (0-255)")
    parser.add_argument("--i", type=bounded("i"), help="I coefficient (0-255)")
    parser.add_argument("--d", type=bounded("d"), help="D coefficient (0-255)")
    parser.add_argument("--startup-force", type=bounded("startup_force"), help="Minimum_Startup_Force (0-1000)")
    parser.add_argument("--max-torque", type=bounded("max_torque"), help="Max_Torque_Limit (0-1000, 0.1 %% steps)")
    parser.add_argument(
        "--set", type=parse_set, action="append", default=[], metavar="MOTOR=P,I,D", help="Per-motor gains"
    )
    args = parser.parse_args()

    # Values to write per motor: the common options for --motors first, then --set overrides.
    targets: dict[str, dict[str, int]] = {}
    common = {k: getattr(args, k) for k in REGISTERS if getattr(args, k) is not None}
    if common:
        for motor in args.motors:
            targets.setdefault(motor, {}).update(common)
    for motor, gains in args.set:
        targets.setdefault(motor, {}).update(gains)

    bus = SerialMotorsBus(
        port=args.port,
        motors={name: Motor(id_, "sts3215", MotorNormMode.RANGE_M100_100) for name, id_ in MOTORS.items()},
    )
    bus.connect()
    try:
        all_motors = list(MOTORS)
        print_values("Current values:", read_values(bus, all_motors))

        if not targets:
            print("\nNothing to write (pass --p/--i/--d/--startup-force/--max-torque or --set).")
            return

        # Torque off opens the EEPROM lock. It is deliberately left off afterwards, so the arm
        # does not jump to a stale Goal_Position: hold the arm or put it in a rest position.
        bus.disable_torque(list(targets))
        for motor, values in targets.items():
            for k, value in values.items():
                bus.write(REGISTERS[k][0], motor, value)
                if k == "max_torque":
                    # Max_Torque_Limit (EEPROM) is only copied to Torque_Limit (RAM) at power-up.
                    bus.write("Torque_Limit", motor, value)

        print_values("New values:", read_values(bus, all_motors))
        print("\nTorque is off. It is enabled again by the next lerobot command that connects to the arm.")
    finally:
        bus.disconnect(disable_torque=False)


if __name__ == "__main__":
    main()
