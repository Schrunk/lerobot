#!/usr/bin/env python
"""Continuously read the load (torque), current, temperature and position of an SO-101 arm's motors.

The script only reads: it does not change the torque state or any register, so you can move the
arm by hand (torque off) or run it while the motors hold a position.
The serial port can only be opened by one program, so it cannot run at the same time as
`lerobot-teleoperate` / `lerobot-record` on the same arm.

Examples:
    uv run python scripts/read_torque.py --port /dev/ttyACM1
    uv run python scripts/read_torque.py --port /dev/ttyACM1 --motors shoulder_lift elbow_flex --hz 50
    uv run python scripts/read_torque.py --port /dev/ttyACM1 --csv torque_log.csv

Columns:
    Load    Present_Load in % of the torque limit, signed by direction (raw value / 10).
    Current Present_Current in mA (raw value * 6.5 mA, STS3215).
    Temp    Present_Temperature in °C.
    Pos     Present_Position in raw steps (0..4095), uncalibrated.

Stop with Ctrl+C.
"""

import argparse
import csv
import sys
import time

from lerobot.motors import Motor, MotorNormMode, SerialMotorsBus

MOTORS = {
    "shoulder_pan": 1,
    "shoulder_lift": 2,
    "elbow_flex": 3,
    "wrist_flex": 4,
    "wrist_roll": 5,
    "gripper": 6,
}
CURRENT_MA_PER_STEP = 6.5  # STS3215


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", required=True, help="Serial port of the arm, e.g. /dev/ttyACM1")
    parser.add_argument("--motors", nargs="+", choices=list(MOTORS), default=list(MOTORS), help="Motors to read")
    parser.add_argument("--hz", type=float, default=10.0, help="Read rate in Hz (default 10)")
    parser.add_argument("--csv", help="Also log every reading to this CSV file")
    args = parser.parse_args()

    motors = args.motors
    bus = SerialMotorsBus(
        port=args.port,
        motors={name: Motor(MOTORS[name], "sts3215", MotorNormMode.RANGE_M100_100) for name in motors},
    )
    bus.connect()

    csv_file = open(args.csv, "w", newline="") if args.csv else None  # noqa: SIM115
    writer = None
    if csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            ["time_s"]
            + [f"{m}.{q}" for m in motors for q in ("load_pct", "current_ma", "temp_c", "pos_raw")]
        )

    period = 1.0 / args.hz
    interactive = sys.stdout.isatty()
    header = f"{'motor':<15}{'Load %':>9}{'Current mA':>12}{'Temp °C':>9}{'Pos':>7}"
    t0 = time.perf_counter()
    try:
        while True:
            start = time.perf_counter()
            load = bus.sync_read("Present_Load", motors, normalize=False)
            current = bus.sync_read("Present_Current", motors, normalize=False)
            temp = bus.sync_read("Present_Temperature", motors, normalize=False)
            pos = bus.sync_read("Present_Position", motors, normalize=False)
            t = start - t0

            rows = {
                m: (load[m] / 10, current[m] * CURRENT_MA_PER_STEP, int(temp[m]), int(pos[m])) for m in motors
            }

            lines = [f"t = {t:7.2f} s", header]
            for m, (ld, cur, tc, p) in rows.items():
                lines.append(f"{m:<15}{ld:>9.1f}{cur:>12.0f}{tc:>9}{p:>7}")
            if interactive:
                # Redraw the table in place.
                sys.stdout.write("\033[H\033[J" + "\n".join(lines) + "\n")
            else:
                sys.stdout.write("\n".join(lines) + "\n\n")
            sys.stdout.flush()

            if writer:
                writer.writerow([f"{t:.3f}"] + [v for m in motors for v in rows[m]])

            time.sleep(max(0.0, period - (time.perf_counter() - start)))
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        if csv_file:
            csv_file.close()
            print(f"Log written to {args.csv}")
        bus.disconnect(disable_torque=False)


if __name__ == "__main__":
    main()
