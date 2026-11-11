#!/usr/bin/env python3

import json
import math
from pathlib import Path

from lerobot.datasets.lerobot_dataset import LeRobotDataset


SETTINGS_FILE = Path(__file__).with_name("training_estimator_settings.json")


def load_settings() -> dict:
    try:
        return json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save_settings(settings: dict) -> None:
    SETTINGS_FILE.write_text(
        json.dumps(settings, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def ask_positive_int(prompt: str, default: int | None = None) -> int:
    while True:
        suffix = f" [{default}]" if default is not None else ""
        raw = input(f"{prompt}{suffix}: ").strip()

        if not raw and default is not None:
            return default

        try:
            value = int(raw)
            if value > 0:
                return value
            print("Bitte eine ganze Zahl größer als 0 eingeben.")
        except ValueError:
            print("Ungültige Eingabe. Bitte eine ganze Zahl eingeben.")


def ask_positive_float(prompt: str, default: float | None = None) -> float:
    while True:
        suffix = f" [{default}]" if default is not None else ""
        raw = input(f"{prompt}{suffix}: ").strip()

        if not raw and default is not None:
            return default

        try:
            value = float(raw.replace(",", "."))
            if value > 0:
                return value
            print("Bitte eine Zahl größer als 0 eingeben.")
        except ValueError:
            print("Ungültige Eingabe. Beispiel: 0.35")


def main() -> None:
    settings = load_settings()

    previous_path = settings.get("dataset_path")
    path_prompt = "Pfad zum lokalen LeRobot-Dataset-Verzeichnis"
    if previous_path:
        path_prompt += f" [{previous_path}]"

    dataset_input = input(f"{path_prompt}: ").strip()
    if not dataset_input and previous_path:
        dataset_input = previous_path

    dataset_path = Path(dataset_input).expanduser()
    if not dataset_path.exists():
        raise FileNotFoundError(f"Pfad nicht gefunden: {dataset_path}")

    dataset = LeRobotDataset(
        repo_id="local/dataset",
        root=dataset_path,
    )

    total_frames = dataset.meta.total_frames
    total_episodes = dataset.meta.total_episodes
    fps = dataset.meta.fps

    print("\nDataset-Informationen")
    print("---------------------")
    print(f"Dataset-Pfad:    {dataset_path}")
    print(f"Episoden:        {total_episodes:,}")
    print(f"Frames gesamt:   {total_frames:,}")
    print(f"FPS:             {fps}")

    if total_frames == 0:
        print("\nDas Dataset enthält laut Metadaten keine Frames.")
        return

    print("\nTrainingsparameter (Enter übernimmt den letzten Wert)")
    print("------------------------------------------------------")

    num_gpus = ask_positive_int("Anzahl GPUs", settings.get("num_gpus"))
    batch_size = ask_positive_int(
        "Batchgröße pro GPU", settings.get("batch_size")
    )
    epochs = ask_positive_int("Anzahl Epochen", settings.get("epochs"))
    seconds_per_step = ask_positive_float(
        "Dauer eines Steps in Sekunden",
        settings.get("seconds_per_step"),
    )

    # Die zuletzt verwendeten Eingaben für den nächsten Start speichern.
    settings.update(
        {
            "dataset_path": str(dataset_path),
            "num_gpus": num_gpus,
            "batch_size": batch_size,
            "epochs": epochs,
            "seconds_per_step": seconds_per_step,
        }
    )
    save_settings(settings)

    effective_batch_size = num_gpus * batch_size
    steps_per_epoch = math.ceil(total_frames / effective_batch_size)
    total_steps = epochs * steps_per_epoch
    wall_clock_seconds = total_steps * seconds_per_step

    hours, remainder = divmod(int(wall_clock_seconds), 3600)
    minutes, seconds = divmod(remainder, 60)

    print("\nSchätzung")
    print("---------")
    print(f"Frames gesamt:              {total_frames:,}")
    print(f"GPUs:                       {num_gpus}")
    print(f"Batchgröße pro GPU:         {batch_size}")
    print(f"Effektive Batchgröße:       {effective_batch_size}")
    print(f"Schritte pro Epoche:        {steps_per_epoch:,}")
    print(f"Epochen:                    {epochs}")
    print(f"Trainingsschritte gesamt:   {total_steps:,}")
    print(f"Dauer pro Step:             {seconds_per_step:g} s")
    print(
        "Geschätzte Wall-clock:     "
        f"{wall_clock_seconds:,.1f} s "
        f"({hours} h {minutes} min {seconds} s)"
    )
    print(f"\nEingaben gespeichert in: {SETTINGS_FILE}")


if __name__ == "__main__":
    main()