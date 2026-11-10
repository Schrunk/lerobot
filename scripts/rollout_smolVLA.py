"""Reale Inferenz von SmolVLA auf einem SO-101 Follower mit Real-Time Chunking (RTC).

Ein Hintergrund-Thread berechnet so schnell wie möglich neue Action-Chunks, während der
Haupt-Thread mit fester Frequenz Aktionen aus der Queue an den Roboter sendet. Jeder neue
Chunk wird per RTC-Guidance an den noch nicht ausgeführten Rest des alten Chunks angeglichen
und ersetzt ihn dann (die während der Inferenz bereits ausgeführten Schritte werden übersprungen).

Alle anzupassenden Parameter stehen im Block KONFIGURATION direkt unter den Imports.
"""

import math
import threading
import time

import numpy as np
import torch

from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.configs import RTCAttentionSchedule
from lerobot.policies import make_pre_post_processors
from lerobot.policies.rtc import ActionQueue, LatencyTracker, RTCConfig
from lerobot.policies.smolvla import SmolVLAPolicy
from lerobot.policies.utils import build_inference_frame, make_robot_action
from lerobot.robots import make_robot_from_config
from lerobot.robots.so_follower import SO101FollowerConfig
from lerobot.utils.feature_utils import hw_to_dataset_features

# =============================================================================
# KONFIGURATION – hier alles anpassen
# =============================================================================

# Policy
POLICY_PATH = "/home/student/lerobot_franz/outputs/train/2026-10-27/13-38-37_smolVLA_FMB/checkpoints/018000/pretrained_model"
TASK_PROMPT = "pick up yellow cuboid and place it left from the assembly board"
DEVICE = "cuda"

# Roboter
ROBOT_PORT = "/dev/ttyACM1"
ROBOT_ID = "my_follower_arm"  # bestimmt, welche Kalibrierdatei geladen wird
ROBOT_TYPE = "so101_follower"  # wird als "robot_type" an die Policy übergeben

# Kameras: Name -> Index/Pfad. Die Namen müssen denen aus dem Trainingsdatensatz entsprechen!
CAMERAS = {
    "camera1": 4,
    "camera2": 2,
}
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30

# Rollout
NUM_STEPS = 600
WARMUP_STEPS = 3  # Warmup-Chunks vor dem Start; Aktionen werden dabei NICHT gesendet
CONTROL_FPS = 30

# Real-Time Chunking
RTC_EXECUTION_HORIZON = 10  # Anzahl Schritte des alten Chunks, an die der neue angeglichen wird
RTC_MAX_GUIDANCE_WEIGHT = 10.0  # Stärke der Angleichung (10.0 passt für 10 Flow-Matching-Schritte)
RTC_SCHEDULE = RTCAttentionSchedule.EXP  # Gewichtsverlauf im Überlappungsbereich: LINEAR, EXP, ONES, ZEROS
# Neuer Chunk wird berechnet, sobald höchstens so viele Aktionen in der Queue übrig sind.
# None = chunk_size der Policy -> neue Chunks werden ohne Pause direkt hintereinander berechnet.
QUEUE_THRESHOLD: int | None = None

# Auswertung
THRESHOLD_MS = 100.0  # Grenze für die Inferenzzeit pro Chunk

# =============================================================================


def make_robot_config() -> SO101FollowerConfig:
    cameras = {
        name: OpenCVCameraConfig(
            index_or_path=index_or_path,
            width=CAMERA_WIDTH,
            height=CAMERA_HEIGHT,
            fps=CAMERA_FPS,
        )
        for name, index_or_path in CAMERAS.items()
    }
    return SO101FollowerConfig(port=ROBOT_PORT, id=ROBOT_ID, cameras=cameras)


def print_latency_report(
    chunk_latencies_ms: list[float],
    chunk_delays: list[int],
    starved_steps: int,
    sent_steps: int,
    threshold_ms: float,
) -> dict:
    if not chunk_latencies_ms:
        print("Keine Latenzen gemessen.")
        return {"latencies_ms": []}

    latencies = np.array(chunk_latencies_ms)
    avg_latency = float(np.mean(latencies))
    min_latency = float(np.min(latencies))
    max_latency = float(np.max(latencies))
    count_above_threshold = int(np.sum(latencies > threshold_ms))
    total_measured = len(latencies)

    print("\n" + "=" * 50)
    print("        RTC-INFERENZ LATENZ-AUSWERTUNG          ")
    print("=" * 50)
    print(f"Berechnete Chunks:        {total_measured}")
    print(f"Durchschnittliche Zeit:   {avg_latency:.2f} ms")
    print(f"Minimale Zeit:            {min_latency:.2f} ms")
    print(f"Maximale Zeit:            {max_latency:.2f} ms")
    print(
        f"Chunks > {threshold_ms:.1f} ms:        {count_above_threshold} / {total_measured} "
        f"({(count_above_threshold / total_measured) * 100:.1f}%)"
    )
    print(f"Verzögerung (Schritte):   Ø {np.mean(chunk_delays):.1f}, max {max(chunk_delays)}")
    print(f"Gesendete Aktionen:       {sent_steps}")
    print(f"Ticks mit leerer Queue:   {starved_steps}")
    print("=" * 50)

    return {
        "latencies_ms": chunk_latencies_ms,
        "avg_ms": avg_latency,
        "min_ms": min_latency,
        "max_ms": max_latency,
        "over_threshold_count": count_above_threshold,
        "delays": chunk_delays,
        "starved_steps": starved_steps,
    }


def run_real_robot_rollout() -> dict:
    assert torch.cuda.is_available(), "Keine CUDA-GPU gefunden."
    device = torch.device(DEVICE)
    dt = 1.0 / CONTROL_FPS

    # 1. Policy mit aktiviertem RTC + Pre-/Postprocessor (Normalisierung, Tokenizer, Device) laden
    print(f"Lade Policy: {POLICY_PATH}...")
    policy = SmolVLAPolicy.from_pretrained(POLICY_PATH)
    rtc_config = RTCConfig(
        enabled=True,
        execution_horizon=RTC_EXECUTION_HORIZON,
        max_guidance_weight=RTC_MAX_GUIDANCE_WEIGHT,
        prefix_attention_schedule=RTC_SCHEDULE,
    )
    policy.config.rtc_config = rtc_config
    policy.init_rtc_processor()
    policy.to(device)
    policy.eval()
    preprocess, postprocess = make_pre_post_processors(
        policy.config,
        POLICY_PATH,
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )
    queue_threshold = QUEUE_THRESHOLD if QUEUE_THRESHOLD is not None else policy.config.chunk_size

    # 2. Hardware initialisieren
    print(f"Verbinde mit Roboter ({ROBOT_TYPE}) und Kameras...")
    robot = make_robot_from_config(make_robot_config())
    robot.connect()

    # Mappt die Roh-Keys des Roboters auf die vom Modell erwarteten Feature-Keys
    dataset_features = {
        **hw_to_dataset_features(robot.action_features, "action"),
        **hw_to_dataset_features(robot.observation_features, "observation"),
    }

    def predict_chunk(
        obs: dict, inference_delay: int, prev_actions: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Berechnet einen Chunk. Gibt (normalisiert für RTC, denormalisiert für den Roboter) zurück."""
        obs_frame = build_inference_frame(
            observation=obs,
            ds_features=dataset_features,
            device=device,
            task=TASK_PROMPT,
            robot_type=ROBOT_TYPE,
        )
        # Kein inference_mode: die RTC-Guidance braucht intern Autograd (torch.enable_grad)
        with torch.no_grad():
            actions = policy.predict_action_chunk(
                preprocess(obs_frame), inference_delay=inference_delay, prev_chunk_left_over=prev_actions
            )
            original = actions.squeeze(0).clone()
            processed = postprocess(actions).squeeze(0)
        return original, processed

    # Zwischen den Threads geteilter Zustand
    action_queue = ActionQueue(rtc_config)
    latest_obs: dict = {"obs": None}
    obs_lock = threading.Lock()
    stop_event = threading.Event()
    inference_error: list[BaseException] = []

    latency_tracker = LatencyTracker()
    chunk_latencies_ms: list[float] = []
    chunk_delays: list[int] = []

    def inference_loop() -> None:
        """Berechnet im Hintergrund fortlaufend neue Chunks und verrechnet sie mit der Queue."""
        try:
            while not stop_event.is_set():
                with obs_lock:
                    obs = latest_obs["obs"]
                if obs is None or action_queue.qsize() > queue_threshold:
                    time.sleep(0.001)
                    continue

                start = time.perf_counter()
                idx_before = action_queue.get_action_index()
                prev_actions = action_queue.get_left_over()
                if prev_actions is not None and prev_actions.numel() == 0:
                    prev_actions = None
                # Geschätzte Anzahl Schritte, die der Roboter während dieser Inferenz ausführt
                delay = math.ceil(latency_tracker.max() / dt)

                original, processed = predict_chunk(obs, delay, prev_actions)
                torch.cuda.synchronize()

                latency = time.perf_counter() - start
                real_delay = math.ceil(latency / dt)
                latency_tracker.add(latency)
                chunk_latencies_ms.append(latency * 1000)
                chunk_delays.append(real_delay)

                # Ersetzt den Rest des alten Chunks durch den neuen, ohne die bereits ausgeführten Schritte
                action_queue.merge(original, processed, real_delay, idx_before)
        except BaseException as e:  # noqa: BLE001 - wird im Haupt-Thread erneut ausgelöst
            inference_error.append(e)
            stop_event.set()

    inference_thread = threading.Thread(target=inference_loop, daemon=True, name="RTCInference")
    sent_steps = 0
    starved_steps = 0

    try:
        # 3. Warmup (Sicherheit: Aktionen werden hier noch NICHT gesendet!)
        print(f"Führe {WARMUP_STEPS} Warmup-Chunks auf echten Kamerabildern aus...")
        policy.reset()
        for _ in range(WARMUP_STEPS):
            predict_chunk(robot.get_observation(), inference_delay=0, prev_actions=None)
        torch.cuda.synchronize()
        policy.reset()

        print(f"Starte Live-Rollout ({NUM_STEPS} Schritte bei {CONTROL_FPS} Hz, RTC aktiv)...")
        inference_thread.start()

        # 4. Kontrollschleife: Beobachtung veröffentlichen, nächste Aktion aus der Queue senden
        while sent_steps < NUM_STEPS and not stop_event.is_set():
            loop_start = time.perf_counter()

            obs = robot.get_observation()
            with obs_lock:
                latest_obs["obs"] = obs

            action = action_queue.get()
            if action is not None:
                robot.send_action(make_robot_action(action, dataset_features))
                sent_steps += 1
            elif sent_steps > 0:
                # Inferenz war langsamer als die Queue lang war -> Roboter bleibt kurz stehen
                starved_steps += 1

            # Feste Kontrollfrequenz halten
            sleep_time = dt - (time.perf_counter() - loop_start)
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\nRollout manuell abgebrochen.")
    finally:
        stop_event.set()
        if inference_thread.is_alive():
            inference_thread.join(timeout=5.0)
        print("Trenne Roboterverbindung...")
        robot.disconnect()

    if inference_error:
        raise inference_error[0]

    # 5. Statistische Auswertung
    return print_latency_report(chunk_latencies_ms, chunk_delays, starved_steps, sent_steps, THRESHOLD_MS)


if __name__ == "__main__":
    run_real_robot_rollout()
