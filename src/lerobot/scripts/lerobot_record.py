# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

"""
Records a dataset via teleoperation.  This is a pure data-collection
tool — no policy inference.  For deploying trained policies, use
``lerobot-rollout`` instead.

Requires: pip install 'lerobot[core_scripts]'  (includes dataset + hardware + viz extras)

Example:

```shell
lerobot-record \\
    --robot.type=so100_follower \\
    --robot.port=/dev/tty.usbmodem58760431541 \\
    --robot.cameras="{laptop: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" \\
    --robot.id=black \\
    --teleop.type=so100_leader \\
    --teleop.port=/dev/tty.usbmodem58760431551 \\
    --teleop.id=blue \\
    --dataset.repo_id=<my_username>/<my_dataset_name> \\
    --dataset.num_episodes=2 \\
    --dataset.single_task="Grab the cube" \\
    --dataset.streaming_encoding=true \\
    --dataset.encoder_threads=2 \\
    --display_data=true
```

To stream the data to Foxglove instead of Rerun, add ``--display_mode=foxglove`` (then connect the
Foxglove app to ``ws://127.0.0.1:8765``; override the port with ``--display_port=<port>``).

Example recording with bimanual so100:
```shell
lerobot-record \\
  --robot.type=bi_so_follower \\
  --robot.left_arm_config.port=/dev/tty.usbmodem5A460822851 \\
  --robot.right_arm_config.port=/dev/tty.usbmodem5A460814411 \\
  --robot.id=bimanual_follower \\
  --robot.left_arm_config.cameras='{
    wrist: {"type": "opencv", "index_or_path": 1, "width": 640, "height": 480, "fps": 30},
    top: {"type": "opencv", "index_or_path": 3, "width": 640, "height": 480, "fps": 30},
  }' --robot.right_arm_config.cameras='{
    wrist: {"type": "opencv", "index_or_path": 2, "width": 640, "height": 480, "fps": 30},
    front: {"type": "opencv", "index_or_path": 4, "width": 640, "height": 480, "fps": 30},
  }' \\
  --teleop.type=bi_so_leader \\
  --teleop.left_arm_config.port=/dev/tty.usbmodem5A460852721 \\
  --teleop.right_arm_config.port=/dev/tty.usbmodem5A460819811 \\
  --teleop.id=bimanual_leader \\
  --display_data=true \\
  --dataset.repo_id=${HF_USER}/bimanual-so-handover-cube \\
  --dataset.num_episodes=25 \\
  --dataset.single_task="Grab and handover the red cube to the other arm" \\
  --dataset.streaming_encoding=true \\
  --dataset.encoder_threads=2
```

Example recording with custom video encoding parameters:
```shell
lerobot-record \\
    --robot.type=so100_follower \\
    --robot.port=/dev/tty.usbmodem58760431541 \\
    --robot.cameras="{laptop: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" \\
    --robot.id=black \\
    --teleop.type=so100_leader \\
    --teleop.port=/dev/tty.usbmodem58760431551 \\
    --teleop.id=blue \\
    --dataset.repo_id=<my_username>/<my_dataset_name> \\
    --dataset.num_episodes=2 \\
    --dataset.single_task="Grab the cube" \\
    --dataset.streaming_encoding=true \\
    --dataset.encoder_threads=2 \\
    --dataset.rgb_encoder.vcodec=h264 \\
    --dataset.rgb_encoder.preset=fast \\
    --dataset.rgb_encoder.extra_options={"tune": "film", "profile:v": "high", "bf": 2} \\
    --display_data=true
```
"""

import logging
import time
from contextlib import ExitStack
from dataclasses import asdict, dataclass
from pathlib import Path
from pprint import pformat

from huggingface_hub.errors import HFValidationError
from huggingface_hub.utils import validate_repo_id

from lerobot.cameras import CameraConfig  # noqa: F401
from lerobot.cameras.opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.reachy2_camera import Reachy2CameraConfig  # noqa: F401
from lerobot.cameras.realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.cameras.zmq import ZMQCameraConfig  # noqa: F401
from lerobot.common.control_utils import sanity_check_dataset_robot_compatibility
from lerobot.configs import parser
from lerobot.configs.dataset import DatasetRecordConfig
from lerobot.datasets import (
    LeRobotDataset,
    VideoEncodingManager,
    aggregate_pipeline_dataset_features,
    create_initial_features,
    safe_stop_image_writer,
)
from lerobot.processor import (
    RobotAction,
    RobotObservation,
    RobotProcessorPipeline,
    make_default_processors,
)
from lerobot.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    bi_openarm_follower,
    bi_rebot_b601_follower,
    bi_so_follower,
    bi_yam_follower,
    earthrover_mini_plus,
    hope_jr,
    koch_follower,
    make_robot_from_config,
    omx_follower,
    openarm_follower,
    reachy2,
    rebot_b601_follower,
    sim_robot,
    so_follower,
    unitree_g1 as unitree_g1_robot,
    yam_follower,
)
from lerobot.robots.utils import check_control_rate, episode_ended, start_episode
from lerobot.teleoperators import (  # noqa: F401
    Teleoperator,
    TeleoperatorConfig,
    bi_openarm_leader,
    bi_openarm_mini,
    bi_rebot_102_leader,
    bi_so_leader,
    homunculus,
    koch_leader,
    make_teleoperator_from_config,
    omx_leader,
    openarm_leader,
    openarm_mini,
    reachy2_teleoperator,
    rebot_102_leader,
    so_leader,
    unitree_g1,
)
from lerobot.teleoperators.keyboard import KeyboardTeleop
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.cycle_timer import CycleTimer
from lerobot.utils.feature_utils import build_dataset_frame, combine_feature_dicts
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.keyboard_input import init_keyboard_listener
from lerobot.utils.motor_telemetry import MotorTelemetry
from lerobot.utils.utils import (
    init_logging,
    log_say,
)
from lerobot.utils.visualization_utils import (
    init_visualization,
    log_visualization_data,
    shutdown_visualization,
)


@dataclass
class RecordConfig:
    robot: RobotConfig
    dataset: DatasetRecordConfig
    # Teleoperator to control the robot (required)
    teleop: TeleoperatorConfig | None = None
    # Display all cameras on screen
    display_data: bool = False
    # Visualization backend used when display_data is True: "rerun" or "foxglove".
    display_mode: str = "rerun"
    # For "rerun": IP of a remote server to send to. For "foxglove": interface to bind the WebSocket
    # server to (127.0.0.1 for local only, 0.0.0.0 for all interfaces).
    display_ip: str | None = None
    # For "rerun": port of the remote server. For "foxglove": port to bind the WebSocket server to.
    display_port: int | None = None
    # Whether to display compressed (JPEG) images instead of raw frames
    display_compressed_images: bool = False
    # Use vocal synthesis to read events.
    play_sounds: bool = True
    # Resume recording on an existing dataset.
    resume: bool = False
    # Also record a second dataset, named `<repo_id><current_dataset_suffix>`, with the same frames
    # plus the current of every motor (in mA) in `observation.state`, after the joint positions, as
    # "<motor>.current" (e.g. "shoulder_pan.current"). The main dataset keeps the plain 6-DoF state.
    # The current is read from the motor bus in every frame.
    record_motor_current: bool = False
    # Suffix of the dataset with motor current. Note: "+" is not allowed in Hugging Face Hub repo ids,
    # so with the default suffix that dataset is only stored locally (push_to_hub skips it).
    current_dataset_suffix: str = "+current"

    def __post_init__(self):
        if self.teleop is None:
            raise ValueError(
                "A teleoperator is required for recording. "
                "Use --teleop.type=... to specify one. "
                "For policy-based deployment, use lerobot-rollout instead."
            )


""" --------------- record_loop() data flow --------------------------
       [ Robot ]
           V
     [ robot.get_observation() ] ---> raw_obs
           V
     [ robot_observation_processor ] ---> processed_obs
           V
     [ Teleoperator ]
     |
     |  [teleop.get_action] -> raw_action
     |          |
     |          V
     | [teleop_action_processor]
     |          |
     '---> processed_teleop_action
                               V
                  [ robot_action_processor ] --> robot_action_to_send
                               V
                    [ robot.send_action() ] -- (Robot Executes)
                               V
                    ( Save to Dataset )
                               V
                  ( Rerun Log / Loop Wait )
"""


@safe_stop_image_writer
def record_loop(
    robot: Robot,
    events: dict[str, bool],
    fps: int,
    teleop_action_processor: RobotProcessorPipeline[
        tuple[RobotAction, RobotObservation], RobotAction
    ],  # runs after teleop
    robot_action_processor: RobotProcessorPipeline[
        tuple[RobotAction, RobotObservation], RobotAction
    ],  # runs before robot
    robot_observation_processor: RobotProcessorPipeline[
        RobotObservation, RobotObservation
    ],  # runs after robot
    dataset: LeRobotDataset | None = None,
    teleop: Teleoperator | list[Teleoperator] | None = None,
    control_time_s: float | None = None,
    single_task: str | None = None,
    display_data: bool = False,
    display_mode: str = "rerun",
    display_compressed_images: bool = False,
    timer: CycleTimer | None = None,
    motor_current: MotorTelemetry | None = None,
    current_dataset: LeRobotDataset | None = None,
) -> None:
    """Drive the robot from the teleoperator at *fps*, optionally recording each frame.

    *timer* lets a caller that runs several phases — :func:`record` records one episode
    per call, with an unrecorded reset phase in between — keep one
    :class:`~lerobot.utils.cycle_timer.CycleTimer` across all of them, so the cadence
    statistics span the whole session and are reported per episode.  Without it each
    call gets a private timer: identical pacing and identical slow-loop warnings, just
    no end-of-run summary, since a single phase has no run to summarise.

    *motor_current*, if given, is read every frame and its values ("<motor>.current", in mA)
    are added to the raw robot observation. *current_dataset* gets the same frames as *dataset*,
    but with its own features, so the currents end up in its ``observation.state`` (while
    *dataset*, whose features don't list them, keeps the plain joint state).
    """
    for ds in (dataset, current_dataset):
        if ds is not None and ds.fps != fps:
            raise ValueError(f"The dataset fps should be equal to requested fps ({ds.fps} != {fps}).")
    if current_dataset is not None and dataset is None:
        raise ValueError("`current_dataset` is only recorded together with `dataset`.")
    if control_time_s is None:
        raise ValueError("`control_time_s` is required: it is how long the loop runs.")

    # Multi-teleop is LeKiwi-only: one arm leader plus a keyboard that drives the base.
    multi_teleop: tuple[Teleoperator, KeyboardTeleop] | None = None
    if isinstance(teleop, list):
        teleop_keyboard = next((t for t in teleop if isinstance(t, KeyboardTeleop)), None)
        teleop_arm = next(
            (
                t
                for t in teleop
                if isinstance(
                    t,
                    (
                        so_leader.SO100Leader
                        | so_leader.SO101Leader
                        | koch_leader.KochLeader
                        | omx_leader.OmxLeader
                    ),
                )
            ),
            None,
        )

        if not (teleop_arm and teleop_keyboard and len(teleop) == 2 and robot.name == "lekiwi_client"):
            raise ValueError(
                "For multi-teleop, the list must contain exactly one KeyboardTeleop and one arm teleoperator. Currently only supported for LeKiwi robot."
            )
        multi_teleop = (teleop_arm, teleop_keyboard)

    if timer is None:
        timer = CycleTimer(fps, records_data=dataset is not None)

    current_keys = list(motor_current.observation_features()) if motor_current is not None else []

    no_action_count = 0
    timestamp = 0.0
    start_episode_t = time.perf_counter()
    while timestamp < control_time_s:
        # Checked before `tick()`: this iteration is not a control tick, so it should not
        # be timed as one.
        if events["exit_early"]:
            events["exit_early"] = False
            break

        timer.tick()

        with timer.section("observe"):
            # Get robot observation
            obs = robot.get_observation()

            if motor_current is not None:
                current_obs = MotorTelemetry.as_observation(motor_current.read())
                # A motor whose current could never be read yet gets 0, so the frame stays complete.
                obs = {**obs, **{k: current_obs.get(k, 0.0) for k in current_keys}}

        with timer.section("process_obs"):
            # Applies a pipeline to the raw robot observation, default is IdentityProcessor
            obs_processed = robot_observation_processor(obs)

            if dataset is not None:
                observation_frame = build_dataset_frame(dataset.features, obs_processed, prefix=OBS_STR)
            if current_dataset is not None:
                current_observation_frame = build_dataset_frame(
                    current_dataset.features, obs_processed, prefix=OBS_STR
                )

        with timer.section("teleop"):
            # Get action from teleop
            if isinstance(teleop, Teleoperator):
                act = teleop.get_action()
                if robot.name == "unitree_g1":
                    teleop.send_feedback(obs)

                # Applies a pipeline to the raw teleop action, default is IdentityProcessor
                act_processed_teleop = teleop_action_processor((act, obs))
                action_values = act_processed_teleop
                robot_action_to_send = robot_action_processor((act_processed_teleop, obs))

            elif multi_teleop is not None:
                arm_teleop, keyboard_teleop = multi_teleop
                arm_action = arm_teleop.get_action()
                arm_action = {f"arm_{k}": v for k, v in arm_action.items()}
                keyboard_action = keyboard_teleop.get_action()
                base_action = robot._from_keyboard_to_base_action(keyboard_action)  # type: ignore[attr-defined]
                act = {**arm_action, **base_action} if len(base_action) > 0 else arm_action
                act_processed_teleop = teleop_action_processor((act, obs))
                action_values = act_processed_teleop
                robot_action_to_send = robot_action_processor((act_processed_teleop, obs))
            else:
                robot_action_to_send = None
                no_action_count += 1
                if no_action_count == 1 or no_action_count % 10 == 0:
                    logging.warning(
                        "No teleoperator provided, skipping action generation. "
                        "This is likely to happen when resetting the environment without a teleop device. "
                        "The robot won't be at its rest position at the start of the next episode."
                    )

        # Nothing to send and nothing to record, but the phase still has to be paced and
        # still has to end: `continue`ing straight past the tail of the loop body used to
        # spin at full CPU speed on a `control_time_s` that never advanced.
        if robot_action_to_send is None:
            timer.wait()
            timestamp = time.perf_counter() - start_episode_t
            continue

        with timer.section("send"):
            # Send action to robot
            # Action can eventually be clipped using `max_relative_target`,
            # so action actually sent is saved in the dataset. action = postprocessor.process(action)
            # TODO(steven, pepijn, adil): we should use a pipeline step to clip the action, so the sent action is the action that we input to the robot.
            _sent_action = robot.send_action(robot_action_to_send)

        # Write to dataset
        if dataset is not None:
            with timer.section("record"):
                action_frame = build_dataset_frame(dataset.features, action_values, prefix=ACTION)
                frame = {**observation_frame, **action_frame, "task": single_task}
                dataset.add_frame(frame)
                if current_dataset is not None:
                    # A new dict: `add_frame` pops "task" from the frame it is given.
                    current_dataset.add_frame(
                        {**current_observation_frame, **action_frame, "task": single_task}
                    )

        if display_data:
            with timer.section("telemetry"):
                log_visualization_data(
                    display_mode,
                    observation=obs_processed,
                    action=action_values,
                    compress_images=display_compressed_images,
                )

        # End the recorded episode when the sim robot's episode ends (success, failure or step limit).
        if dataset is not None and episode_ended(robot):
            break

        timer.wait()

        timestamp = time.perf_counter() - start_episode_t


@parser.wrap()
def record(
    cfg: RecordConfig,
    teleop_action_processor: RobotProcessorPipeline | None = None,
    robot_action_processor: RobotProcessorPipeline | None = None,
    robot_observation_processor: RobotProcessorPipeline | None = None,
) -> LeRobotDataset:
    init_logging()
    logging.info(pformat(asdict(cfg)))
    if cfg.display_data:
        init_visualization(
            cfg.display_mode, session_name="recording", ip=cfg.display_ip, port=cfg.display_port
        )
    display_compressed_images = (
        True
        if (cfg.display_data and cfg.display_ip is not None and cfg.display_port is not None)
        else cfg.display_compressed_images
    )

    robot = make_robot_from_config(cfg.robot)
    check_control_rate(robot, cfg.dataset.fps, "--dataset.fps")
    teleop = make_teleoperator_from_config(cfg.teleop) if cfg.teleop is not None else None

    # Fall back to identity pipelines when the caller doesn't supply processors.
    if (
        teleop_action_processor is None
        or robot_action_processor is None
        or robot_observation_processor is None
    ):
        _t, _r, _o = make_default_processors()
        teleop_action_processor = teleop_action_processor or _t
        robot_action_processor = robot_action_processor or _r
        robot_observation_processor = robot_observation_processor or _o

    def make_features(observation_features: dict) -> dict:
        return combine_feature_dicts(
            aggregate_pipeline_dataset_features(
                pipeline=teleop_action_processor,
                initial_features=create_initial_features(
                    action=robot.action_features
                ),  # TODO(steven, pepijn): in future this should be come from teleop or policy
                use_videos=cfg.dataset.video,
            ),
            aggregate_pipeline_dataset_features(
                pipeline=robot_observation_processor,
                initial_features=create_initial_features(observation=observation_features),
                use_videos=cfg.dataset.video,
            ),
        )

    dataset_features = make_features(robot.observation_features)

    # Motor current, read every frame (rate 0), only for the second "+current" dataset.
    motor_current = None
    current_dataset_features = None
    if cfg.record_motor_current:
        motor_current = MotorTelemetry(robot, rate_hz=0, quantities=("current",))
        current_features = motor_current.observation_features()
        if current_features:
            current_dataset_features = make_features({**robot.observation_features, **current_features})
            logging.info(f"Recording motor current in the second dataset: {list(current_features)}")
        else:
            logging.warning("record_motor_current: the robot has no motor with a current register.")
            motor_current = None

    dataset = None
    current_dataset = None
    listener = None
    # One timer for the whole session, so its statistics describe the recording rather
    # than one episode's slice of it.  The reset phases below deliberately run on their
    # own private timers: they write no frames, so folding their ticks in would dilute
    # every number that answers "did I record at `fps`?".
    timer = CycleTimer(cfg.dataset.fps)

    def current_repo_and_root() -> tuple[str, Path | None]:
        """Name and local root of the "+current" dataset, next to the main one."""
        suffix = cfg.current_dataset_suffix
        root = Path(f"{cfg.dataset.root}{suffix}") if cfg.dataset.root is not None else None
        return f"{cfg.dataset.repo_id}{suffix}", root

    try:
        if cfg.resume:
            num_cameras = len(robot.cameras) if hasattr(robot, "cameras") else 0

            def resume(repo_id: str, root: str | Path | None) -> LeRobotDataset:
                return LeRobotDataset.resume(
                    repo_id,
                    root=root,
                    batch_encoding_size=cfg.dataset.video_encoding_batch_size,
                    rgb_encoder=cfg.dataset.rgb_encoder,
                    depth_encoder=cfg.dataset.depth_encoder,
                    encoder_threads=cfg.dataset.encoder_threads,
                    streaming_encoding=cfg.dataset.streaming_encoding,
                    encoder_queue_maxsize=cfg.dataset.encoder_queue_maxsize,
                    image_writer_processes=cfg.dataset.num_image_writer_processes if num_cameras > 0 else 0,
                    image_writer_threads=cfg.dataset.num_image_writer_threads_per_camera * num_cameras
                    if num_cameras > 0
                    else 0,
                )

            dataset = resume(cfg.dataset.repo_id, cfg.dataset.root)
            sanity_check_dataset_robot_compatibility(dataset, robot, cfg.dataset.fps, dataset_features)
            if current_dataset_features is not None:
                current_repo_id, current_root = current_repo_and_root()
                current_dataset = resume(current_repo_id, current_root)
                sanity_check_dataset_robot_compatibility(
                    current_dataset, robot, cfg.dataset.fps, current_dataset_features
                )
                if current_dataset.num_episodes != dataset.num_episodes:
                    raise ValueError(
                        f"'{current_repo_id}' has {current_dataset.num_episodes} episodes but "
                        f"'{cfg.dataset.repo_id}' has {dataset.num_episodes}, they must match to resume both."
                    )
        else:
            # Reject eval_ prefix — for policy evaluation use lerobot-rollout
            repo_name = cfg.dataset.repo_id.split("/", 1)[-1]
            if repo_name.startswith("eval_"):
                raise ValueError(
                    "Dataset names starting with 'eval_' are reserved for policy evaluation. "
                    "lerobot-record is for data collection only. Use lerobot-rollout for policy deployment."
                )
            cfg.dataset.stamp_repo_id()
            num_cameras = len(robot.cameras)  # type: ignore[attr-defined]

            def create(repo_id: str, root: str | Path | None, features: dict) -> LeRobotDataset:
                return LeRobotDataset.create(
                    repo_id,
                    cfg.dataset.fps,
                    root=root,
                    robot_type=robot.robot_type,
                    features=features,
                    use_videos=cfg.dataset.video,
                    image_writer_processes=cfg.dataset.num_image_writer_processes,
                    image_writer_threads=cfg.dataset.num_image_writer_threads_per_camera * num_cameras,
                    batch_encoding_size=cfg.dataset.video_encoding_batch_size,
                    rgb_encoder=cfg.dataset.rgb_encoder,
                    depth_encoder=cfg.dataset.depth_encoder,
                    encoder_threads=cfg.dataset.encoder_threads,
                    streaming_encoding=cfg.dataset.streaming_encoding,
                    encoder_queue_maxsize=cfg.dataset.encoder_queue_maxsize,
                )

            dataset = create(cfg.dataset.repo_id, cfg.dataset.root, dataset_features)
            if current_dataset_features is not None:
                current_repo_id, current_root = current_repo_and_root()
                current_dataset = create(current_repo_id, current_root, current_dataset_features)

        datasets = [ds for ds in (dataset, current_dataset) if ds is not None]
        for ds in datasets:
            logging.info(
                f"Recording to '{ds.repo_id}' at {ds.root}, observation.state = {ds.features['observation.state']['names']}"
            )

        # Connect the teleoperator before the robot so the robot isn't left idle (and possibly
        # tripping a firmware watchdog) during teleop init. Matches lerobot_teleoperate.py.
        if teleop is not None:
            teleop.connect()
        robot.connect()

        listener, events = init_keyboard_listener()

        if not cfg.dataset.streaming_encoding:
            logging.info(
                "Streaming encoding is disabled. If you have capable hardware, consider enabling it for way faster episode saving. --dataset.streaming_encoding=true --dataset.encoder_threads=2 # --dataset.rgb_encoder.vcodec=auto. More info in the documentation: https://huggingface.co/docs/lerobot/streaming_video_encoding"
            )

        with ExitStack() as stack:
            for ds in datasets:
                stack.enter_context(VideoEncodingManager(ds))
            recorded_episodes = 0
            while recorded_episodes < cfg.dataset.num_episodes and not events["stop_recording"]:
                episode_index = dataset.num_episodes
                start_episode(robot)
                log_say(f"Recording episode {episode_index}", cfg.play_sounds)
                record_loop(
                    robot=robot,
                    events=events,
                    fps=cfg.dataset.fps,
                    teleop_action_processor=teleop_action_processor,
                    robot_action_processor=robot_action_processor,
                    robot_observation_processor=robot_observation_processor,
                    teleop=teleop,
                    dataset=dataset,
                    current_dataset=current_dataset,
                    control_time_s=cfg.dataset.episode_time_s,
                    single_task=cfg.dataset.single_task,
                    display_data=cfg.display_data,
                    display_mode=cfg.display_mode,
                    display_compressed_images=display_compressed_images,
                    timer=timer,
                    motor_current=motor_current,
                )

                # Execute a few seconds without recording to give time to manually reset the environment
                # Skip reset for the last episode to be recorded
                if not events["stop_recording"] and (
                    (recorded_episodes < cfg.dataset.num_episodes - 1) or events["rerecord_episode"]
                ):
                    log_say("Reset the environment", cfg.play_sounds)

                    record_loop(
                        robot=robot,
                        events=events,
                        fps=cfg.dataset.fps,
                        teleop_action_processor=teleop_action_processor,
                        robot_action_processor=robot_action_processor,
                        robot_observation_processor=robot_observation_processor,
                        teleop=teleop,
                        control_time_s=cfg.dataset.reset_time_s,
                        single_task=cfg.dataset.single_task,
                        display_data=cfg.display_data,
                        display_mode=cfg.display_mode,
                        display_compressed_images=display_compressed_images,
                        motor_current=motor_current,
                    )

                if events["rerecord_episode"]:
                    log_say("Re-record episode", cfg.play_sounds)
                    events["rerecord_episode"] = False
                    events["exit_early"] = False
                    for ds in datasets:
                        ds.clear_episode_buffer()
                    timer.log_episode_summary("discarded episode")
                    timer.restart()
                    continue

                for ds in datasets:
                    ds.save_episode()
                recorded_episodes += 1
                # Close the window on the episode just saved.  The digest is emitted on
                # the next episode's first tick, so the reset phase, `save_episode` and
                # the spoken prompts in between are excluded from the cadence instead of
                # being charged to whichever episode they sit next to.  `restart()` then
                # exempts that first tick, whose cameras have been idle for seconds.
                timer.log_episode_summary(f"episode {episode_index}")
                timer.restart()
    finally:
        # First, and in `finally`: ^C is how most recording sessions end, and the summary
        # is most useful before the video encoding and the hub upload scroll it away.
        timer.log_run_summary()

        log_say("Stop recording", cfg.play_sounds, blocking=True)

        for ds in (dataset, current_dataset):
            if ds:
                ds.finalize()

        if robot.is_connected:
            robot.disconnect()
        if teleop and teleop.is_connected:
            teleop.disconnect()

        if listener is not None:
            listener.stop()

        if cfg.display_data:
            shutdown_visualization(cfg.display_mode)

        if cfg.dataset.push_to_hub:
            for ds in (dataset, current_dataset):
                if not ds:
                    continue
                if ds.num_episodes == 0:
                    logging.warning(f"No episodes saved in '{ds.repo_id}' — skipping push to hub")
                    continue
                try:
                    validate_repo_id(ds.repo_id)
                except HFValidationError:
                    logging.warning(
                        f"'{ds.repo_id}' is not a valid Hugging Face Hub repo id — it is only stored locally "
                        f"at {ds.root}. Use e.g. --current_dataset_suffix=_current to push it too."
                    )
                    continue
                ds.push_to_hub(tags=cfg.dataset.tags, private=cfg.dataset.private)

        log_say("Exiting", cfg.play_sounds)
    return dataset


def main():
    register_third_party_plugins()
    record()


if __name__ == "__main__":
    main()
