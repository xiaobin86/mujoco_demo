"""Teleoperate the simulated SO101 slave arm with a real SO101 master arm and record demonstrations.

Key bindings are inspired by LeRobot's `lerobot-record` script:

- Right arrow : finish the current episode early and save it
- Left arrow  : abort the current episode and rerecord it (discard data)
- ESC         : stop recording and save all completed episodes

Each episode consists of a reset phase followed by a recording phase. The
reset phase gives you time to move the master arm to the desired starting pose
without recording. During the recording phase the master arm pose is sampled
at a fixed rate and saved as (image, proprio, action) frames.
"""

from __future__ import annotations

import argparse
import contextlib
from pathlib import Path
import time

import cv2
import mujoco.viewer
import numpy as np
from lerobot.teleoperators import make_teleoperator_from_config
from lerobot.teleoperators.so101_leader.config_so101_leader import SO101LeaderConfig
from pynput import keyboard

from jaka_zu35_mujoco_rl.envs import make_env
from jaka_zu35_mujoco_rl.teleop_utils import (
    flip_image_if_needed,
    master_to_action,
    sync_env_to_action,
)


def _start_keyboard_listener(events: dict[str, bool]) -> keyboard.Listener:
    """Start a global keyboard listener matching LeRobot's record keys."""

    def on_press(key: keyboard.Key | keyboard.KeyCode | None) -> None:
        if key == keyboard.Key.right:
            print("Right arrow pressed: finish episode early and save it.")
            events["exit_early"] = True
        elif key == keyboard.Key.left:
            print("Left arrow pressed: abort current episode and rerecord it.")
            events["rerecord"] = True
            events["exit_early"] = True
        elif key == keyboard.Key.esc:
            print("ESC pressed: stop recording and save.")
            events["stop"] = True
            events["exit_early"] = True

    listener = keyboard.Listener(on_press=on_press)
    listener.start()
    return listener


def _display_image(
    image: np.ndarray,
    display_size: tuple[int, int],
    status: str,
    color: tuple[int, int, int],
    episode_text: str,
    timer_text: str,
) -> None:
    disp = cv2.resize(image, display_size)
    cv2.putText(disp, f"{status} {episode_text} {timer_text}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.putText(
        disp,
        "-> finish early    <- abort episode    ESC stop & save",
        (10, disp.shape[0] - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
    )
    cv2.imshow("SO101 teleop", disp)
    cv2.waitKey(1)


def _select_view(
    wrist_image: np.ndarray,
    overhead_image: np.ndarray,
    camera: str,
    flip_wrist_camera: bool,
) -> np.ndarray:
    """Build the operator display frame from the two camera views."""
    wrist_image = flip_image_if_needed(wrist_image, "wrist_cam", flip_wrist_camera)
    if camera == "wrist_cam":
        return wrist_image
    if camera == "overhead":
        return overhead_image
    if wrist_image.shape[:2] != overhead_image.shape[:2]:
        overhead_image = cv2.resize(overhead_image, (wrist_image.shape[1], wrist_image.shape[0]))
    combined = np.hstack([wrist_image, overhead_image])
    h, w = wrist_image.shape[:2]
    cv2.putText(combined, "wrist", (w - 110, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    cv2.putText(combined, "overhead", (2 * w - 170, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    return combined


def _throttle_loop(loop_start: float, target_fps: float) -> None:
    sleep = 1.0 / target_fps - (time.perf_counter() - loop_start)
    if sleep > 0:
        time.sleep(sleep)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Teleoperate SO101 slave with real SO101 master arm and record demonstrations."
    )
    parser.add_argument("--port", type=str, default="/dev/ttyACM0", help="Serial port of the master arm")
    parser.add_argument("--id", type=str, default="07252802", help="Robot id for calibration file")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output demo file (default: data/teleop_demos_<start-timestamp>.npz)",
    )
    parser.add_argument("--episodes", type=int, default=10, help="Number of episodes to record")
    parser.add_argument("--image-size", type=int, default=128, help="Recorded image size (square)")
    parser.add_argument("--display-size", type=str, default="1280,480", help="OpenCV display window size (width,height)")
    parser.add_argument("--record-fps", type=float, default=20.0, help="Target recording/control loop frequency (Hz)")
    parser.add_argument("--episode-time-s", type=float, default=60.0, help="Maximum duration of one recorded episode")
    parser.add_argument("--reset-time-s", type=float, default=10.0, help="Pause between episodes to reposition the master arm")
    parser.add_argument("--no-calibrate", action="store_true", help="Skip automatic calibration prompt")
    parser.add_argument("--no-viewer", action="store_true", help="Disable the MuJoCo 3D viewer")
    parser.add_argument(
        "--camera",
        type=str,
        default="both",
        choices=["both", "wrist_cam", "overhead"],
        help="OpenCV display layout: 'both' shows wrist+overhead side by side (default), or show a single camera",
    )
    parser.add_argument(
        "--flip-wrist-camera",
        action="store_true",
        help="Horizontally flip the wrist_cam image to match a mirrored physical mount",
    )
    args = parser.parse_args()

    if args.output is None:
        args.output = str(Path("data") / f"teleop_demos_{time.strftime('%Y%m%d_%H%M%S')}.npz")

    display_size = tuple(int(x.strip()) for x in args.display_size.split(","))
    if len(display_size) != 2:
        raise ValueError("--display-size must be width,height")

    config = SO101LeaderConfig(port=args.port, id=args.id, use_degrees=True)
    teleop = make_teleoperator_from_config(config)
    teleop.connect(calibrate=not args.no_calibrate)
    teleop.bus.disable_torque()

    max_episode_steps = int(args.episode_time_s * args.record_fps * 2)
    env = make_env(
        robot="so101",
        render_mode="rgb_array",
        max_episode_steps=max_episode_steps,
    )
    # Real-time sync: each control tick at record_fps must advance the
    # simulation by 1/record_fps seconds of sim time. Without frame_skip the
    # single 5ms mj_step per tick runs the slave arm in slow motion (~0.1x),
    # which is perceived as teleoperation latency.
    env.frame_skip = max(1, round(1.0 / (args.record_fps * env.model.opt.timestep)))
    env.reset()

    master_obs = teleop.get_action()
    init_action = master_to_action(master_obs, env)
    sync_env_to_action(env, init_action)
    print("Simulation synced to master arm.")
    print("Keys: Right arrow = finish episode, Left arrow = abort episode, ESC = stop & save.")

    viewer_cm = (
        mujoco.viewer.launch_passive(env.model, env.data)
        if not args.no_viewer
        else contextlib.nullcontext()
    )

    images_wrist: list[np.ndarray] = []
    images_overhead: list[np.ndarray] = []
    observations: list[np.ndarray] = []
    proprios: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    rewards: list[float] = []
    terminals: list[bool] = []
    ep_starts: list[bool] = []
    cube_positions: list[np.ndarray] = []

    cv2.namedWindow("SO101 teleop", cv2.WINDOW_NORMAL)
    events: dict[str, bool] = {"exit_early": False, "rerecord": False, "stop": False}
    listener = _start_keyboard_listener(events)

    try:
        with viewer_cm as viewer:
            episode_idx = 0
            while episode_idx < args.episodes and not events["stop"]:
                # ----------------------------------------------------------------
                # Reset phase: give the user time to move the master arm to the
                # desired starting pose. Nothing is recorded.
                # ----------------------------------------------------------------
                events["exit_early"] = False
                events["rerecord"] = False

                env.reset()
                sync_env_to_action(
                    env,
                    master_to_action(
                        teleop.get_action(),
                        env,
                    ),
                )

                reset_deadline = time.perf_counter() + args.reset_time_s
                print(
                    f"Episode {episode_idx + 1}/{args.episodes}: RESET for {args.reset_time_s}s. "
                    "Move the master arm to the start pose."
                )
                while time.perf_counter() < reset_deadline and not events["exit_early"]:
                    loop_start = time.perf_counter()
                    master_obs = teleop.get_action()
                    action = master_to_action(master_obs, env)
                    wrist_obs = env.get_observation_dict(camera="wrist_cam")
                    overhead_obs = env.get_observation_dict(camera="overhead")
                    camera_image = _select_view(
                        wrist_obs["image"], overhead_obs["image"], args.camera, args.flip_wrist_camera
                    )
                    remaining = reset_deadline - time.perf_counter()
                    _display_image(
                        camera_image,
                        display_size,
                        "RESET",
                        (0, 255, 255),
                        f"ep={episode_idx + 1}/{args.episodes}",
                        f"t={remaining:.1f}s",
                    )
                    # Drive the slave arm through the physics actuators only.
                    # Do NOT teleport qpos here: kinematic snaps bypass contact
                    # dynamics, letting the gripper pass through the table and
                    # preventing stable grasps.
                    env.step(action)
                    if viewer is not None:
                        viewer.sync()
                    _throttle_loop(loop_start, args.record_fps)

                if events["stop"]:
                    break

                # ----------------------------------------------------------------
                # Recording phase: sample master pose and save frames at the
                # target rate until the episode time expires or the user presses
                # the right arrow to finish early.
                # ----------------------------------------------------------------
                events["exit_early"] = False
                events["rerecord"] = False

                episode_start_index = len(images_wrist)
                record_deadline = time.perf_counter() + args.episode_time_s
                print(
                    f"Episode {episode_idx + 1}/{args.episodes}: RECORDING for {args.episode_time_s}s. "
                    "Press right arrow to finish early, left arrow to abort."
                )
                step_idx = 0
                while time.perf_counter() < record_deadline and not events["exit_early"]:
                    loop_start = time.perf_counter()
                    master_obs = teleop.get_action()
                    action = master_to_action(master_obs, env)
                    wrist_obs = env.get_observation_dict(camera="wrist_cam")
                    overhead_obs = env.get_observation_dict(camera="overhead")
                    wrist_image = flip_image_if_needed(
                        wrist_obs["image"], "wrist_cam", args.flip_wrist_camera
                    )
                    overhead_image = overhead_obs["image"]
                    camera_image = _select_view(
                        wrist_obs["image"], overhead_obs["image"], args.camera, args.flip_wrist_camera
                    )
                    remaining = record_deadline - time.perf_counter()
                    _display_image(
                        camera_image,
                        display_size,
                        "REC",
                        (0, 255, 0),
                        f"ep={episode_idx + 1}/{args.episodes}",
                        f"t={remaining:.1f}s",
                    )

                    # Physics-only driving: see the note in the reset loop.
                    step_obs, reward, terminated, truncated, _ = env.step(action)
                    images_wrist.append(cv2.resize(wrist_image, (args.image_size, args.image_size)))
                    images_overhead.append(cv2.resize(overhead_image, (args.image_size, args.image_size)))
                    observations.append(step_obs)
                    proprios.append(wrist_obs["proprio"])
                    actions.append(action)
                    rewards.append(float(reward))
                    terminals.append(False)
                    ep_starts.append(step_idx == 0)
                    cube_positions.append(env._get_cube_position())
                    step_idx += 1

                    if viewer is not None:
                        viewer.sync()
                    _throttle_loop(loop_start, args.record_fps)

                    if terminated or truncated:
                        break

                # ----------------------------------------------------------------
                # Finalize the episode.
                # ----------------------------------------------------------------
                if events["rerecord"]:
                    images_wrist = images_wrist[:episode_start_index]
                    images_overhead = images_overhead[:episode_start_index]
                    observations = observations[:episode_start_index]
                    proprios = proprios[:episode_start_index]
                    actions = actions[:episode_start_index]
                    rewards = rewards[:episode_start_index]
                    terminals = terminals[:episode_start_index]
                    ep_starts = ep_starts[:episode_start_index]
                    cube_positions = cube_positions[:episode_start_index]
                    print(f"Episode {episode_idx + 1} aborted; data discarded. Retaking...")
                    continue

                if len(images_wrist) > episode_start_index:
                    if terminals:
                        terminals[-1] = True
                    episode_idx += 1
                    print(f"Episode {episode_idx} saved ({step_idx} frames).")
                else:
                    print(f"Episode {episode_idx + 1} had no data; skipping.")
                    episode_idx += 1

    finally:
        cv2.destroyAllWindows()
        try:
            teleop.disconnect()
        except (OSError, RuntimeError, AttributeError) as e:
            print(f"Warning: failed to disconnect teleop cleanly: {e}")
        listener.stop()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        images_wrist=np.array(images_wrist, dtype=np.uint8),
        images_overhead=np.array(images_overhead, dtype=np.uint8),
        observations=np.array(observations, dtype=np.float64),
        proprios=np.array(proprios, dtype=np.float64),
        actions=np.array(actions, dtype=np.float64),
        rewards=np.array(rewards, dtype=np.float64),
        terminals=np.array(terminals, dtype=bool),
        episode_starts=np.array(ep_starts, dtype=bool),
        cube_positions=np.array(cube_positions, dtype=np.float64),
    )
    print(f"Saved {len(images_wrist)} frames from {episode_idx} episodes to {output_path}")


if __name__ == "__main__":
    main()
