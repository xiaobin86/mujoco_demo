"""Replay recorded teleoperation demonstrations in the MuJoCo simulation.

Key bindings (while a replay is running):

- Right arrow : jump to the next episode
- Left arrow  : jump to the previous episode
- ESC         : stop and exit
"""

from __future__ import annotations

import argparse
import contextlib
from pathlib import Path
import time
from typing import Any

import cv2
import mujoco
import mujoco.viewer
import numpy as np

from jaka_zu35_mujoco_rl.envs import make_env
from jaka_zu35_mujoco_rl.teleop_utils import (
    flip_image_if_needed,
    set_env_cube_position,
    sync_env_to_action,
)


def load_demos(path: Path) -> dict[str, np.ndarray]:
    """Load a `.npz` demonstration file written by `teleop_record.py`."""
    data = np.load(path)
    images = data["images_wrist"] if "images_wrist" in data.files else data["images"]
    return {
        "images": images,
        "observations": data["observations"],
        "proprios": data["proprios"],
        "actions": data["actions"],
        "rewards": data["rewards"],
        "terminals": data["terminals"],
        "episode_starts": data["episode_starts"],
        "cube_positions": data["cube_positions"] if "cube_positions" in data else np.array([]),
    }


def split_episodes(actions: np.ndarray, episode_starts: np.ndarray) -> list[np.ndarray]:
    """Split a flat action array into per-episode chunks."""
    if actions.shape[0] == 0:
        return []

    starts = np.asarray(episode_starts, dtype=bool).copy()
    if starts.shape[0] != actions.shape[0]:
        raise ValueError("episode_starts must have the same length as actions")

    # Always treat the first frame as the start of an episode if it is not marked.
    starts[0] = True
    indices = np.nonzero(starts)[0].tolist()
    indices.append(actions.shape[0])

    episodes: list[np.ndarray] = []
    for i in range(len(indices) - 1):
        episodes.append(actions[indices[i] : indices[i + 1]])
    return episodes


def _throttle_loop(loop_start: float, target_fps: float) -> None:
    sleep = 1.0 / target_fps - (time.perf_counter() - loop_start)
    if sleep > 0:
        time.sleep(sleep)


def _display_image(
    image: np.ndarray,
    display_size: tuple[int, int],
    status: str,
    episode_text: str,
    timer_text: str,
) -> None:
    disp = cv2.resize(image, display_size)
    cv2.putText(disp, f"{status} {episode_text} {timer_text}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(
        disp,
        "-> next episode    <- prev episode    ESC stop",
        (10, disp.shape[0] - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
    )
    cv2.imshow("SO101 replay", disp)
    cv2.waitKey(1)


class _DummyListener:
    """No-op listener used when pynput is not installed (e.g. headless tests)."""

    def stop(self) -> None:
        pass


def _start_keyboard_listener(events: dict[str, bool]) -> Any:
    """Start a global keyboard listener for replay navigation.

    Falls back to a dummy listener if pynput is not available.
    """
    try:
        from pynput import keyboard
    except ImportError:
        print("Warning: pynput not installed; keyboard navigation disabled.")
        return _DummyListener()

    def on_press(key: keyboard.Key | keyboard.KeyCode | None) -> None:
        if key == keyboard.Key.right:
            print("Right arrow pressed: next episode.")
            events["next"] = True
        elif key == keyboard.Key.left:
            print("Left arrow pressed: previous episode.")
            events["prev"] = True
        elif key == keyboard.Key.esc:
            print("ESC pressed: stop replay.")
            events["stop"] = True

    listener = keyboard.Listener(on_press=on_press)
    listener.start()
    return listener


def replay_episode(
    env: Any,
    actions: np.ndarray,
    cube_positions: np.ndarray | None,
    *,
    fps: float,
    show_camera: bool,
    camera: str = "wrist_cam",
    flip_wrist_camera: bool = False,
) -> int:
    """Replay one episode's actions inside the simulation.

    Returns the number of frames that were stepped.
    """
    env.reset()
    sync_env_to_action(env, actions[0])
    if cube_positions is not None and len(cube_positions) > 0:
        set_env_cube_position(env, cube_positions[0])

    for step_idx, action in enumerate(actions):
        loop_start = time.perf_counter()
        if step_idx > 0:
            sync_env_to_action(env, action)
        if cube_positions is not None and step_idx < len(cube_positions):
            set_env_cube_position(env, cube_positions[step_idx])

        if show_camera:
            obs_dict = env.get_observation_dict(camera=camera)
            camera_image = flip_image_if_needed(obs_dict["image"], camera, flip_wrist_camera)
            _display_image(
                camera_image,
                display_size=(640, 480),
                status="REPLAY",
                episode_text=f"step={step_idx + 1}/{len(actions)}",
                timer_text=f"fps={fps:.1f}",
            )

        env.step(action)
        _throttle_loop(loop_start, fps)

    return len(actions)


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay teleoperation demonstrations in MuJoCo.")
    parser.add_argument("--input", type=str, default="data/teleop_demos.npz", help="Demonstration file to replay")
    parser.add_argument("--robot", type=str, default="so101", help="Robot to replay on (so101 or panda)")
    parser.add_argument("--fps", type=float, default=20.0, help="Replay frequency (Hz)")
    parser.add_argument("--display-size", type=str, default="640,480", help="OpenCV display window size (width,height)")
    parser.add_argument("--headless", action="store_true", help="Disable the OpenCV camera window and MuJoCo 3D viewer")
    parser.add_argument("--no-viewer", action="store_true", help="Disable the MuJoCo 3D viewer only")
    parser.add_argument(
        "--camera",
        type=str,
        default="wrist_cam",
        help="Camera name for the OpenCV display (default: wrist_cam, also accepts overhead)",
    )
    parser.add_argument(
        "--flip-wrist-camera",
        action="store_true",
        help="Horizontally flip the wrist_cam image to match a mirrored physical mount",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Demo file not found: {input_path}")

    demos = load_demos(input_path)
    episodes = split_episodes(demos["actions"], demos["episode_starts"])
    if not episodes:
        print("No episodes found in the demo file.")
        return

    display_size = tuple(int(x.strip()) for x in args.display_size.split(","))
    if len(display_size) != 2:
        raise ValueError("--display-size must be width,height")

    max_steps = max(len(ep) for ep in episodes)
    env = make_env(
        robot=args.robot,
        render_mode="rgb_array",
        max_episode_steps=max(1000, max_steps + 100),
    )

    show_camera = not args.headless
    show_viewer = not args.headless and not args.no_viewer
    camera = args.camera
    flip_wrist_camera = args.flip_wrist_camera

    viewer_cm = (
        mujoco.viewer.launch_passive(env.model, env.data)
        if show_viewer
        else contextlib.nullcontext()
    )

    if show_camera:
        cv2.namedWindow("SO101 replay", cv2.WINDOW_NORMAL)

    events: dict[str, bool] = {"next": False, "prev": False, "stop": False}
    listener = _start_keyboard_listener(events)

    episode_idx = 0
    episode_boundaries: list[tuple[int, int]] = []
    offset = 0
    for ep in episodes:
        next_offset = offset + len(ep)
        episode_boundaries.append((offset, next_offset))
        offset = next_offset

    try:
        with viewer_cm as viewer:
            while 0 <= episode_idx < len(episodes) and not events["stop"]:
                events["next"] = False
                events["prev"] = False
                print(f"Replaying episode {episode_idx + 1}/{len(episodes)} ({len(episodes[episode_idx])} frames)")

                start_idx, end_idx = episode_boundaries[episode_idx]
                ep_actions = demos["actions"][start_idx:end_idx]
                ep_cubes = (
                    demos["cube_positions"][start_idx:end_idx]
                    if len(demos["cube_positions"]) > 0
                    else None
                )

                env.reset()
                sync_env_to_action(env, ep_actions[0])
                if ep_cubes is not None and len(ep_cubes) > 0:
                    set_env_cube_position(env, ep_cubes[0])

                for step_idx, action in enumerate(ep_actions):
                    if events["next"] or events["prev"] or events["stop"]:
                        break

                    loop_start = time.perf_counter()
                    if step_idx > 0:
                        sync_env_to_action(env, action)
                    if ep_cubes is not None and step_idx < len(ep_cubes):
                        set_env_cube_position(env, ep_cubes[step_idx])

                    if show_camera:
                        obs_dict = env.get_observation_dict(camera=camera)
                        camera_image = flip_image_if_needed(
                            obs_dict["image"], camera, flip_wrist_camera
                        )
                        _display_image(
                            camera_image,
                            display_size,
                            "REPLAY",
                            f"ep={episode_idx + 1}/{len(episodes)}",
                            f"step={step_idx + 1}/{len(ep_actions)}",
                        )

                    env.step(action)
                    if viewer is not None:
                        viewer.sync()
                    _throttle_loop(loop_start, args.fps)

                if events["prev"]:
                    episode_idx = max(0, episode_idx - 1)
                else:
                    episode_idx += 1

    finally:
        if show_camera:
            cv2.destroyAllWindows()
        listener.stop()

    print("Replay finished.")


if __name__ == "__main__":
    main()
