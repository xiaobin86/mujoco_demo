# JAKA Zu35 MuJoCo RL Environment

A minimal, runnable **MuJoCo + Gymnasium** reinforcement-learning environment with a simplified JAKA Zu35 6-axis robotic arm and a single `cola_24` box target.

## Features

- Simplified JAKA Zu35 6-axis arm MuJoCo model
- Single `cola_24` box scene (0.4 × 0.27 × 0.24 m)
- Gymnasium-compatible `Env` interface
- 18-dimensional observation + 6-dimensional normalized joint-position control
- Random-agent, API-check, rendering, and interactive 3D viewer examples
- Lightweight pytest suite

## Installation

```bash
cd /mnt/d/work/jaka_zu35_mujoco_rl
pip install -e ".[dev]"
```

## Quick Start

### Check the environment against the Gymnasium API

```bash
python examples/check_env.py
```

Expected output: `check_env passed`

The script uses `gymnasium.utils.env_checker.check_env` because the installed `gymnasium==1.3.0` release does not re-export `check_env` at the top level.

### Run a random agent

```bash
python examples/random_agent.py
```

Expected: three episodes complete without errors and print step counts, rewards, and final distances.

### Render a scene to an image

```bash
python examples/render_scene.py --output /tmp/jaka_reach_scene.png
```

Expected: a non-empty PNG file is created at `/tmp/jaka_reach_scene.png`.

### Watch it run in the interactive 3D viewer

```bash
python examples/viewer_demo.py
```

This opens a MuJoCo 3D window. You will see the arm moving and the red box on the ground, and the terminal prints the distance from the end-effector to the box top after each episode.

Viewer controls:
- Left drag: rotate camera
- Right drag: pan camera
- Scroll: zoom
- Close the window: stop the demo

> **Note:** The 3D viewer needs a display (X11 / Wayland / Windows / macOS). It will not open in a headless server or WSL without an X server. On WSL, install an X server such as VcXsrv or WSLg; on a remote server, use X11 forwarding or run locally.

## Tests

Run the lightweight test suite:

```bash
pytest tests/test_env.py -v
```

Expected: all 3 tests pass (`test_model_loads`, `test_env_reset_and_step`, `test_gymnasium_api`).

## Environment Interface

```python
from jaka_zu35_mujoco_rl import JakaReachEnv

env = JakaReachEnv()
obs, info = env.reset()
action = env.action_space.sample()
obs, reward, terminated, truncated, info = env.step(action)
```

### Observation (18-dim)

| Slice | Description |
|-------|-------------|
| 0:6   | Arm joint positions |
| 6:12  | Arm joint velocities |
| 12:15 | End-effector position |
| 15:18 | Target box top-center position |

### Action (6-dim)

Normalized joint-position targets in `[-1, 1]`, linearly mapped to each joint's controller range.

### Reward and Termination

- Reward: negative Euclidean distance from end-effector to box top, minus a small action penalty.
- `terminated`: distance falls below `success_threshold` (default 0.05 m).
- `truncated`: episode reaches `max_episode_steps` (default 500).

## Extension Roadmap

- Add a two-finger gripper and grasp-success criterion
- Multi-box pallet scenes and mixed-product (`cola_24` / `water_12`) stacks
- Visual observations (RGB / depth) from a camera mounted on the arm or tower
- ROS2 integration and connection to the `pallet_vision_lidar` perception stack
- Train a stable reaching / grasping policy with RL or model-based methods

## License

MIT
