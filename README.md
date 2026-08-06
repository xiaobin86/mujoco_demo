# Franka Panda MuJoCo RL Environment

A minimal, runnable **MuJoCo + Gymnasium** reinforcement-learning environment with a Franka Emika Panda 7-axis robotic arm and a single `cola_24` box target.

## Features

- Franka Emika Panda 7-axis arm MuJoCo model with Robotiq 2F85 gripper
- Single `cola_24` box scene (0.4 × 0.27 × 0.24 m)
- Gymnasium-compatible `Env` interface
- 20-dimensional observation + 7-dimensional normalized joint-position control
- Motor actuators with internal PD position tracking (physical torque control preserved)
- Random-agent, API-check, rendering, and interactive 3D viewer examples
- Backward-compatible `JakaReachEnv` alias for existing RL training scripts
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
python examples/render_scene.py --output /tmp/panda_reach_scene.png
```

Expected: a non-empty PNG file is created at `/tmp/panda_reach_scene.png`.

### Watch it run in the interactive 3D viewer

```bash
python examples/viewer_demo.py
```

This opens a MuJoCo 3D window. You will see the Panda arm moving and the red box on the ground, and the terminal prints the distance from the gripper pinch point to the box top after each episode.

Viewer controls:
- Left drag: rotate camera
- Right drag: pan camera
- Scroll: zoom
- Close the window: stop the demo

> **Note:** The 3D viewer needs a display (X11 / Wayland / Windows / macOS). It will not open in a headless server or WSL without an X server. On WSL, install an X server such as VcXsrv or WSLg; on a remote server, use X11 forwarding or run locally.

### Train PPO with live 3D visualization

First install the RL extras:

```bash
pip install -e ".[rl]"
```

Then start training:

```bash
python examples/train_ppo.py --total-timesteps 200000 --sync-every 100
```

A MuJoCo 3D window opens and the Panda arm starts training. Close the window to stop training. The terminal shows a progress bar and Stable-Baselines3 metrics; TensorBoard logs are written to `logs/` and the final model is saved to `checkpoints/ppo_panda_final.zip`.

> **Note:** On a headless server or WSL without an X server, the viewer cannot open. Use `--no-viewer` to train without the interactive window:
> ```bash
> python examples/train_ppo.py --no-viewer --total-timesteps 200000
> ```

### Monitor training with TensorBoard

```bash
tensorboard --logdir logs/
```

Open `http://localhost:6006` to see reward curves, episode lengths, and losses.

### Evaluate the trained model

```bash
python examples/evaluate_ppo.py --model checkpoints/ppo_panda_final.zip --episodes 5
```

A 3D window opens and the trained policy runs for 5 episodes. The terminal prints the final distance and success flag for each episode.

## Tests

Run the lightweight test suite:

```bash
pytest tests/test_env.py -v
```

Expected: all 3 tests pass (`test_model_loads`, `test_env_reset_and_step`, `test_gymnasium_api`).

## Environment Interface

```python
from jaka_zu35_mujoco_rl import PandaReachEnv

env = PandaReachEnv()
obs, info = env.reset()
action = env.action_space.sample()
obs, reward, terminated, truncated, info = env.step(action)
```

`JakaReachEnv` is kept as an alias for `PandaReachEnv` so existing training scripts continue to work without changes.

### Observation (20-dim)

| Slice | Description |
|-------|-------------|
| 0:7   | Arm joint positions |
| 7:14  | Arm joint velocities |
| 14:17 | Pinch (end-effector) position |
| 17:20 | Target box top-center position |

### Action (7-dim)

Normalized joint-position targets in `[-1, 1]`, linearly mapped to each arm joint's controller range. The arm uses motor (torque) actuators; an internal PD controller computes the torques required to track the targets. The gripper is held open and is not part of the action space.

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
