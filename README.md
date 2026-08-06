# Franka Panda MuJoCo Pick-Place RL Environment

A minimal, runnable **MuJoCo + Gymnasium** reinforcement-learning environment with a Franka Emika Panda 7-axis robotic arm and Robotiq 2F85 gripper. The task is **pick-and-place**: grasp a small red cube from a random table position and place it in a fixed target tray.

## Features

- Franka Emika Panda 7-axis arm + Robotiq 2F85 gripper MuJoCo model
- 5 cm red cube on a table with a fixed blue target tray
- Simulated overhead RGB camera + color-based cube detector (YOLO placeholder)
- Gymnasium-compatible `Env` interface
- 32-dimensional observation + 8-dimensional normalized joint-position + gripper control
- Motor actuators with internal PD position tracking (physical torque control preserved)
- Random-agent, API-check, rendering, and interactive 3D viewer examples
- PPO training with live MuJoCo viewer and TensorBoard logging
- Backward-compatible `PandaReachEnv` and `JakaReachEnv` aliases for existing scripts
- Lightweight pytest suite

## Installation

```bash
cd /mnt/d/work/jaka_zu35_mujoco_rl
pip install -e ".[dev]"
```

For RL training, also install the `rl` extras:

```bash
pip install -e ".[rl]"
```

## Quick Start

### Check the environment against the Gymnasium API

```bash
python examples/check_env.py
```

Expected output: `check_env passed`

### Run a random agent

```bash
python examples/random_agent.py
```

Expected: three episodes complete without errors and print step counts, rewards, and final cube-to-tray distances.

### Render a scene to an image

```bash
python examples/render_scene.py --output /tmp/panda_pick_scene.png
```

Expected: a non-empty PNG file is created at `/tmp/panda_pick_scene.png`.

### Watch it run in the interactive 3D viewer

```bash
python examples/viewer_demo.py
```

This opens a MuJoCo 3D window. You will see the Panda arm, the red cube, the blue tray, and the arm attempting to pick and place the cube. The terminal prints the cube-to-tray distance after each episode.

Viewer controls:
- Left drag: rotate camera
- Right drag: pan camera
- Scroll: zoom
- Close the window: stop the demo

> **Note:** The 3D viewer needs a display (X11 / Wayland / Windows / macOS). It will not open in a headless server or WSL without an X server. On WSL, install an X server such as VcXsrv or WSLg; on a remote server, use X11 forwarding or run locally.

### Train PPO with live 3D visualization

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

Open `http://localhost:6006` to see reward curves, episode lengths, and losses. Key metrics:
- `rollout/ep_rew_mean`: average episode reward, should increase toward 0
- `rollout/ep_len_mean`: average episode length, should decrease as the policy succeeds faster
- `train/explained_variance`: should approach 1.0

### Evaluate the trained model

```bash
python examples/evaluate_ppo.py --model checkpoints/ppo_panda_final.zip --episodes 5
```

A 3D window opens and the trained policy runs for 5 episodes. The terminal prints the final cube-to-tray distance and success flag for each episode.

## Tests

Run the lightweight test suite:

```bash
pytest tests/test_env.py -v
```

Expected: all 4 tests pass (`test_model_loads`, `test_env_reset_and_step`, `test_gymnasium_api`, `test_backward_reach_alias_still_works`).

## Environment Interface

```python
from jaka_zu35_mujoco_rl import PandaPickEnv

env = PandaPickEnv()
obs, info = env.reset()
action = env.action_space.sample()
obs, reward, terminated, truncated, info = env.step(action)
```

`PandaReachEnv` and `JakaReachEnv` are kept as aliases for `PandaPickEnv` so existing training scripts continue to work without changes.

### Observation (32-dim)

| Slice | Description |
|-------|-------------|
| 0:7   | Arm joint positions |
| 7:14  | Arm joint velocities |
| 14:17 | End-effector position |
| 17    | Gripper opening width |
| 18:21 | Estimated cube 3D position (from simulated camera + color detector) |
| 21:24 | Target tray position |
| 24:27 | Cube relative to end effector |
| 27:30 | Cube relative to target tray |
| 30    | Detector confidence |
| 31    | Normalized elapsed steps |

### Action (8-dim)

Normalized joint-position targets in `[-1, 1]` for the 7 arm joints, plus one gripper opening command. The arm uses motor (torque) actuators; an internal PD controller computes the torques required to track the targets. The gripper is controlled by the 8th action dimension.

### Reward and Termination

- Dense reward: negative distances for end-effector-to-cube and cube-to-tray, plus small action penalty.
- Sparse reward: +50 when the cube is placed inside the target tray and remains stable for 50 consecutive steps.
- Penalty: -20 if the cube falls off the table.
- `terminated`: cube successfully placed in the tray.
- `truncated`: episode reaches `max_episode_steps` (default 1000).

## Vision Pipeline

The simulated overhead camera renders the scene at 640×480. The `CubeDetector` class thresholds the red cube in HSV space, extracts the 2D bounding box, and back-projects the centre to a 3D world position using the known camera height and table height. This is a placeholder for a real YOLO or instance-segmentation detector; the interface is designed so it can be swapped out later.

## Extension Roadmap

- Replace color detector with a real YOLO model trained on synthetic MuJoCo renders
- Multi-cube scenes and mixed object classes
- Arm-mounted or tower-mounted camera matching the `pallet_vision_lidar` setup
- ROS2 integration and connection to the real perception stack
- Real-world deployment on the Franka Panda or JAKA Zu35 arm

## License

MIT
