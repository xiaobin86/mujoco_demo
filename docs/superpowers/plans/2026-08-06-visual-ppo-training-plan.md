# Visual PPO Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a Stable-Baselines3 PPO training loop with live MuJoCo 3D viewer, TensorBoard logging, and model checkpoints, plus an evaluation script to visualize a trained policy.

**Architecture:** Wrap `JakaReachEnv` in a `DummyVecEnv`, create PPO with TensorBoard logging, and run `model.learn()` inside the MuJoCo viewer loop. A custom `ViewerSyncCallback` syncs the 3D window every N steps and stops training when the window closes. The evaluation script loads the saved `.zip` and runs deterministic episodes in the viewer.

**Tech Stack:** Python 3.10+, MuJoCo 3.x, Gymnasium, Stable-Baselines3, TensorBoard, NumPy

**Reference Spec:** `/mnt/d/work/jaka_zu35_mujoco_rl/docs/superpowers/specs/2026-08-06-visual-ppo-training-design.md`

---

## File Structure

```
jaka_zu35_mujoco_rl/
├── pyproject.toml                      # add [rl] optional dependency
├── examples/
│   ├── train_ppo.py                    # PPO training + live viewer
│   ├── evaluate_ppo.py                 # load checkpoint + run in viewer
│   ├── viewer_demo.py                  # existing
│   ├── random_agent.py                 # existing
│   ├── check_env.py                    # existing
│   └── render_scene.py                 # existing
├── checkpoints/                        # created at runtime
├── logs/                               # created at runtime
└── README.md                           # updated
```

---

## Task 1: Add RL optional dependency to pyproject.toml

**Files:**
- Modify: `/mnt/d/work/jaka_zu35_mujoco_rl/pyproject.toml`

- [ ] **Step 1: Add rl optional dependency**

Change the `[project.optional-dependencies]` section from:

```toml
[project.optional-dependencies]
dev = [
    "pytest",
    "imageio",
]
```

To:

```toml
[project.optional-dependencies]
dev = [
    "pytest",
    "imageio",
]
rl = [
    "stable-baselines3",
    "tensorboard",
]
```

- [ ] **Step 2: Reinstall package with RL dependencies**

```bash
cd /mnt/d/work/jaka_zu35_mujoco_rl
pip install -e ".[rl]"
```

Expected: installation succeeds with `stable-baselines3` and `tensorboard` installed. No version conflicts.

- [ ] **Step 3: Verify imports work**

```bash
python -c "from stable_baselines3 import PPO; from stable_baselines3.common.vec_env import DummyVecEnv; from stable_baselines3.common.callbacks import BaseCallback; print('imports OK')"
```

Expected: prints `imports OK`.

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml
git commit -m "deps: add rl optional dependency for stable-baselines3 + tensorboard"
```

---

## Task 2: Implement PPO training script with live viewer

**Files:**
- Create: `/mnt/d/work/jaka_zu35_mujoco_rl/examples/train_ppo.py`

- [ ] **Step 1: Create the training script with the following content**

```python
"""Train a PPO agent on JakaReachEnv with a live MuJoCo 3D viewer.

--- How to run ---

    cd /mnt/d/work/jaka_zu35_mujoco_rl
    pip install -e ".[rl]"
    python examples/train_ppo.py

The viewer window opens immediately. Closing it stops training. Training metrics are
written to TensorBoard under `logs/` and the final model is saved to `checkpoints/`.

Use TensorBoard to inspect progress:

    tensorboard --logdir logs/
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Any

import mujoco.viewer
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from jaka_zu35_mujoco_rl import JakaReachEnv


class ViewerSyncCallback(BaseCallback):
    """Sync the MuJoCo viewer and stop training when the window closes."""

    def __init__(self, viewer: Any, sync_every: int = 100, checkpoint_every: int = 50_000) -> None:
        super().__init__(verbose=0)
        self._viewer = viewer
        self._sync_every = sync_every
        self._checkpoint_every = checkpoint_every
        self._checkpoint_dir = Path("checkpoints")
        self._checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def _on_step(self) -> bool:
        if not self._viewer.is_running():
            print("\nViewer closed, stopping training.")
            return False

        if self.n_calls % self._sync_every == 0:
            self._viewer.sync()

        if self.n_calls % self._checkpoint_every == 0 and self.n_calls > 0:
            path = self._checkpoint_dir / f"ppo_jaka_{self.n_calls}.zip"
            self.model.save(str(path))
            print(f"Saved checkpoint: {path}")

        return True

    def _on_training_end(self) -> None:
        final_path = self._checkpoint_dir / "ppo_jaka_final.zip"
        self.model.save(str(final_path))
        print(f"Saved final model: {final_path}")


def make_env() -> JakaReachEnv:
    return JakaReachEnv()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PPO on JakaReachEnv with live MuJoCo viewer")
    parser.add_argument("--total-timesteps", type=int, default=200_000, help="Total training steps")
    parser.add_argument("--sync-every", type=int, default=100, help="Sync viewer every N steps")
    parser.add_argument("--checkpoint-every", type=int, default=50_000, help="Save checkpoint every N steps")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    args = parser.parse_args()

    env = DummyVecEnv([make_env])
    env.seed(args.seed)

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log="logs/",
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        seed=args.seed,
    )

    print("Opening MuJoCo viewer... Close the window to stop training.")
    with mujoco.viewer.launch_passive(env.envs[0].model, env.envs[0].data) as viewer:
        callback = ViewerSyncCallback(viewer, sync_every=args.sync_every, checkpoint_every=args.checkpoint_every)
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=callback,
            progress_bar=True,
            reset_num_timesteps=True,
        )

    env.close()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke test the script (without full training)**

Run a short training to verify the script starts, viewer code path is valid, and the callback runs without exceptions. Since the viewer cannot open in a headless environment, test with a tiny number of timesteps and expect the viewer to be closed immediately:

```bash
cd /mnt/d/work/jaka_zu35_mujoco_rl
python examples/train_ppo.py --total-timesteps 10 --sync-every 5
```

Expected: The script reaches the `with mujoco.viewer.launch_passive(...)` block, then exits quickly because the viewer is not running in the headless environment. The important check is that it does not fail on import or syntax errors.

- [ ] **Step 3: Verify syntax and imports**

```bash
python -m py_compile examples/train_ppo.py
python -c "import examples.train_ppo"
```

Expected: no errors.

- [ ] **Step 4: Commit**

```bash
git add examples/train_ppo.py
git commit -m "feat: add PPO training script with live MuJoCo viewer and TensorBoard logging"
```

---

## Task 3: Implement PPO evaluation script

**Files:**
- Create: `/mnt/d/work/jaka_zu35_mujoco_rl/examples/evaluate_ppo.py`

- [ ] **Step 1: Create the evaluation script with the following content**

```python
"""Evaluate a trained PPO model on JakaReachEnv with a live MuJoCo 3D viewer.

--- How to run ---

    cd /mnt/d/work/jaka_zu35_mujoco_rl
    python examples/evaluate_ppo.py --model checkpoints/ppo_jaka_final.zip --episodes 5

The viewer window opens and the trained policy runs deterministically for the requested
number of episodes. Terminal prints the final distance and success flag for each episode.
"""

from __future__ import annotations

import argparse
import time
from typing import Any

import mujoco.viewer
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from jaka_zu35_mujoco_rl import JakaReachEnv


def make_env() -> JakaReachEnv:
    return JakaReachEnv()


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained PPO model on JakaReachEnv")
    parser.add_argument("--model", type=str, default="checkpoints/ppo_jaka_final.zip", help="Path to trained model")
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes to evaluate")
    parser.add_argument("--sleep", type=float, default=0.005, help="Seconds between viewer syncs")
    args = parser.parse_args()

    env = DummyVecEnv([make_env])
    model = PPO.load(args.model, env=env)

    print(f"Loading model from {args.model}")
    print("Opening MuJoCo viewer... Close the window to stop evaluation.")

    with mujoco.viewer.launch_passive(env.envs[0].model, env.envs[0].data) as viewer:
        for episode in range(args.episodes):
            if not viewer.is_running():
                break

            obs, info = env.reset()
            terminated = False
            truncated = False
            step = 0

            while viewer.is_running() and not (terminated or truncated):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                step += 1
                viewer.sync()
                time.sleep(args.sleep)

            if viewer.is_running():
                print(
                    f"Episode {episode + 1}: "
                    f"steps={step}, reward={reward:.4f}, "
                    f"success={terminated}, final_distance={info['distance']:.4f}"
                )

    env.close()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify syntax and imports**

```bash
python -m py_compile examples/evaluate_ppo.py
python -c "import examples.evaluate_ppo"
```

Expected: no errors.

- [ ] **Step 3: Commit**

```bash
git add examples/evaluate_ppo.py
git commit -m "feat: add PPO evaluation script with live MuJoCo viewer"
```

---

## Task 4: Update README with training instructions

**Files:**
- Modify: `/mnt/d/work/jaka_zu35_mujoco_rl/README.md`

- [ ] **Step 1: Add training and evaluation section after the existing Quick Start section**

Add the following section after the `### Render a scene to an image` section and before `## Tests`:

```markdown
### Train PPO with live 3D visualization

```bash
pip install -e ".[rl]"
python examples/train_ppo.py --total-timesteps 200000 --sync-every 100
```

A MuJoCo 3D window opens and the arm starts training. Close the window to stop training. Terminal shows a progress bar and SB3 metrics; TensorBoard logs are written to `logs/` and the final model is saved to `checkpoints/ppo_jaka_final.zip`.

### Monitor training with TensorBoard

```bash
tensorboard --logdir logs/
```

Open `http://localhost:6006` to see reward curves, episode lengths, and losses.

### Evaluate the trained model

```bash
python examples/evaluate_ppo.py --model checkpoints/ppo_jaka_final.zip --episodes 5
```

A 3D window opens and the trained policy runs for 5 episodes. Terminal prints the final distance and success flag for each episode.
```

- [ ] **Step 2: Update the Features list**

Add one bullet to the Features list:

```markdown
- PPO training with live 3D viewer, TensorBoard logging, and model checkpoints
```

- [ ] **Step 3: Verify README renders correctly**

```bash
python -c "import pathlib; text = pathlib.Path('README.md').read_text(); assert 'train_ppo.py' in text; assert 'evaluate_ppo.py' in text; assert 'tensorboard' in text; print('README OK')"
```

Expected: prints `README OK`.

- [ ] **Step 4: Commit**

```bash
git add README.md
git commit -m "docs: add PPO training, TensorBoard, and evaluation instructions"
```

---

## Task 5: Final integration smoke test

**Files:**
- None (verification only)

- [ ] **Step 1: Verify all imports**

```bash
cd /mnt/d/work/jaka_zu35_mujoco_rl
python -c "from examples.train_ppo import ViewerSyncCallback, main; from examples.evaluate_ppo import main; print('all imports OK')"
```

Expected: prints `all imports OK`.

- [ ] **Step 2: Verify short training runs without import/syntax errors**

```bash
python examples/train_ppo.py --total-timesteps 10 --sync-every 5
```

Expected: script exits quickly (viewer cannot open in headless environment). No import errors, no syntax errors, no crashes before or after the viewer block. If a checkpoint is saved, it may be incomplete or zero-size due to early termination; this is acceptable for the smoke test.

- [ ] **Step 3: Run existing tests to ensure no regressions**

```bash
pytest tests/test_env.py -v
```

Expected: 3 tests pass.

- [ ] **Step 4: Commit if any fixes were needed**

If no fixes were needed, skip this step. If fixes were needed, commit them with a clear message.

---

## Self-Review

- [ ] **Spec coverage:**
  - `pyproject.toml` rl dependency → Task 1
  - `train_ppo.py` with viewer, callback, TensorBoard, checkpoints → Task 2
  - `evaluate_ppo.py` load model and run in viewer → Task 3
  - README updates → Task 4
  - Integration smoke test → Task 5

- [ ] **Placeholder scan:** No TBD, TODO, "implement later", or vague requirements.

- [ ] **Type consistency:** `ViewerSyncCallback` uses `BaseCallback` API consistently. `model.save` paths are strings. `env.envs[0]` refers to the unwrapped `JakaReachEnv` in both Task 2 and Task 3.
