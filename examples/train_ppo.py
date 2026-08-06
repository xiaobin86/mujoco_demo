"""Train a PPO agent on PandaReachEnv with a live MuJoCo 3D viewer.

--- How to run ---

Make sure the project is installed in editable mode with the RL extras, then:

    cd /mnt/d/work/jaka_zu35_mujoco_rl
    pip install -e ".[rl]"
    python examples/train_ppo.py

The MuJoCo 3D viewer opens immediately. Closing the window stops training. Training
metrics are written to TensorBoard under ``logs/`` and the final model is saved to
``checkpoints/ppo_panda_final.zip``.

Use TensorBoard to inspect progress:

    tensorboard --logdir logs/

Controls inside the viewer window:
- Left drag  : rotate camera
- Right drag : pan camera
- Scroll     : zoom
- Esc / close window : stop the demo
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import mujoco.viewer
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from jaka_zu35_mujoco_rl import PandaReachEnv


class ViewerSyncCallback(BaseCallback):
    """Sync the MuJoCo viewer during training and stop when the window closes."""

    def __init__(
        self,
        viewer: Any,
        sync_every: int = 100,
        checkpoint_every: int = 50_000,
    ) -> None:
        super().__init__(verbose=0)
        self._viewer = viewer
        self._sync_every = sync_every
        self._checkpoint_every = checkpoint_every
        self._checkpoint_dir = Path("checkpoints")
        self._checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def _on_step(self) -> bool:
        if self._viewer is not None and not self._viewer.is_running():
            print("\nViewer closed, stopping training.")
            return False

        if self._viewer is not None and self.n_calls % self._sync_every == 0:
            self._viewer.sync()

        if self.n_calls % self._checkpoint_every == 0 and self.n_calls > 0:
            path = self._checkpoint_dir / f"ppo_panda_{self.n_calls}.zip"
            self.model.save(str(path))
            print(f"Saved checkpoint: {path}")

        return True

    def _on_training_end(self) -> None:
        final_path = self._checkpoint_dir / "ppo_panda_final.zip"
        self.model.save(str(final_path))
        print(f"Saved final model: {final_path}")


def make_env() -> PandaReachEnv:
    """Factory for the vectorized environment."""
    return PandaReachEnv()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PPO on PandaReachEnv with a live MuJoCo viewer")
    parser.add_argument("--total-timesteps", type=int, default=200_000, help="Total training steps")
    parser.add_argument("--sync-every", type=int, default=100, help="Sync viewer every N steps")
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=50_000,
        help="Save checkpoint every N steps",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument(
        "--no-viewer",
        action="store_true",
        help="Run training without the interactive MuJoCo viewer (useful for headless servers)",
    )
    args = parser.parse_args()

    env = DummyVecEnv([make_env])
    env.seed(args.seed)

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log="logs/",
        device="cpu",
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        seed=args.seed,
    )

    if args.no_viewer:
        print("Training without MuJoCo viewer (headless mode).")
        callback = ViewerSyncCallback(None, sync_every=args.sync_every, checkpoint_every=args.checkpoint_every)
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=callback,
            progress_bar=True,
            reset_num_timesteps=True,
        )
    else:
        print("Opening MuJoCo viewer... Close the window to stop training.")
        with mujoco.viewer.launch_passive(env.envs[0].model, env.envs[0].data) as viewer:
            callback = ViewerSyncCallback(
                viewer,
                sync_every=args.sync_every,
                checkpoint_every=args.checkpoint_every,
            )
            model.learn(
                total_timesteps=args.total_timesteps,
                callback=callback,
                progress_bar=True,
                reset_num_timesteps=True,
            )

    env.close()


if __name__ == "__main__":
    main()
