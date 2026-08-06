"""Train a PPO agent on PandaPickEnv with a live MuJoCo 3D viewer.

The task is pick-and-place: a Franka Panda arm with a Robotiq 2F85 gripper must
pick up a small red cube from a random table position and place it in a fixed
target tray.

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
from gymnasium.wrappers import RecordEpisodeStatistics
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.vec_env import DummyVecEnv

from jaka_zu35_mujoco_rl import PandaPickEnv
from jaka_zu35_mujoco_rl.callbacks import RewardLoggerCallback
from jaka_zu35_mujoco_rl.envs.panda_pick_env import load_reward_config


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


def make_env(reward_config: dict[str, Any] | str | Path | None = None) -> PandaPickEnv:
    """Factory for the vectorized environment."""
    env = PandaPickEnv(reward_config=reward_config)
    env = RecordEpisodeStatistics(env)
    return env


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PPO on PandaPickEnv with a live MuJoCo viewer")
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
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device for PPO training (auto picks CUDA if available)",
    )
    parser.add_argument(
        "--reward-config",
        type=str,
        default=None,
        help="Path to a YAML reward configuration file (default: use env defaults)",
    )
    args = parser.parse_args()

    reward_config = Path(args.reward_config) if args.reward_config else None
    env = DummyVecEnv([lambda: make_env(reward_config=reward_config)])
    env.seed(args.seed)

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log="logs/",
        device=args.device,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        seed=args.seed,
    )

    reward_cfg = load_reward_config(reward_config)
    success_rate_window = reward_cfg["logging"]["success_rate_window"]
    reward_logger = RewardLoggerCallback(
        success_rate_window=success_rate_window,
    )

    if args.no_viewer:
        print("Training without MuJoCo viewer (headless mode).")
        viewer_callback = ViewerSyncCallback(
            None,
            sync_every=args.sync_every,
            checkpoint_every=args.checkpoint_every,
        )
        callback = CallbackList([viewer_callback, reward_logger])
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=callback,
            progress_bar=True,
            reset_num_timesteps=True,
        )
    else:
        print("Opening MuJoCo viewer... Close the window to stop training.")
        with mujoco.viewer.launch_passive(env.envs[0].unwrapped.model, env.envs[0].unwrapped.data) as viewer:
            viewer_callback = ViewerSyncCallback(
                viewer,
                sync_every=args.sync_every,
                checkpoint_every=args.checkpoint_every,
            )
            callback = CallbackList([viewer_callback, reward_logger])
            model.learn(
                total_timesteps=args.total_timesteps,
                callback=callback,
                progress_bar=True,
                reset_num_timesteps=True,
            )

    env.close()


if __name__ == "__main__":
    main()
