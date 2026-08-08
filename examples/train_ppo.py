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
import contextlib
import math
import time
from functools import partial
from pathlib import Path
from typing import Any

import mujoco.viewer
import numpy as np
from gymnasium.wrappers import RecordEpisodeStatistics
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from jaka_zu35_mujoco_rl.callbacks import RewardLoggerCallback
from jaka_zu35_mujoco_rl.envs import make_env
from jaka_zu35_mujoco_rl.envs.panda_pick_env import load_reward_config


class ViewerSyncCallback(BaseCallback):
    """Syncs a live viewer during training and saves checkpoints.

    Two viewer modes:
    - ``viewer``: MuJoCo passive viewer handle (single-env DummyVecEnv).
    - ``video_vec_env``: vectorized env whose envs are rendered as a tiled grid
      in an OpenCV window (multi-env SubprocVecEnv, where the MuJoCo viewer
      cannot attach).
    """

    def __init__(
        self,
        robot: str,
        viewer: Any = None,
        video_vec_env: Any = None,
        sync_every: int = 1,
        checkpoint_every: int = 50_000,
    ) -> None:
        super().__init__(verbose=0)
        self._robot = robot
        self._viewer = viewer
        self._video_vec_env = video_vec_env
        self._sync_every = max(1, int(sync_every))
        self._checkpoint_every = checkpoint_every
        self._checkpoint_dir = Path("checkpoints")
        self._checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._next_checkpoint = checkpoint_every
        self._grid_shape: tuple[int, int] | None = None
        self._video_window = f"train_ppo {robot} (all envs, close or press Esc to stop)"

    def _on_training_start(self) -> None:
        self._next_checkpoint = (
            self.num_timesteps // self._checkpoint_every + 1
        ) * self._checkpoint_every

    def _on_step(self) -> bool:
        if self._viewer is not None:
            if not self._viewer.is_running():
                print("\nViewer closed, stopping training.")
                return False
            if self.n_calls % self._sync_every == 0:
                self._viewer.sync()
        elif self._video_vec_env is not None and self.n_calls % self._sync_every == 0:
            if not self._show_video_frame():
                print("\nViewer closed, stopping training.")
                return False

        if self.num_timesteps >= self._next_checkpoint:
            path = self._checkpoint_dir / f"ppo_{self._robot}_{self.num_timesteps}.zip"
            self.model.save(str(path))
            print(f"Saved checkpoint: {path}")
            self._next_checkpoint += self._checkpoint_every

        return True

    def _show_video_frame(self) -> bool:
        import cv2

        frames = self._video_vec_env.env_method("render")
        if self._grid_shape is None:
            n = len(frames)
            cols = math.ceil(math.sqrt(n))
            while n % cols != 0 and cols < n:
                cols += 1
            self._grid_shape = (n // cols, cols)
        rows, cols = self._grid_shape
        h, w = frames[0].shape[:2]
        scale = min(1.0, 1600.0 / (cols * w))
        tile_size = (int(w * scale), int(h * scale))
        tiles = []
        for i, frame in enumerate(frames):
            tile = np.ascontiguousarray(frame[:, :, ::-1])
            if scale < 1.0:
                tile = cv2.resize(tile, tile_size)
            cv2.putText(tile, f"env {i}", (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            tiles.append(tile)
        grid = np.vstack(
            [np.hstack(tiles[r * cols : (r + 1) * cols]) for r in range(rows)]
        )
        cv2.imshow(self._video_window, grid)
        if cv2.waitKey(1) == 27:
            return False
        return cv2.getWindowProperty(self._video_window, cv2.WND_PROP_VISIBLE) >= 1

    def _on_training_end(self) -> None:
        if self._video_vec_env is not None:
            import cv2

            cv2.destroyAllWindows()
        final_path = self._checkpoint_dir / f"ppo_{self._robot}_final.zip"
        self.model.save(str(final_path))
        print(f"Saved final model: {final_path}")


def make_vec_env(
    robot: str,
    reward_config: dict[str, Any] | str | Path | None = None,
    fast_observation: bool = False,
    render_mode: str | None = None,
):
    env = make_env(
        robot=robot,
        reward_config=reward_config,
        fast_observation=fast_observation,
        render_mode=render_mode,
    )
    env = RecordEpisodeStatistics(env)
    return env


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PPO on PandaPickEnv or SO101PickEnv with a live MuJoCo viewer")
    parser.add_argument("--robot", type=str, default="panda", choices=["panda", "so101"], help="Robot to train")
    parser.add_argument("--total-timesteps", type=int, default=2_000_000, help="Total training steps")
    parser.add_argument(
        "--sync-every",
        type=int,
        default=1,
        help="Sync viewer every N steps (1 = smooth motion; larger values reduce sync overhead but make the viewer jerky)",
    )
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
        help="Path to a YAML reward configuration file (default: config/reward_<robot>_pick*.yaml)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to a saved PPO checkpoint to resume training from (e.g. checkpoints/ppo_<robot>_final.zip)",
    )
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="PPO learning rate")
    parser.add_argument("--n-steps", type=int, default=4096, help="Steps per environment before update")
    parser.add_argument("--batch-size", type=int, default=256, help="Minibatch size for PPO updates")
    parser.add_argument("--n-epochs", type=int, default=10, help="Epochs per PPO update")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--gae-lambda", type=float, default=0.95, help="GAE lambda")
    parser.add_argument("--clip-range", type=float, default=0.1, help="PPO clip range")
    parser.add_argument("--ent-coef", type=float, default=0.01, help="Entropy coefficient")
    parser.add_argument("--vf-coef", type=float, default=0.5, help="Value function loss coefficient")
    parser.add_argument(
        "--fast-obs",
        action="store_true",
        help="Skip offscreen rendering in cube detection and use ground-truth cube position with noise (much faster training)",
    )
    parser.add_argument(
        "--n-envs",
        type=int,
        default=1,
        help="Parallel environment instances for rollout collection (>1 uses SubprocVecEnv, one process per env)",
    )
    args = parser.parse_args()

    if args.n_envs < 1:
        parser.error("--n-envs must be >= 1")
    rollout_samples = args.n_steps * args.n_envs
    if rollout_samples % args.batch_size != 0:
        parser.error(
            f"batch_size ({args.batch_size}) must divide n_steps * n_envs ({rollout_samples})"
        )

    if args.resume and not Path(args.resume).exists():
        raise FileNotFoundError(f"Resume checkpoint not found: {args.resume}")

    if args.reward_config:
        reward_config = Path(args.reward_config)
    elif args.robot == "panda":
        reward_config = Path("config/reward_panda_pick_v2.yaml")
    else:
        reward_config = Path("config/reward_so101_pick.yaml")

    video_viewer = not args.no_viewer and args.n_envs > 1
    env_factory = partial(
        make_vec_env,
        robot=args.robot,
        reward_config=reward_config,
        fast_observation=args.fast_obs,
        render_mode="rgb_array" if video_viewer else None,
    )
    if args.n_envs == 1:
        env = DummyVecEnv([env_factory])
    else:
        # forkserver avoids fork-after-OpenGL/CUDA pitfalls; env_factory is a
        # module-level partial so it pickles into the worker processes.
        env = SubprocVecEnv([env_factory] * args.n_envs, start_method="forkserver")
    env.seed(args.seed)

    common_kwargs = {
        "verbose": 1,
        "tensorboard_log": "logs/",
        "device": args.device,
        "learning_rate": args.learning_rate,
        "n_steps": args.n_steps,
        "batch_size": args.batch_size,
        "n_epochs": args.n_epochs,
        "gamma": args.gamma,
        "gae_lambda": args.gae_lambda,
        "clip_range": args.clip_range,
        "ent_coef": args.ent_coef,
        "vf_coef": args.vf_coef,
        "seed": args.seed,
    }

    if args.resume:
        print(f"Resuming training from {args.resume}")
        # Command-line hyperparameters override the values stored in the checkpoint.
        # PPO.load applies these kwargs after loading the saved data and then rebuilds
        # the rollout buffer in _setup_model(), so n_steps/batch_size take effect.
        model = PPO.load(
            args.resume,
            env=env,
            verbose=1,
            tensorboard_log="logs/",
            device=args.device,
            learning_rate=args.learning_rate,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            clip_range=args.clip_range,
            ent_coef=args.ent_coef,
            vf_coef=args.vf_coef,
            seed=args.seed,
        )
        reset_num_timesteps = False
    else:
        model = PPO("MlpPolicy", env, **common_kwargs)
        reset_num_timesteps = True

    print(
        f"Effective hyperparameters: n_steps={model.n_steps} batch_size={model.batch_size} "
        f"n_epochs={model.n_epochs} learning_rate={model.learning_rate} "
        f"ent_coef={model.ent_coef} gamma={model.gamma} gae_lambda={model.gae_lambda}"
    )

    reward_cfg = load_reward_config(reward_config)
    success_rate_window = reward_cfg["logging"]["success_rate_window"]
    reward_logger = RewardLoggerCallback(
        success_rate_window=success_rate_window,
    )

    if args.no_viewer:
        print("Training without viewer (headless mode).")
        viewer_cm: Any = contextlib.nullcontext()
    elif args.n_envs == 1:
        print("Opening MuJoCo viewer... Close the window to stop training.")
        viewer_cm = mujoco.viewer.launch_passive(
            env.envs[0].unwrapped.model, env.envs[0].unwrapped.data
        )
    else:
        print("Opening OpenCV video viewer (all envs tiled)... Close the window or press Esc to stop training.")
        viewer_cm = contextlib.nullcontext()

    with viewer_cm as viewer:
        viewer_callback = ViewerSyncCallback(
            robot=args.robot,
            viewer=viewer,
            video_vec_env=env if video_viewer else None,
            sync_every=args.sync_every,
            checkpoint_every=args.checkpoint_every,
        )
        callback = CallbackList([viewer_callback, reward_logger])
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=callback,
            progress_bar=True,
            reset_num_timesteps=reset_num_timesteps,
        )

    env.close()


if __name__ == "__main__":
    main()
