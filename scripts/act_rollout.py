"""Rollout a trained ACT policy in simulation to generate demonstrations for PPO BC pretraining."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from jaka_zu35_mujoco_rl.act import ACTPolicy
from jaka_zu35_mujoco_rl.envs import make_env


class TemporalEnsembler:
    """Exponential temporal ensembling from the original ACT paper.

    Unbatched port of LeRobot's ACTTemporalEnsembler: weights w_i = exp(-coeff * i)
    give older predictions higher weight. Every step a full chunk is predicted and
    fused online with previous predictions; the first fused action is executed.
    """

    def __init__(self, coeff: float, chunk_size: int) -> None:
        self.chunk_size = chunk_size
        self.weights = torch.exp(-coeff * torch.arange(chunk_size))
        self.weights_cumsum = torch.cumsum(self.weights, dim=0)
        self.reset()

    def reset(self) -> None:
        self.ensembled_actions = None
        self.ensembled_actions_count = None

    def update(self, actions: torch.Tensor) -> torch.Tensor:
        if self.ensembled_actions is None:
            self.ensembled_actions = actions.clone()
            self.ensembled_actions_count = torch.ones(self.chunk_size, dtype=torch.long)
        else:
            prev_cum = self.weights_cumsum[self.ensembled_actions_count - 1].unsqueeze(-1)
            new_w = self.weights[self.ensembled_actions_count].unsqueeze(-1)
            new_cum = self.weights_cumsum[self.ensembled_actions_count].unsqueeze(-1)
            self.ensembled_actions = (self.ensembled_actions * prev_cum + actions[:-1] * new_w) / new_cum
            self.ensembled_actions_count = torch.clamp(self.ensembled_actions_count + 1, max=self.chunk_size)
            self.ensembled_actions = torch.cat([self.ensembled_actions, actions[-1:]])
            self.ensembled_actions_count = torch.cat(
                [self.ensembled_actions_count, torch.ones(1, dtype=torch.long)]
            )
        action = self.ensembled_actions[0]
        self.ensembled_actions = self.ensembled_actions[1:]
        self.ensembled_actions_count = self.ensembled_actions_count[1:]
        return action


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description="Rollout ACT policy in SO101PickEnv")
    parser.add_argument("--model", type=str, default="checkpoints/act_so101.pt", help="Trained ACT checkpoint")
    parser.add_argument("--output", type=str, default="data/act_demos.npz", help="Output demo file")
    parser.add_argument("--episodes", type=int, default=50, help="Number of rollout episodes")
    parser.add_argument("--max-steps", type=int, default=300, help="Max steps per episode")
    parser.add_argument("--chunk-size", type=int, default=8, help="Action chunk size used by ACT")
    parser.add_argument("--hidden-dim", type=int, default=128, help="Transformer hidden dim used by ACT")
    parser.add_argument(
        "--single-camera",
        action="store_true",
        help="Use only the wrist camera for rollout (default uses wrist + overhead, matching the dual-camera training data)",
    )
    parser.add_argument(
        "--temporal-ensemble-coeff",
        type=float,
        default=0.01,
        help="Temporal ensembling coefficient: predict every step and exponentially fuse overlapping chunks (original ACT default); <=0 disables and executes chunks open-loop",
    )
    parser.add_argument("--device", type=str, default="cuda", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--image-size", type=int, default=128, help="Input image size")
    args = parser.parse_args()

    dual_camera = not args.single_camera

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device)

    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    # Auto-detect hyperparameters from the checkpoint so the rollout script does
    # not need to be invoked with the exact same training flags.
    checkpoint = torch.load(model_path, map_location="cpu")
    if "action_queries.weight" not in checkpoint:
        raise RuntimeError("Checkpoint is missing action_queries.weight; cannot infer hyperparameters")
    detected_chunk_size = int(checkpoint["action_queries.weight"].shape[0])
    detected_hidden_dim = int(checkpoint["action_queries.weight"].shape[1])
    n_encoder_layers = len({k.split(".")[2] for k in checkpoint if k.startswith("encoder.layers.")})
    n_decoder_layers = len({k.split(".")[2] for k in checkpoint if k.startswith("decoder.layers.")})

    if detected_chunk_size != args.chunk_size:
        print(
            f"Warning: --chunk-size {args.chunk_size} does not match checkpoint chunk_size "
            f"{detected_chunk_size}; using checkpoint value."
        )
        args.chunk_size = detected_chunk_size
    if detected_hidden_dim != args.hidden_dim:
        print(
            f"Warning: --hidden-dim {args.hidden_dim} does not match checkpoint hidden_dim "
            f"{detected_hidden_dim}; using checkpoint value."
        )
        args.hidden_dim = detected_hidden_dim

    env = make_env(robot="so101", render_mode="rgb_array")
    obs, _ = env.reset()
    obs_dict = env.get_observation_dict()
    proprio_dim = obs_dict["proprio"].shape[0]
    action_dim = env.action_space.shape[0]

    policy = ACTPolicy(
        proprio_dim=proprio_dim,
        action_dim=action_dim,
        chunk_size=args.chunk_size,
        hidden_dim=args.hidden_dim,
        n_encoder_layers=max(1, n_encoder_layers),
        n_decoder_layers=max(1, n_decoder_layers),
    ).to(device)
    policy.load_state_dict(checkpoint)
    policy.eval()
    print(
        f"Loaded ACT checkpoint from {model_path} "
        f"(chunk_size={args.chunk_size}, hidden_dim={args.hidden_dim}, "
        f"n_encoder_layers={n_encoder_layers}, n_decoder_layers={n_decoder_layers}, device={device})"
    )

    ensembler = (
        TemporalEnsembler(args.temporal_ensemble_coeff, args.chunk_size)
        if args.temporal_ensemble_coeff > 0
        else None
    )
    if ensembler is not None:
        print(f"Temporal ensembling on (coeff={args.temporal_ensemble_coeff}): predicting every step")

    def get_obs() -> tuple[dict, np.ndarray | None]:
        wrist_obs = env.get_observation_dict(camera="wrist_cam")
        overhead_img = env.get_observation_dict(camera="overhead")["image"] if dual_camera else None
        return wrist_obs, overhead_img

    def to_img_tensor(image: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(cv2.resize(image, (args.image_size, args.image_size))).unsqueeze(0).to(device)

    def predict_chunk(wrist_obs: dict, overhead_img: np.ndarray | None) -> np.ndarray:
        imgs = [to_img_tensor(wrist_obs["image"])]
        if overhead_img is not None:
            imgs.append(to_img_tensor(overhead_img))
        prop = torch.from_numpy(wrist_obs["proprio"]).float().unsqueeze(0).to(device)
        return policy(imgs, prop).squeeze(0).cpu().numpy()

    images: list[np.ndarray] = []
    observations: list[np.ndarray] = []
    proprios: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    rewards: list[float] = []
    terminals: list[bool] = []
    ep_starts: list[bool] = []

    total_success = 0
    total_fail = 0
    total_truncated = 0
    ep_returns: list[float] = []
    ep_lengths: list[int] = []

    for ep in range(1, args.episodes + 1):
        env.reset()
        wrist_obs, overhead_img = get_obs()
        if ensembler is not None:
            ensembler.reset()
        action_buffer: list[np.ndarray] = []
        terminated = False
        truncated = False
        step = 0
        ep_reward = 0.0
        ep_start_idx = len(actions)
        while step < args.max_steps and not (terminated or truncated):
            if ensembler is not None:
                action = ensembler.update(torch.from_numpy(predict_chunk(wrist_obs, overhead_img))).numpy()
            else:
                if not action_buffer:
                    chunk = predict_chunk(wrist_obs, overhead_img)
                    action_buffer = [chunk[t] for t in range(args.chunk_size)]
                action = action_buffer.pop(0)

            images.append(cv2.resize(wrist_obs["image"], (args.image_size, args.image_size)))
            observations.append(np.zeros(env.observation_space.shape, dtype=np.float64))
            proprios.append(wrist_obs["proprio"])
            actions.append(action)
            ep_starts.append(step == 0)

            step_obs, reward, terminated, truncated, _ = env.step(action)
            observations[-1] = step_obs
            rewards.append(float(reward))
            ep_reward += float(reward)
            terminals.append(False)
            step += 1
            if not (terminated or truncated):
                wrist_obs, overhead_img = get_obs()

        if terminals:
            terminals[-1] = True

        ep_len = len(actions) - ep_start_idx
        ep_returns.append(ep_reward)
        ep_lengths.append(ep_len)
        final_reward = rewards[-1] if rewards else 0.0
        if terminated:
            if final_reward > 0:
                total_success += 1
                status = "success"
            else:
                total_fail += 1
                status = "fail"
        else:
            total_truncated += 1
            status = "truncated"
        print(
            f"Episode {ep}/{args.episodes}: {ep_len} steps, return={ep_reward:.3f}, "
            f"final_reward={final_reward:.3f}, status={status}"
        )

    n_eps = len(ep_returns)
    success_rate = total_success / n_eps * 100 if n_eps else 0.0
    avg_return = sum(ep_returns) / max(n_eps, 1)
    avg_length = sum(ep_lengths) / max(n_eps, 1)
    print(
        f"Rollout summary: success={total_success} ({success_rate:.1f}%), "
        f"fail={total_fail}, truncated={total_truncated}, "
        f"avg_return={avg_return:.3f}, avg_length={avg_length:.1f}"
    )

    env.close()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        images=np.array(images, dtype=np.uint8),
        observations=np.array(observations, dtype=np.float64),
        proprios=np.array(proprios, dtype=np.float64),
        actions=np.array(actions, dtype=np.float64),
        rewards=np.array(rewards, dtype=np.float64),
        terminals=np.array(terminals, dtype=bool),
        episode_starts=np.array(ep_starts, dtype=bool),
    )
    print(f"Saved {len(images)} frames to {output_path}")


if __name__ == "__main__":
    main()
