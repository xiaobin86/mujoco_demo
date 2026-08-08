"""Behavior-clone the PPO actor network on ACT demonstrations to warm-start RL."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from torch.utils.data import DataLoader, TensorDataset

from jaka_zu35_mujoco_rl.envs import make_env


def train_bc(
    policy: nn.Module,
    loader: DataLoader[Any],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    policy.train()
    total_loss = 0.0
    for obs, action in loader:
        obs = obs.to(device)
        action = action.to(device)
        dist = policy.get_distribution(obs)
        pred = dist.distribution.mean
        loss = nn.functional.mse_loss(pred, action)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()
        total_loss += float(loss.item()) * obs.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def eval_bc(policy: nn.Module, loader: DataLoader[Any], device: torch.device) -> float:
    policy.eval()
    total_loss = 0.0
    for obs, action in loader:
        obs = obs.to(device)
        action = action.to(device)
        dist = policy.get_distribution(obs)
        pred = dist.distribution.mean
        loss = nn.functional.mse_loss(pred, action)
        total_loss += float(loss.item()) * obs.size(0)
    return total_loss / len(loader.dataset)


def main() -> None:
    parser = argparse.ArgumentParser(description="Pretrain PPO actor with BC on ACT demonstrations")
    parser.add_argument("--demos", type=str, default="data/act_demos.npz", help="Demonstration file")
    parser.add_argument("--output", type=str, default="checkpoints/ppo_so101_bc_init.zip", help="Output PPO checkpoint")
    parser.add_argument("--epochs", type=int, default=100, help="BC epochs")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--val-split", type=float, default=0.1, help="Validation split")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device)

    data = np.load(args.demos)
    observations = data["observations"].astype(np.float32)
    actions = data["actions"].astype(np.float32)

    n_val = int(len(observations) * args.val_split)
    train_obs = torch.from_numpy(observations[n_val:])
    train_act = torch.from_numpy(actions[n_val:])
    val_obs = torch.from_numpy(observations[:n_val])
    val_act = torch.from_numpy(actions[:n_val])

    train_loader = DataLoader(TensorDataset(train_obs, train_act), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_obs, val_act), batch_size=args.batch_size, shuffle=False)

    env = make_env(robot="so101")
    model = PPO("MlpPolicy", env, verbose=0, device=device)
    policy = model.policy

    actor_params = list(policy.mlp_extractor.policy_net.parameters()) + list(policy.action_net.parameters())
    optimizer = torch.optim.Adam(actor_params, lr=args.lr)

    best_val_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_loss = train_bc(policy, train_loader, optimizer, device)
        val_loss = eval_bc(policy, val_loader, device)
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: train_loss={train_loss:.6f} val_loss={val_loss:.6f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    model.save(args.output)
    print(f"BC pretraining complete. Best val loss: {best_val_loss:.6f}. Saved to {args.output}")
    env.close()


if __name__ == "__main__":
    main()
