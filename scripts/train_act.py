"""Train an ACT policy on teleoperation demonstrations."""

from __future__ import annotations

import argparse
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from jaka_zu35_mujoco_rl.act import ACTPolicy

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None


class RandomAugment:
    """Lightweight batched image augmentation for ACT training.

    Operates on (B, H, W, C) uint8 tensors with random crop, brightness and
    contrast jitter. The random crop uses reflect padding to avoid changing
    the image mean, which is important because the downstream ResNet-18
    encoder expects ImageNet normalized inputs.
    """

    def __init__(
        self,
        pad: int = 4,
        brightness: float = 0.2,
        contrast: float = 0.2,
    ) -> None:
        self.pad = pad
        self.brightness = brightness
        self.contrast = contrast

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float() / 255.0
        if self.brightness > 0.0:
            x = self._brightness(x)
        if self.contrast > 0.0:
            x = self._contrast(x)
        if self.pad > 0:
            x = self._random_crop(x)
        return (x * 255.0).clamp_(0.0, 255.0).to(torch.uint8)

    def _brightness(self, x: torch.Tensor) -> torch.Tensor:
        delta = torch.empty(x.size(0), 1, 1, 1).uniform_(-self.brightness, self.brightness)
        return torch.clamp(x + delta, 0.0, 1.0)

    def _contrast(self, x: torch.Tensor) -> torch.Tensor:
        factor = torch.empty(x.size(0), 1, 1, 1).uniform_(1 - self.contrast, 1 + self.contrast)
        mean = x.mean(dim=(1, 2, 3), keepdim=True)
        return torch.clamp((x - mean) * factor + mean, 0.0, 1.0)

    def _random_crop(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, C = x.shape
        x = x.permute(0, 3, 1, 2)
        x = torch.nn.functional.pad(x, (self.pad, self.pad, self.pad, self.pad), mode="reflect")
        top = torch.randint(0, 2 * self.pad + 1, (B,))
        left = torch.randint(0, 2 * self.pad + 1, (B,))
        crops = torch.stack([x[i, :, top[i] : top[i] + H, left[i] : left[i] + W] for i in range(B)])
        return crops.permute(0, 2, 3, 1)


class EarlyStopping:
    """Stop training when validation loss has not improved for a patience window."""

    def __init__(self, patience: int, min_delta: float = 0.0) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")

    def __call__(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


class TeleopDataset:
    """Demo dataset resident on the training device.

    All frames are uploaded once at construction; training batches are pure
    device-side index gathers, so epochs have no DataLoader or per-batch
    CPU->GPU transfer overhead.
    """

    def __init__(
        self, npz_paths: Sequence[str | Path], chunk_size: int, device: torch.device, augment: bool = False
    ) -> None:
        proprios, actions, ep_starts_parts = [], [], []
        image_arrays: dict[str, list[np.ndarray]] = {}
        for path in npz_paths:
            data = np.load(path)
            if proprios and data["proprios"].shape[1] != proprios[0].shape[1]:
                raise ValueError(
                    f"proprio dim mismatch: {path} has {data['proprios'].shape[1]}, "
                    f"expected {proprios[0].shape[1]}"
                )
            if actions and data["actions"].shape[1] != actions[0].shape[1]:
                raise ValueError(
                    f"action dim mismatch: {path} has {data['actions'].shape[1]}, "
                    f"expected {actions[0].shape[1]}"
                )
            keys = [k for k in ("images_wrist", "images_overhead") if k in data.files] or ["images"]
            if not image_arrays:
                image_arrays = {k: [] for k in keys}
            if set(keys) != set(image_arrays):
                raise ValueError(f"camera keys mismatch: {path} has {keys}, expected {list(image_arrays)}")
            for k in keys:
                image_arrays[k].append(data[k])
            proprios.append(data["proprios"])
            actions.append(data["actions"])
            if "episode_starts" in data:
                ep_starts_parts.append(np.asarray(data["episode_starts"], dtype=bool))

        n_frames = sum(len(p) for p in proprios)
        n = n_frames - chunk_size
        if n <= 0:
            raise ValueError(f"Not enough frames ({n_frames}) for chunk_size={chunk_size}")

        # Images stay uint8 to save memory. When augmentation is enabled, they
        # are kept on CPU so transforms can be applied per batch; otherwise they
        # are uploaded to the target device once at construction.
        self.image_tensors = [
            torch.from_numpy(np.concatenate(arrs, axis=0))[:n] for arrs in image_arrays.values()
        ]
        if not augment:
            self.image_tensors = [t.to(device) for t in self.image_tensors]
        prop = torch.from_numpy(np.concatenate(proprios, axis=0)).float().to(device)
        act = torch.from_numpy(np.concatenate(actions, axis=0)).float().to(device)

        self.proprios = prop[:n]
        self.action_chunks = act.unfold(0, chunk_size, 1)[:n].transpose(1, 2).contiguous()
        self.proprio_dim = int(prop.shape[1])
        self.action_dim = int(act.shape[1])
        self.device = device
        self.augment = augment
        self.transform = RandomAugment() if augment else None

        # Restrict sampling so action chunks never cross episode boundaries
        # (same idea as LeRobot's EpisodeAwareSampler / action_is_pad masking).
        if len(ep_starts_parts) == len(npz_paths):
            ep_starts = np.concatenate(ep_starts_parts, axis=0)
            starts = np.flatnonzero(ep_starts)
            ends = np.concatenate([starts[1:], [n_frames]])
            valid = np.zeros(n, dtype=bool)
            for s, e in zip(starts, ends):
                hi = min(e - chunk_size + 1, n)
                if hi > s:
                    valid[s:hi] = True
            self.valid_idx = torch.from_numpy(np.flatnonzero(valid)).long().to(device)
            self.episode_starts = ep_starts
        else:
            print("Warning: episode_starts missing in some demo files; chunks may cross episode boundaries")
            self.valid_idx = torch.arange(n, device=device)
            self.episode_starts = None

    def __len__(self) -> int:
        return len(self.image_tensors[0])

    def gather(self, idx: torch.Tensor) -> tuple[list[torch.Tensor], torch.Tensor, torch.Tensor]:
        images = [img[idx] for img in self.image_tensors]
        if self.transform is not None:
            images = [self.transform(img) for img in images]
            images = [img.to(self.device) for img in images]
        return images, self.proprios[idx], self.action_chunks[idx]


def train_epoch(
    model: nn.Module,
    dataset: TeleopDataset,
    idx: torch.Tensor,
    batch_size: int,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    use_amp: bool,
) -> float:
    model.train()
    total_loss = 0.0
    count = 0
    order = idx[torch.randperm(idx.numel(), device=device)]
    n_batches = order.numel() // batch_size
    for i in range(n_batches):
        b = order[i * batch_size : (i + 1) * batch_size]
        images, proprios, actions = dataset.gather(b)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
            pred = model(images, proprios)
            loss = nn.functional.l1_loss(pred, actions)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += float(loss.item()) * b.numel()
        count += b.numel()
    return total_loss / max(count, 1)


@torch.no_grad()
def eval_epoch(
    model: nn.Module,
    dataset: TeleopDataset,
    idx: torch.Tensor,
    batch_size: int,
    device: torch.device,
    use_amp: bool,
) -> float:
    model.eval()
    total_loss = 0.0
    count = 0
    n_batches = idx.numel() // batch_size
    for i in range(n_batches):
        b = idx[i * batch_size : (i + 1) * batch_size]
        images, proprios, actions = dataset.gather(b)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
            pred = model(images, proprios)
            loss = nn.functional.l1_loss(pred, actions)
        total_loss += float(loss.item()) * b.numel()
        count += b.numel()
    return total_loss / max(count, 1)


def _next_run_dir(log_dir: Path, prefix: str) -> Path:
    log_dir.mkdir(parents=True, exist_ok=True)
    existing = {p.name for p in log_dir.iterdir() if p.is_dir()}
    idx = 1
    while f"{prefix}_{idx}" in existing:
        idx += 1
    return log_dir / f"{prefix}_{idx}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ACT on teleop demonstrations")
    parser.add_argument(
        "--input",
        type=str,
        nargs="+",
        dest="inputs",
        default=None,
        help="Input demo file(s); pass multiple files to train on merged recording sessions (default: most recently modified data/*.npz)",
    )
    parser.add_argument("--output", type=str, default="checkpoints/act_so101.pt", help="Output model checkpoint")
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to an ACT checkpoint to initialize weights from before training (e.g. checkpoints/act_so101.pt)",
    )
    parser.add_argument("--epochs", type=int, default=200, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW weight decay")
    parser.add_argument("--chunk-size", type=int, default=8, help="Action chunk length")
    parser.add_argument("--hidden-dim", type=int, default=128, help="Transformer hidden dim")
    parser.add_argument("--val-split", type=float, default=0.1, help="Validation split ratio (episode-level)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for train/val split and torch")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--augment", action="store_true", help="Enable random crop/brightness/contrast augmentation")
    parser.add_argument("--patience", type=int, default=50, help="Early stopping patience; 0 disables early stopping")
    parser.add_argument("--freeze-backbone", action="store_true", help="Freeze ResNet-18 backbone, only train transformer and policy head")
    parser.add_argument("--no-pretrained", action="store_true", help="Do not load ImageNet pretrained weights for ResNet-18")
    parser.add_argument(
        "--tensorboard-log",
        type=str,
        default="logs/",
        help="TensorBoard log directory; the run is written to <dir>/ACT_N alongside SB3's PPO_N runs",
    )
    parser.add_argument("--no-compile", action="store_true", help="Disable torch.compile (enabled by default on CUDA)")
    parser.add_argument("--no-amp", action="store_true", help="Disable bfloat16 autocast (enabled by default on CUDA)")
    args = parser.parse_args()

    if args.inputs is None:
        candidates = sorted(Path("data").glob("*.npz"), key=lambda p: p.stat().st_mtime)
        if not candidates:
            raise FileNotFoundError("No .npz demo files found in data/; run teleop_record.py first or pass --input")
        args.inputs = [str(candidates[-1])]

    total_frames = 0
    total_episodes = 0
    for path in args.inputs:
        if not Path(path).exists():
            raise FileNotFoundError(f"Demo file not found: {path}")
        data = np.load(path)
        n_frames = len(data["proprios"])
        n_episodes = int(data["episode_starts"].sum()) if "episode_starts" in data else 0
        total_frames += n_frames
        total_episodes += n_episodes
        print(f"  demos: {path} ({n_frames} frames, {n_episodes} episodes)")
    print(f"Training on {len(args.inputs)} file(s): {total_frames} frames, {total_episodes} episodes total")

    if args.resume and not Path(args.resume).exists():
        raise FileNotFoundError(f"Resume checkpoint not found: {args.resume}")

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    if args.seed is not None:
        torch.manual_seed(args.seed)

    dataset = TeleopDataset(args.inputs, chunk_size=args.chunk_size, device=device, augment=args.augment)
    print(
        f"Dataset on {device.type}: {dataset.valid_idx.numel()}/{len(dataset)} boundary-safe samples, "
        f"{len(dataset.image_tensors)} camera(s), images {tuple(dataset.image_tensors[0].shape)} uint8, "
        f"action chunks {tuple(dataset.action_chunks.shape)}"
    )

    valid_np = dataset.valid_idx.cpu().numpy()
    if dataset.episode_starts is not None:
        ep_start_pos = np.flatnonzero(dataset.episode_starts)
        n_eps = len(ep_start_pos)
        ep_of_frame = np.searchsorted(ep_start_pos, np.arange(len(dataset.episode_starts)), side="right") - 1
        rng = np.random.default_rng(args.seed)
        ep_perm = rng.permutation(n_eps)
        n_val_eps = max(1, int(n_eps * args.val_split))
        is_val = np.isin(ep_of_frame[valid_np], ep_perm[:n_val_eps])
        val_idx = torch.from_numpy(valid_np[is_val]).long().to(device)
        train_idx = torch.from_numpy(valid_np[~is_val]).long().to(device)
        print(f"Episode-level split: {n_eps - n_val_eps} train / {n_val_eps} val episodes")
    else:
        perm = torch.randperm(dataset.valid_idx.numel(), device=device)
        val_size = int(dataset.valid_idx.numel() * args.val_split)
        val_idx = dataset.valid_idx[perm[:val_size]]
        train_idx = dataset.valid_idx[perm[val_size:]]

    model = ACTPolicy(
        proprio_dim=dataset.proprio_dim,
        action_dim=dataset.action_dim,
        chunk_size=args.chunk_size,
        hidden_dim=args.hidden_dim,
        encoder_pretrained=not args.no_pretrained,
    ).to(device)

    if args.freeze_backbone:
        for param in model.image_encoder.features.parameters():
            param.requires_grad = False
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"Frozen ResNet-18 backbone: {trainable}/{total} parameters trainable")

    if args.resume:
        model.load_state_dict(torch.load(args.resume, map_location=device))
        print(f"Initialized weights from {args.resume}")

    # Keep the raw module for checkpoint saving: torch.compile wraps it and
    # would prefix state_dict keys with "_orig_mod.", breaking act_rollout.
    raw_model = model
    if device.type == "cuda" and not args.no_compile:
        model = torch.compile(model)
        print("torch.compile enabled (first epoch includes one-time compilation)")

    writer = None
    if SummaryWriter is not None:
        run_dir = _next_run_dir(Path(args.tensorboard_log), "ACT")
        writer = SummaryWriter(run_dir)
        writer.add_text("config", str(vars(args)))
        print(f"TensorBoard logging to {run_dir}")
    else:
        print("tensorboard package not installed; skipping TensorBoard logging")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

    best_val_loss = float("inf")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    use_amp = device.type == "cuda" and not args.no_amp

    early_stopping = EarlyStopping(patience=args.patience) if args.patience > 0 else None

    t_start = time.time()
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, dataset, train_idx, args.batch_size, optimizer, device, use_amp)
        val_loss = eval_epoch(model, dataset, val_idx, args.batch_size, device, use_amp)
        scheduler.step(val_loss)
        if writer is not None:
            writer.add_scalar("train/loss", train_loss, epoch)
            writer.add_scalar("val/loss", val_loss, epoch)
            writer.add_scalar("train/learning_rate", optimizer.param_groups[0]["lr"], epoch)
        if epoch % 10 == 0:
            sps = (time.time() - t_start) / epoch
            print(f"Epoch {epoch}: train_loss={train_loss:.6f} val_loss={val_loss:.6f} ({sps:.2f} s/epoch)")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(raw_model.state_dict(), args.output)
        if early_stopping is not None and early_stopping(val_loss):
            print(f"Early stopping triggered at epoch {epoch} (best val_loss={best_val_loss:.6f})")
            break

    print(f"Best val loss: {best_val_loss:.6f}, checkpoint saved to {args.output}")
    if writer is not None:
        writer.add_text("result", f"best_val_loss={best_val_loss:.6f}")
        writer.close()


if __name__ == "__main__":
    main()
