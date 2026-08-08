"""Offline image augmentation for ACT demonstration datasets.

Generates N augmented copies of a recorded .npz demo file so that training
can run without on-the-fly augmentation. All non-image arrays (proprios,
actions, rewards, terminals, episode_starts, cube_positions) are repeated
to match the augmented copies.

Example:
    python scripts/augment_demos.py \
        --input data/teleop_demos.npz \
        --copies 5 \
        --output data/teleop_demos_aug.npz

The output file contains 5x the frames of the input, with each copy having
its own random crop/brightness/contrast jitter. Train with:

    python scripts/train_act.py \
        --input data/teleop_demos.npz data/teleop_demos_aug.npz \
        ...
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch


def _random_crop(x: torch.Tensor, pad: int) -> torch.Tensor:
    """Crop a (B,H,W,C) uint8 tensor by ``pad`` pixels on each side using reflect padding."""
    B, H, W, C = x.shape
    x = x.permute(0, 3, 1, 2).float() / 255.0
    x = torch.nn.functional.pad(x, (pad, pad, pad, pad), mode="reflect")
    top = torch.randint(0, 2 * pad + 1, (B,))
    left = torch.randint(0, 2 * pad + 1, (B,))
    crops = torch.stack([x[i, :, top[i] : top[i] + H, left[i] : left[i] + W] for i in range(B)])
    return (crops * 255.0).clamp_(0.0, 255.0).to(torch.uint8).permute(0, 2, 3, 1)


def _brightness(x: torch.Tensor, delta: float) -> torch.Tensor:
    x = x.float() / 255.0
    b = torch.empty(x.size(0), 1, 1, 1).uniform_(-delta, delta)
    return (torch.clamp(x + b, 0.0, 1.0) * 255.0).to(torch.uint8)


def _contrast(x: torch.Tensor, delta: float) -> torch.Tensor:
    x = x.float() / 255.0
    factor = torch.empty(x.size(0), 1, 1, 1).uniform_(1 - delta, 1 + delta)
    mean = x.mean(dim=(1, 2, 3), keepdim=True)
    return (torch.clamp((x - mean) * factor + mean, 0.0, 1.0) * 255.0).to(torch.uint8)


def augment_images(images: torch.Tensor, pad: int, brightness: float, contrast: float) -> torch.Tensor:
    """Apply random crop, brightness and contrast jitter to a uint8 (B,H,W,C) tensor."""
    if pad > 0:
        images = _random_crop(images, pad)
    if brightness > 0.0:
        images = _brightness(images, brightness)
    if contrast > 0.0:
        images = _contrast(images, contrast)
    return images


def augment_npz(
    input_paths: Sequence[str | Path],
    output_path: str | Path,
    copies: int = 5,
    pad: int = 4,
    brightness: float = 0.2,
    contrast: float = 0.2,
) -> None:
    output_path = Path(output_path)
    all_data: dict[str, list[np.ndarray]] = {}
    image_keys: list[str] | None = None
    total_frames = 0

    for path in input_paths:
        path = Path(path)
        data = dict(np.load(path))
        if image_keys is None:
            image_keys = [k for k in ("images_wrist", "images_overhead") if k in data] or ["images"]
        current_image_keys = [k for k in ("images_wrist", "images_overhead") if k in data] or ["images"]
        if set(current_image_keys) != set(image_keys):
            raise ValueError(f"Camera keys mismatch: {path} has {current_image_keys}, expected {image_keys}")

        n = len(data[image_keys[0]])
        total_frames += n
        print(f"Loaded {path}: {n} frames, {len(image_keys)} camera(s)")
        for k in data:
            all_data.setdefault(k, []).append(np.asarray(data[k]))

    merged: dict[str, np.ndarray] = {}
    for k, parts in all_data.items():
        if k in image_keys:
            merged[k] = np.concatenate(parts, axis=0)
        else:
            # Only repeat non-image arrays that have one row per frame.
            merged[k] = np.concatenate(parts, axis=0)

    n_frames = len(merged[image_keys[0]])
    print(f"Generating {copies} augmented copies -> {output_path}")

    out_images: dict[str, list[np.ndarray]] = {k: [] for k in image_keys}
    for copy in range(1, copies + 1):
        t0 = time.time()
        for k in image_keys:
            tensor = torch.from_numpy(merged[k])
            aug = augment_images(tensor, pad=pad, brightness=brightness, contrast=contrast)
            out_images[k].append(aug.numpy())
        print(f"  copy {copy}/{copies} done in {time.time() - t0:.2f}s")

    for k in image_keys:
        merged[k] = np.concatenate(out_images[k], axis=0)

    # Repeat non-image arrays for each augmented copy.
    for k in merged:
        if k in image_keys:
            continue
        arr = merged[k]
        if len(arr) != n_frames:
            print(f"  skipping {k}: length {len(arr)} != {n_frames}")
            continue
        merged[k] = np.tile(arr, (copies, *([1] * (arr.ndim - 1))))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **merged)
    print(f"Saved {output_path} ({len(merged[image_keys[0]])} frames, {copies}x augmentation)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline image augmentation for ACT demos")
    parser.add_argument("--input", type=str, nargs="+", required=True, help="Input demo .npz file(s); shell globs are supported")
    parser.add_argument("--output", type=str, default=None, help="Output file (default: <first_input>_aug.npz)")
    parser.add_argument("--copies", type=int, default=5, help="Number of augmented copies to generate")
    parser.add_argument("--pad", type=int, default=4, help="Random crop padding in pixels")
    parser.add_argument("--brightness", type=float, default=0.2, help="Brightness jitter range")
    parser.add_argument("--contrast", type=float, default=0.2, help="Contrast jitter range")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    if args.output is None:
        p = Path(args.input[0])
        args.output = str(p.with_stem(p.stem + "_aug"))

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    augment_npz(
        args.input,
        args.output,
        copies=args.copies,
        pad=args.pad,
        brightness=args.brightness,
        contrast=args.contrast,
    )


if __name__ == "__main__":
    main()
