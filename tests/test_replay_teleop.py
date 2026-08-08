import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from jaka_zu35_mujoco_rl.envs import make_env

spec = importlib.util.spec_from_file_location(
    "replay_teleop",
    Path(__file__).parent.parent / "scripts" / "replay_teleop.py",
)
replay_teleop = importlib.util.module_from_spec(spec)
spec.loader.exec_module(replay_teleop)


@pytest.fixture
def demo_path(tmp_path: Path) -> Path:
    path = tmp_path / "demos.npz"
    np.savez_compressed(
        path,
        images=np.zeros((10, 64, 64, 3), dtype=np.uint8),
        observations=np.zeros((10, 32), dtype=np.float64),
        proprios=np.zeros((10, 8), dtype=np.float64),
        actions=np.linspace(-1, 1, 60).reshape(10, 6),
        rewards=np.zeros(10, dtype=np.float64),
        terminals=np.zeros(10, dtype=bool),
        episode_starts=np.array([True, False, False, False, False, True, False, False, False, False]),
        cube_positions=np.zeros((10, 3), dtype=np.float64),
    )
    return path


def test_load_demos_reads_all_keys(demo_path: Path) -> None:
    demos = replay_teleop.load_demos(demo_path)
    assert set(demos.keys()) == {
        "images",
        "observations",
        "proprios",
        "actions",
        "rewards",
        "terminals",
        "episode_starts",
        "cube_positions",
    }
    assert demos["actions"].shape == (10, 6)
    assert demos["episode_starts"].shape == (10,)


def test_split_episodes_splits_on_episode_starts(demo_path: Path) -> None:
    demos = replay_teleop.load_demos(demo_path)
    episodes = replay_teleop.split_episodes(demos["actions"], demos["episode_starts"])
    assert len(episodes) == 2
    assert len(episodes[0]) == 5
    assert len(episodes[1]) == 5


def test_split_episodes_starts_at_zero_if_no_start_flag() -> None:
    actions = np.arange(12).reshape(4, 3)
    starts = np.array([False, False, True, False])
    episodes = replay_teleop.split_episodes(actions, starts)
    assert len(episodes) == 2
    assert np.array_equal(episodes[0], actions[:2])
    assert np.array_equal(episodes[1], actions[2:])


def test_replay_episode_runs_all_steps() -> None:
    env = make_env("so101", render_mode="rgb_array", max_episode_steps=100)
    env.reset(seed=0)
    actions = np.zeros((4, 6), dtype=np.float64)
    steps = replay_teleop.replay_episode(env, actions, cube_positions=None, fps=1000, show_camera=False)
    assert steps == 4


def test_main_headless_replays_demos(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    demo_path = tmp_path / "demos.npz"
    np.savez_compressed(
        demo_path,
        images=np.zeros((4, 64, 64, 3), dtype=np.uint8),
        observations=np.zeros((4, 32), dtype=np.float64),
        proprios=np.zeros((4, 8), dtype=np.float64),
        actions=np.linspace(-1, 1, 24).reshape(4, 6),
        rewards=np.zeros(4, dtype=np.float64),
        terminals=np.zeros(4, dtype=bool),
        episode_starts=np.array([True, False, True, False]),
        cube_positions=np.zeros((4, 3), dtype=np.float64),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["replay_teleop.py", "--input", str(demo_path), "--robot", "so101", "--headless", "--fps", "1000"],
    )
    replay_teleop.main()
