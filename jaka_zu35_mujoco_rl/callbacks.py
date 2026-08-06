"""Stable-Baselines3 callbacks for fine-grained training diagnostics."""

from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class RewardLoggerCallback(BaseCallback):
    """Logs per-component reward means and episode success rate to TensorBoard.

    The callback reads ``reward_components`` and ``is_success`` from the
    environment ``info`` dict, averages them over a rollout window, and writes
    scalar summaries so reward shaping can be tuned without guessing.
    """

    def __init__(
        self,
        component_names: list[str] | None = None,
        log_interval: int = 2048,
        success_rate_window: int = 100,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)
        self._component_names = component_names
        self._log_interval = log_interval
        self._success_window = success_rate_window
        self._reward_buffers: dict[str, deque[float]] | None = None
        self._success_buffer: deque[float] = deque(maxlen=success_rate_window)

    def _on_training_start(self) -> None:
        if self._component_names is not None:
            self._reward_buffers = {
                name: deque(maxlen=self._log_interval)
                for name in self._component_names
            }

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])

        for info, done in zip(infos, dones):
            components = info.get("reward_components")
            if components:
                if self._reward_buffers is None:
                    self._reward_buffers = {
                        name: deque(maxlen=self._log_interval)
                        for name in components.keys()
                    }
                for name, value in components.items():
                    if name in self._reward_buffers:
                        self._reward_buffers[name].append(float(value))

            if done:
                self._success_buffer.append(1.0 if info.get("is_success") else 0.0)

        if self.n_calls % self._log_interval == 0 and self.n_calls > 0:
            self._record_metrics()

        return True

    def _record_metrics(self) -> None:
        if self._reward_buffers is not None:
            for name, buffer in self._reward_buffers.items():
                if buffer:
                    self.logger.record(f"reward/{name}_mean", float(np.mean(buffer)))

        if self._success_buffer:
            self.logger.record(
                "rollout/success_rate",
                float(np.mean(self._success_buffer)),
            )
