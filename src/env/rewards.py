from collections import deque

import numpy as np


SHARPE_WINDOW = 100
DRAWDOWN_PENALTY_SCALE = 0.1
OVERTRADE_PENALTY_SCALE = 0.001
MIN_BUFFER_SIZE = 2
EPSILON = 0.01


class RewardCalculator:
    def __init__(
        self,
        window: int = SHARPE_WINDOW,
        drawdown_scale: float = DRAWDOWN_PENALTY_SCALE,
        overtrade_scale: float = OVERTRADE_PENALTY_SCALE,
    ):
        self.window = window
        self.drawdown_scale = drawdown_scale
        self.overtrade_scale = overtrade_scale
        self.returns_buffer: deque = deque(maxlen=window)

    def reset(self) -> None:
        self.returns_buffer.clear()

    def _sharpe_reward(self, step_return: float) -> float:
        self.returns_buffer.append(step_return)

        if len(self.returns_buffer) < MIN_BUFFER_SIZE:
            return 0.0

        mean_r = np.mean(self.returns_buffer)
        std_r = np.std(self.returns_buffer) + EPSILON
        return float(mean_r / std_r)

    def _drawdown_penalty(self, drawdown: float) -> float:
        return self.drawdown_scale * max(drawdown, 0.0)

    def _overtrade_penalty(self, position_change: float) -> float:
        return self.overtrade_scale * abs(position_change)

    def calculate(
        self,
        step_return: float,
        drawdown: float,
        position_change: float,
    ) -> float:
        sharpe  = self._sharpe_reward(step_return)
        dd_pen  = self._drawdown_penalty(drawdown)
        ot_pen  = self._overtrade_penalty(position_change)
        return sharpe - dd_pen - ot_pen

    @property
    def buffer_mean(self) -> float:
        if not self.returns_buffer:
            return 0.0
        return float(np.mean(self.returns_buffer))

    @property
    def buffer_std(self) -> float:
        if not self.returns_buffer:
            return 0.0
        return float(np.std(self.returns_buffer))

    @property
    def annualized_sharpe(self) -> float:
        if len(self.returns_buffer) < MIN_BUFFER_SIZE:
            return 0.0
        mean_r = np.mean(self.returns_buffer)
        std_r  = np.std(self.returns_buffer) + EPSILON
        return float((mean_r / std_r) * np.sqrt(8760))
