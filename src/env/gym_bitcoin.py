from __future__ import annotations
from src.environment.rewards import RewardCalculator

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from pathlib import Path
from loguru import logger
from src.utils import config, root


class GymBitcoinEnv(gym.Env):

    metadata = {"render_modes": []}

    def __init__(
        self,
        data_path: str | None = None,
        window_len: int = 48,
        max_trade_step: float = 0.2,
        transaction_cost_rate: float = 0.0005,
        max_drawdown: float = 0.3,
        initial_capital: float = 10000.0,
    ):
        super().__init__()

        #Load config if not provided
        if data_path is None:
            data_path = str(root(config["paths"]["feature_engineered_dir"]) / "train.parquet")

        self.data_path = data_path
        self.window_len = window_len
        self.max_trade_step = max_trade_step
        self.transaction_cost_rate = transaction_cost_rate
        self.max_drawdown = max_drawdown
        self.initial_capital = initial_capital

        #Load data
        self.load_data()

        #Define spaces
        obs_dim = self.window_len * self.n_features + 4  # market window + portfolio
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )

        # Episode state (initialized in reset)
        self.current_step = 0
        self.capital = 0.0
        self.position = 0.0
        self.entry_price = 0.0
        self.holding_time = 0
        self.peak_capital = 0.0

        logger.info(
            f"GymBitcoinEnv initialized: "
            f"data={self.data_path}, window={window_len}, "
            f"features={self.n_features}, obs_dim={obs_dim}, max_steps={self.max_steps}"
        )

    def load_data(self) -> None:
        df = pd.read_parquet(self.data_path)

        #Identify feature columns (exclude OHLCV + timestamp)
        exclude_cols = {"timestamp", "open", "high", "low", "close", "volume"}
        self.feature_cols = [c for c in df.columns if c not in exclude_cols]

        self.features = df[self.feature_cols].values.astype(np.float32)
        self.prices = df["close"].values.astype(np.float32)
        self.timestamps = df["timestamp"].values
        self.n_features = len(self.feature_cols)

        #Max steps = data length - window_len - 1 (need at least one step ahead)
        self.max_steps = len(self.features) - self.window_len - 1

        if self.max_steps <= 0:
            raise ValueError(
                f"Not enough data: {len(self.features)} rows, "
                f"need at least {self.window_len + 2}"
            )

    def get_obs(self) -> np.ndarray:
        start = self.current_step - self.window_len
        market_window = self.features[start:self.current_step].flatten()

        #Portfolio vector: [cash_pct, position, unrealized_pnl_pct, holding_time_norm]
        #Use percentage return for unrealized PnL
        if self.entry_price > 0:
            unrealized_return = (self.prices[self.current_step] - self.entry_price) / self.entry_price
        else:
            unrealized_return = 0.0
        unrealized_pnl = self.position * self.capital * unrealized_return
        portfolio = np.array([
            self.capital / self.initial_capital,
            self.position,
            unrealized_pnl / self.initial_capital,
            self.holding_time / 100.0,  # normalize
        ], dtype=np.float32)

        return np.concatenate([market_window, portfolio])

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)

        #random start for generalization
        self.current_step = self.np_random.integers(
                    self.window_len, 
                max(self.window_len + 1, self.max_steps // 2)
                                )
        self.capital = self.initial_capital
        self.position = 0.0
        self.entry_price = self.prices[self.current_step]
        self.holding_time = 0
        self.peak_capital = self.initial_capital

        obs = self.get_obs()
        info = {
            "step": self.current_step,
            "price": self.prices[self.current_step],
            "position": self.position,
            "capital": self.capital,
        }
        return obs, info

    def step(self, action: np.ndarray):
        # 1. Clip and smooth action
        target_pos = float(np.clip(action[0], -1.0, 1.0))
        delta = np.clip(
            target_pos - self.position,
            -self.max_trade_step,
            self.max_trade_step
        )
        new_pos = self.position + delta

        # 2. Transaction cost (on notional traded)
        trade_notional = abs(delta) * self.capital
        cost = trade_notional * self.transaction_cost_rate
        self.capital -= cost

        # 3. Advance market cursor
        self.current_step += 1
        price = self.prices[self.current_step]
        prev_price = self.prices[self.current_step - 1]

        # 4. P&L from PREVIOUS position (using percentage return)
        if prev_price > 0:
            price_return = (price - prev_price) / prev_price
        else:
            price_return = 0.0
        pnl = self.position * self.capital * price_return
        self.capital += pnl

        # 5. Update portfolio state
        self.holding_time = 0 if delta != 0 else self.holding_time + 1

        if delta != 0:
            self.entry_price = price
        self.position = new_pos

        # 6. Drawdown tracking
        self.peak_capital = max(self.peak_capital, self.capital)
        drawdown = (self.peak_capital - self.capital) / self.peak_capital

        # 7. Reward (minimal: net PnL)
        step_return = price_return * self.position
        reward = self.reward_calc.calculate(
        step_return=step_return,
        drawdown=drawdown,
        position_change=abs(delta),
                        )

        # 8. Termination conditions
        terminated = (
            self.current_step >= self.max_steps
            or drawdown >= self.max_drawdown
            or self.capital <= 0
        )
        truncated = False

        # 9. Observation & info
        obs = self.get_obs()
        info = {
            "step": self.current_step,
            "price": price,
            "position": self.position,
            "pnl": pnl,
            "cost": cost,
            "capital": self.capital,
            "drawdown": drawdown,
            "trade_notional": trade_notional,
        }

        return obs, float(reward), terminated, truncated, info

    def render(self):
        """Optional render - not implemented."""
        pass

    def close(self):
        """Cleanup."""
        pass
