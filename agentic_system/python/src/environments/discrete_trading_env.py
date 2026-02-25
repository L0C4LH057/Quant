"""
Discrete action trading environment for DQN.

Maps discrete actions {0=sell, 1=hold, 2=buy} to continuous trading actions.

Design Decision:
    DQN only supports spaces.Discrete. Rather than reimplementing the entire
    environment, this wraps TradingEnv and translates discrete → continuous.
    This keeps all the validated trading logic in one place.
"""
import logging
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from .trading_env import TradingEnv

logger = logging.getLogger(__name__)


# Action mapping: discrete → continuous
DISCRETE_ACTIONS = {
    0: -1.0,   # Sell (full position)
    1: 0.0,    # Hold
    2: 1.0,    # Buy (full allocation)
}


class DiscreteTradingEnv(TradingEnv):
    """
    Trading environment with discrete actions for DQN.

    Maps 3 discrete actions to continuous:
        0 = Sell all holdings
        1 = Hold (do nothing)
        2 = Buy with all available cash

    Uses the same observation space as TradingEnv.

    Example:
        >>> env = DiscreteTradingEnv(df)
        >>> obs, info = env.reset()
        >>> obs, reward, terminated, truncated, info = env.step(2)  # Buy
    """

    def __init__(
        self,
        df: pd.DataFrame,
        initial_balance: float = 1_000_000,
        transaction_cost_pct: float = 0.001,
        window_size: int = 30,
        reward_scaling: float = 1e-4,
        max_shares: float = 100.0,
        feature_columns: Optional[List[str]] = None,
        sell_amount: float = 1.0,
        buy_amount: float = 1.0,
    ):
        """
        Initialize discrete trading environment.

        Args:
            df: Market data (must have: date, close)
            initial_balance: Starting capital
            transaction_cost_pct: Cost per trade (0.001 = 0.1%)
            window_size: Lookback period for price observations
            reward_scaling: Scale factor for reward normalization
            max_shares: Maximum shares to hold
            feature_columns: Additional columns to include in observation
            sell_amount: Fraction of holdings to sell (0-1), default 1.0 = sell all
            buy_amount: Fraction of cash to use for buying (0-1), default 1.0 = buy max
        """
        super().__init__(
            df=df,
            initial_balance=initial_balance,
            transaction_cost_pct=transaction_cost_pct,
            window_size=window_size,
            reward_scaling=reward_scaling,
            max_shares=max_shares,
            feature_columns=feature_columns,
        )

        # Validate amounts
        if not 0 < sell_amount <= 1:
            raise ValueError(f"sell_amount must be in (0, 1], got {sell_amount}")
        if not 0 < buy_amount <= 1:
            raise ValueError(f"buy_amount must be in (0, 1], got {buy_amount}")

        self.sell_amount = sell_amount
        self.buy_amount = buy_amount

        # Override action space to Discrete(3)
        self.action_space = spaces.Discrete(3)

        # Update action mapping with configurable amounts
        self._action_map = {
            0: -self.sell_amount,  # Sell
            1: 0.0,               # Hold
            2: self.buy_amount,   # Buy
        }

        logger.info(
            f"DiscreteTradingEnv: sell={self.sell_amount:.0%}, "
            f"buy={self.buy_amount:.0%}"
        )

    def step(
        self,
        action: int,
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one trading step with discrete action.

        Args:
            action: Discrete action (0=sell, 1=hold, 2=buy)

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)

        Raises:
            ValueError: If action is not in {0, 1, 2}
        """
        # Validate action
        if not isinstance(action, (int, np.integer)):
            action = int(action)

        if action not in self._action_map:
            raise ValueError(
                f"Invalid action {action}. Must be 0 (sell), 1 (hold), or 2 (buy)."
            )

        # Map to continuous action and delegate to parent
        continuous_action = np.array([self._action_map[action]], dtype=np.float32)
        return super().step(continuous_action)
