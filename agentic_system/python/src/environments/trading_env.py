"""
Base trading environment for RL agents.

Implements gym.Env interface for compatibility with Stable-Baselines3.

Security:
    - All inputs validated
    - Actions clipped to valid range
    - Trade logging for audit trail

Token Optimization:
    - Clear state representation
    - Minimal observation space
    - Efficient reward calculation
"""
import logging
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from ..rewards.reward_functions import RewardCalculator
from ..utils.validators import validate_dataframe, validate_positive

logger = logging.getLogger(__name__)


class TradingEnv(gym.Env):
    """
    Base trading environment for RL agents.

    Observation:
        - Price window (normalized)
        - Technical indicators
        - Portfolio state (cash ratio, holdings)

    Action:
        - Continuous: [-1, 1] per asset
        - -1 = sell all, 0 = hold, +1 = buy max

    Reward:
        - Portfolio value change (can be customized)

    Example:
        >>> import pandas as pd
        >>> df = pd.DataFrame({
        ...     "date": pd.date_range("2020-01-01", periods=100),
        ...     "close": np.random.rand(100) * 100 + 100,
        ...     "volume": np.random.randint(1000, 10000, 100)
        ... })
        >>> env = TradingEnv(df)
        >>> obs, info = env.reset()
        >>> action = env.action_space.sample()
        >>> obs, reward, terminated, truncated, info = env.step(action)
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        df: pd.DataFrame,
        initial_balance: float = 1_000_000,
        transaction_cost_pct: float = 0.001,
        window_size: int = 30,
        reward_scaling: float = 1e-4,
        max_shares: float = 100.0,
        feature_columns: Optional[List[str]] = None,
        reward_type: Optional[str] = None,
        position_change_penalty: float = 0.0,
    ):
        """
        Initialize trading environment.

        Args:
            df: Market data (must have: date, close; optional: high, low, volume)
            initial_balance: Starting capital
            transaction_cost_pct: Cost per trade (0.001 = 0.1%)
            window_size: Lookback period for price observations
            reward_scaling: Scale factor for reward normalization
            max_shares: Maximum shares to hold
            feature_columns: Additional columns to include in observation
            reward_type: Reward strategy to use. None = simple return,
                or one of "sharpe", "sortino", "risk_adjusted", "profit".
                Using Sharpe/Sortino trains agents to maximize risk-adjusted
                returns rather than raw profit.
            position_change_penalty: Penalty per trade to discourage
                excessive position changes (default: 0.0 = no penalty).
                Recommended: 0.0001 for moderate penalty.

        Raises:
            ValueError: If parameters are invalid
        """
        super().__init__()

        # Validate inputs
        df = validate_dataframe(df, required_columns=["close"], min_rows=window_size + 1)
        validate_positive(initial_balance, "initial_balance")
        validate_positive(window_size, "window_size")

        if not 0 <= transaction_cost_pct < 1:
            raise ValueError(f"transaction_cost_pct must be in [0, 1), got {transaction_cost_pct}")

        # Store parameters
        self.df = df.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.transaction_cost_pct = transaction_cost_pct
        self.window_size = window_size
        self.reward_scaling = reward_scaling
        self.max_shares = max_shares
        self.position_change_penalty = position_change_penalty

        # Setup reward calculator if a strategy is specified
        self._reward_calculator: Optional[RewardCalculator] = None
        if reward_type is not None:
            valid_types = {"sharpe", "sortino", "risk_adjusted", "profit"}
            if reward_type not in valid_types:
                raise ValueError(
                    f"Invalid reward_type '{reward_type}'. "
                    f"Must be one of {valid_types}"
                )
            self._reward_calculator = RewardCalculator(
                initial_value=initial_balance,
                window=50,
                reward_type=reward_type,
            )

        # Determine feature columns
        if feature_columns is None:
            # Auto-detect indicator columns (exclude non-numeric metadata)
            exclude = {
                "date", "time", "timestamp", "datetime",
                "open", "high", "low", "close", "volume",
            }
            self.feature_columns = [
                c for c in df.columns
                if c not in exclude and pd.api.types.is_numeric_dtype(df[c])
            ]
        else:
            self.feature_columns = feature_columns

        # Calculate observation dimension
        # window prices + features + (cash_ratio, holdings_ratio)
        self.n_features = len(self.feature_columns)
        obs_dim = window_size + self.n_features + 2

        # Define spaces
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )

        # Action: single continuous value [-1, 1]
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(1,),
            dtype=np.float32,
        )

        # State variables (initialized in reset)
        self.current_step = 0
        self.balance = initial_balance
        self.holdings = 0.0
        self.portfolio_value = initial_balance
        self.trades_log: List[Dict[str, Any]] = []

        # For reward calculation
        self._previous_portfolio_value = initial_balance
        self._previous_action = 0.0

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset environment to initial state.

        Args:
            seed: Random seed
            options: Additional options

        Returns:
            Tuple of (observation, info)
        """
        super().reset(seed=seed)

        self.current_step = self.window_size
        self.balance = self.initial_balance
        self.holdings = 0.0
        self.portfolio_value = self.initial_balance
        self._previous_portfolio_value = self.initial_balance
        self._previous_action = 0.0
        self.trades_log = []

        # Reset reward calculator if used
        if self._reward_calculator is not None:
            self._reward_calculator.reset(initial_value=self.initial_balance)

        obs = self._get_observation()
        info = self._get_info()

        return obs, info

    def step(
        self,
        action: np.ndarray,
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one trading step.

        Args:
            action: Trading action [-1, 1]
                -1 = sell all holdings
                 0 = hold
                +1 = buy with all available cash

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        # Validate and clip action
        action = np.clip(np.atleast_1d(action), -1, 1)[0]

        # Get current price
        current_price = float(self.df.loc[self.current_step, "close"])

        # Execute trade
        self._execute_trade(action, current_price)

        # Update portfolio value
        self.portfolio_value = self.balance + (self.holdings * current_price)

        # Calculate reward
        reward = self._calculate_reward()

        # Apply position-change penalty to discourage excessive trading
        if self.position_change_penalty > 0:
            action_change = abs(action - self._previous_action)
            reward -= self.position_change_penalty * action_change

        self._previous_action = action

        # Store for next reward calculation
        self._previous_portfolio_value = self.portfolio_value

        # Move to next step
        self.current_step += 1

        # Check termination
        terminated = self.current_step >= len(self.df) - 1
        truncated = False

        # Check for bankruptcy
        if self.portfolio_value <= 0:
            terminated = True
            reward = -1.0  # Penalty for bankruptcy

        obs = self._get_observation()
        info = self._get_info()

        return obs, reward, terminated, truncated, info

    def _execute_trade(self, action: float, price: float) -> None:
        """
        Execute a trade based on action.

        Args:
            action: Trading action [-1, 1]
            price: Current asset price
        """
        if action > 0.01:  # Buy threshold
            # Calculate how much we can buy (accounting for transaction costs)
            available_for_purchase = self.balance * action / (1 + self.transaction_cost_pct)
            shares_to_buy = available_for_purchase / price

            # Apply transaction cost
            cost = shares_to_buy * price * (1 + self.transaction_cost_pct)

            # Clamp to balance (handles floating-point precision)
            cost = min(cost, self.balance)

            if cost <= self.balance and shares_to_buy > 0:
                self.holdings += shares_to_buy
                self.balance -= cost
                self._log_trade("buy", shares_to_buy, price, cost)

        elif action < -0.01:  # Sell threshold
            # Calculate how much to sell
            shares_to_sell = self.holdings * abs(action)

            if shares_to_sell > 0:
                # Apply transaction cost
                revenue = shares_to_sell * price * (1 - self.transaction_cost_pct)

                self.holdings -= shares_to_sell
                self.balance += revenue
                self._log_trade("sell", shares_to_sell, price, revenue)

    def _calculate_reward(self) -> float:
        """
        Calculate reward for current step.

        Uses pluggable RewardCalculator if configured (Sharpe/Sortino/
        risk-adjusted), otherwise falls back to simple return-based reward.

        Returns:
            Scaled reward value
        """
        # Use advanced reward calculator if configured
        if self._reward_calculator is not None:
            return self._reward_calculator.calculate(self.portfolio_value) * self.reward_scaling

        # Fallback: simple return-based reward
        if self._previous_portfolio_value > 0:
            pct_change = (
                (self.portfolio_value - self._previous_portfolio_value)
                / self._previous_portfolio_value
            )
        else:
            pct_change = 0.0

        return pct_change * self.reward_scaling

    def _get_observation(self) -> np.ndarray:
        """Get current observation vector."""
        # Price window (normalized)
        # Note: df.loc[a:b] is inclusive on both ends, so use end-exclusive
        end = self.current_step
        start = max(0, end - self.window_size)
        prices = self.df.loc[start:end - 1, "close"].values

        # Normalize prices
        if len(prices) > 1 and prices.std() > 0:
            prices = (prices - prices.mean()) / prices.std()
        else:
            prices = np.zeros(self.window_size)

        # Pad if needed
        if len(prices) < self.window_size:
            prices = np.pad(prices, (self.window_size - len(prices), 0), mode="edge")

        # Truncate if needed (safety guard)
        prices = prices[-self.window_size:]

        # Feature values (current step)
        features = []
        for col in self.feature_columns:
            if col in self.df.columns:
                val = float(self.df.loc[self.current_step, col])
                features.append(val if not np.isnan(val) else 0.0)

        # Portfolio state
        cash_ratio = self.balance / self.initial_balance
        current_price = float(self.df.loc[self.current_step, "close"])
        holdings_value = self.holdings * current_price
        holdings_ratio = holdings_value / self.initial_balance

        # Combine all
        obs = np.concatenate([
            prices.astype(np.float32),
            np.array(features, dtype=np.float32),
            np.array([cash_ratio, holdings_ratio], dtype=np.float32),
        ])

        return obs

    def _get_info(self) -> Dict[str, Any]:
        """Get current info dictionary."""
        current_price = float(self.df.loc[self.current_step, "close"])

        return {
            "step": self.current_step,
            "balance": self.balance,
            "holdings": self.holdings,
            "portfolio_value": self.portfolio_value,
            "current_price": current_price,
            "num_trades": len(self.trades_log),
            "return_pct": (self.portfolio_value - self.initial_balance)
            / self.initial_balance
            * 100,
        }

    def _log_trade(
        self,
        side: str,
        shares: float,
        price: float,
        amount: float,
    ) -> None:
        """Log trade for audit trail."""
        trade = {
            "step": self.current_step,
            "side": side,
            "shares": shares,
            "price": price,
            "amount": amount,
            "balance_after": self.balance,
            "holdings_after": self.holdings,
        }

        # Add date if available
        if "date" in self.df.columns:
            trade["date"] = self.df.loc[self.current_step, "date"]

        self.trades_log.append(trade)

    def render(self) -> None:
        """Render current state to console."""
        info = self._get_info()
        print(
            f"Step: {info['step']:4d} | "
            f"Portfolio: ${info['portfolio_value']:,.2f} | "
            f"Return: {info['return_pct']:+.2f}% | "
            f"Holdings: {info['holdings']:.4f}"
        )

    def get_trades_df(self) -> pd.DataFrame:
        """Get trades log as DataFrame."""
        if not self.trades_log:
            return pd.DataFrame()
        return pd.DataFrame(self.trades_log)
