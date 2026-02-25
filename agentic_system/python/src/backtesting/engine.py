"""
Backtesting engine for RL trading strategies.

Features:
    - Historical simulation with realistic execution
    - Walk-forward analysis with periodic retraining
    - Multi-asset support
    - Comprehensive trade logging

Token Optimization:
    - Single BacktestEngine class handles all simulation
    - Clean separation from training logic
"""
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Type, Union

import numpy as np
import pandas as pd
from stable_baselines3.common.base_class import BaseAlgorithm

from .metrics import BacktestMetrics, MetricsCalculator

logger = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    """
    Backtest configuration.
    
    Attributes:
        initial_balance: Starting capital
        transaction_cost: Cost per trade (as decimal, 0.001 = 0.1%)
        slippage: Slippage estimate per trade
        position_sizing: Max position size as fraction of portfolio
        risk_per_trade: Max risk per trade as fraction
        deterministic: Use deterministic policy
    """
    initial_balance: float = 100_000.0
    transaction_cost: float = 0.001
    slippage: float = 0.0005
    position_sizing: float = 1.0
    risk_per_trade: float = 0.02
    deterministic: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()


@dataclass
class BacktestResult:
    """
    Complete backtest result.
    
    Contains equity curve, trades, and computed metrics.
    """
    # Core results
    equity_curve: np.ndarray
    trades: pd.DataFrame
    metrics: BacktestMetrics
    
    # Metadata
    start_date: str = ""
    end_date: str = ""
    symbols: List[str] = field(default_factory=list)
    config: Optional[Dict[str, Any]] = None
    
    # Positions over time
    positions: Optional[np.ndarray] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "start_date": self.start_date,
            "end_date": self.end_date,
            "symbols": self.symbols,
            "metrics": self.metrics.to_dict(),
            "n_trades": len(self.trades),
            "final_equity": float(self.equity_curve[-1]),
        }


class BacktestEngine:
    """
    Backtesting engine for RL trading models.
    
    Simulates trading on historical data with realistic execution.
    
    Example:
        >>> engine = BacktestEngine(
        ...     model=trained_model,
        ...     env_factory=lambda df: TradingEnv(df),
        ...     config=BacktestConfig(initial_balance=100000)
        ... )
        >>> result = engine.run(market_data)
        >>> print(result.metrics.summary())
    """
    
    def __init__(
        self,
        model: Optional[BaseAlgorithm] = None,
        model_path: Optional[Union[str, Path]] = None,
        algorithm_class: Optional[Type[BaseAlgorithm]] = None,
        env_factory: Optional[Callable[[pd.DataFrame], Any]] = None,
        config: Optional[BacktestConfig] = None,
    ):
        """
        Initialize backtest engine.
        
        Args:
            model: Trained SB3 model instance
            model_path: Path to saved model (alternative to model)
            algorithm_class: Algorithm class for loading (required if model_path)
            env_factory: Factory function that creates env from DataFrame
            config: Backtest configuration
        """
        self.config = config or BacktestConfig()
        self.env_factory = env_factory
        
        # Load model
        if model is not None:
            self.model = model
        elif model_path is not None and algorithm_class is not None:
            self.model = algorithm_class.load(str(model_path))
            logger.info(f"Loaded model from {model_path}")
        else:
            self.model = None
            logger.warning("No model provided - engine in analysis-only mode")
        
        # Results tracking
        self.equity_history: List[float] = []
        self.trades_log: List[Dict[str, Any]] = []
        self.positions_history: List[float] = []
    
    def run(
        self,
        data: pd.DataFrame,
        symbols: Optional[List[str]] = None,
    ) -> BacktestResult:
        """
        Run backtest on historical data.
        
        Args:
            data: Market data DataFrame
            symbols: List of symbols being traded
            
        Returns:
            BacktestResult with equity curve and metrics
        """
        if self.model is None:
            raise ValueError("No model loaded - cannot run backtest")
        
        if self.env_factory is None:
            raise ValueError("No env_factory provided - cannot create environment")
        
        # Create environment
        env = self.env_factory(data)
        
        # Reset tracking
        self.equity_history = [self.config.initial_balance]
        self.trades_log = []
        self.positions_history = [0.0]
        
        # Run simulation
        obs, info = env.reset()
        done = False
        step = 0
        
        logger.info(f"Starting backtest: {len(data)} periods")
        
        while not done:
            # Get action from model
            action, _ = self.model.predict(obs, deterministic=self.config.deterministic)
            
            # Execute step
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Track state
            portfolio_value = info.get("portfolio_value", self.equity_history[-1])
            self.equity_history.append(portfolio_value)
            
            position = info.get("shares_held", 0)
            self.positions_history.append(position)
            
            # Log trade if occurred
            if info.get("trade_executed"):
                self.trades_log.append({
                    "step": step,
                    "date": data.index[min(step, len(data) - 1)] if hasattr(data.index, "__getitem__") else step,
                    "side": info.get("trade_side", "unknown"),
                    "shares": info.get("trade_shares", 0),
                    "price": info.get("trade_price", 0),
                    "amount": info.get("trade_amount", 0),
                    "pnl": info.get("trade_pnl"),
                    "portfolio_value": portfolio_value,
                })
            
            step += 1
        
        # Create trades DataFrame
        trades_df = pd.DataFrame(self.trades_log) if self.trades_log else pd.DataFrame()
        
        # Calculate metrics
        equity = np.array(self.equity_history)
        calculator = MetricsCalculator(equity, trades_df)
        metrics = calculator.calculate_all()
        
        # Determine dates
        start_date = str(data.index[0]) if hasattr(data, "index") and len(data) > 0 else ""
        end_date = str(data.index[-1]) if hasattr(data, "index") and len(data) > 0 else ""
        
        result = BacktestResult(
            equity_curve=equity,
            trades=trades_df,
            metrics=metrics,
            start_date=start_date,
            end_date=end_date,
            symbols=symbols or [],
            config=self.config.to_dict(),
            positions=np.array(self.positions_history),
        )
        
        logger.info(
            f"Backtest complete | Return: {metrics.total_return:.2%} | "
            f"Sharpe: {metrics.sharpe_ratio:.2f} | Trades: {metrics.total_trades}"
        )
        
        return result
    
    def run_walk_forward(
        self,
        data: pd.DataFrame,
        train_window: int,
        test_window: int,
        algorithm_class: Type[BaseAlgorithm],
        algorithm_kwargs: Optional[Dict[str, Any]] = None,
        train_timesteps: int = 50_000,
        symbols: Optional[List[str]] = None,
    ) -> BacktestResult:
        """
        Walk-forward analysis with periodic retraining.
        
        Trains model on rolling window, then tests on next period.
        
        Args:
            data: Full market data
            train_window: Training window size (periods)
            test_window: Testing window size (periods)
            algorithm_class: SB3 algorithm class
            algorithm_kwargs: Algorithm hyperparameters
            train_timesteps: Timesteps per training window
            symbols: Symbol list
            
        Returns:
            Combined BacktestResult
        """
        if self.env_factory is None:
            raise ValueError("No env_factory provided")
        
        algorithm_kwargs = algorithm_kwargs or {}
        n_periods = len(data)
        
        # Track combined results
        all_equity = [self.config.initial_balance]
        all_trades = []
        
        # Walk forward
        i = train_window
        fold = 1
        
        while i + test_window <= n_periods:
            logger.info(f"Walk-forward fold {fold}: training on {i-train_window}:{i}, testing on {i}:{i+test_window}")
            
            # Split data
            train_data = data.iloc[i - train_window:i].copy()
            test_data = data.iloc[i:i + test_window].copy()
            
            # Train
            train_env = self.env_factory(train_data)
            model = algorithm_class(
                policy="MlpPolicy",
                env=train_env,
                verbose=0,
                **algorithm_kwargs,
            )
            model.learn(total_timesteps=train_timesteps)
            
            # Test
            self.model = model
            test_env = self.env_factory(test_data)
            
            obs, _ = test_env.reset()
            done = False
            last_equity = all_equity[-1]
            
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, _, terminated, truncated, info = test_env.step(action)
                done = terminated or truncated
                
                # Scale equity relative to last fold end
                env_equity = info.get("portfolio_value", self.config.initial_balance)
                scaled = last_equity * (env_equity / self.config.initial_balance)
                all_equity.append(scaled)
                
                if info.get("trade_executed"):
                    all_trades.append({
                        "fold": fold,
                        "side": info.get("trade_side"),
                        "price": info.get("trade_price"),
                        "amount": info.get("trade_amount"),
                        "pnl": info.get("trade_pnl"),
                    })
            
            train_env.close()
            test_env.close()
            
            i += test_window
            fold += 1
        
        # Calculate final metrics
        equity = np.array(all_equity)
        trades_df = pd.DataFrame(all_trades)
        calculator = MetricsCalculator(equity, trades_df)
        metrics = calculator.calculate_all()
        
        result = BacktestResult(
            equity_curve=equity,
            trades=trades_df,
            metrics=metrics,
            start_date=str(data.index[train_window]) if hasattr(data, "index") else "",
            end_date=str(data.index[-1]) if hasattr(data, "index") else "",
            symbols=symbols or [],
            config=self.config.to_dict(),
        )
        
        logger.info(
            f"Walk-forward complete | {fold-1} folds | "
            f"Return: {metrics.total_return:.2%} | Sharpe: {metrics.sharpe_ratio:.2f}"
        )
        
        return result
