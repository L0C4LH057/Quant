"""
Integration test for Training and Backtesting systems.

Tests the complete pipeline:
    1. Create trading environment
    2. Train RL model with TrainingManager
    3. Run backtest on test data
    4. Generate performance report

Usage:
    python scripts/test_training_backtest.py
"""
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd
from stable_baselines3 import PPO

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def generate_sample_data(n_days: int = 500) -> pd.DataFrame:
    """Generate synthetic market data for testing."""
    np.random.seed(42)
    
    dates = pd.date_range("2023-01-01", periods=n_days, freq="D")
    
    # Generate price with trend and volatility
    returns = np.random.randn(n_days) * 0.02 + 0.0003  # Small upward drift
    price = 100 * np.cumprod(1 + returns)
    
    # Add OHLCV
    high = price * (1 + np.abs(np.random.randn(n_days)) * 0.01)
    low = price * (1 - np.abs(np.random.randn(n_days)) * 0.01)
    volume = np.random.randint(1000000, 5000000, n_days)
    
    df = pd.DataFrame({
        "date": dates,
        "open": price * (1 + np.random.randn(n_days) * 0.005),
        "high": high,
        "low": low,
        "close": price,
        "volume": volume,
    })
    
    df.set_index("date", inplace=True)
    return df


def test_training_manager():
    """Test TrainingManager with a short training run."""
    from environments.trading_env import TradingEnv
    from training import TrainingManager, TrainingConfig
    
    logger.info("=" * 60)
    logger.info("Testing TrainingManager")
    logger.info("=" * 60)
    
    # Generate data
    data = generate_sample_data(300)
    train_data = data.iloc[:250]
    
    # Create env factory
    def env_factory():
        return TradingEnv(
            df=train_data,
            initial_balance=100_000,
            window_size=20,
        )
    
    # Create training manager
    config = TrainingConfig(
        total_timesteps=5000,  # Short for testing
        checkpoint_freq=2000,
        eval_freq=1000,
        early_stopping_patience=5,
        tensorboard_log=None,  # Disable for test
    )
    
    manager = TrainingManager(
        env_factory=env_factory,
        algorithm_class=PPO,
        algorithm_kwargs={"learning_rate": 3e-4},
        config=config,
        experiment_name="test_run",
    )
    
    # Train
    result = manager.train()
    
    logger.info(f"Training completed:")
    logger.info(f"  Status: {result.status.value}")
    logger.info(f"  Timesteps: {result.timesteps_completed:,}")
    logger.info(f"  Best Reward: {result.best_reward:.2f}")
    logger.info(f"  Training Time: {result.training_time:.1f}s")
    logger.info(f"  Model Path: {result.model_path}")
    
    return result


def test_backtesting(model_path: str = None):
    """Test BacktestEngine with trained or random model."""
    from environments.trading_env import TradingEnv
    from backtesting import BacktestEngine, BacktestConfig, BacktestReporter
    
    logger.info("=" * 60)
    logger.info("Testing BacktestEngine")
    logger.info("=" * 60)
    
    # Generate test data
    data = generate_sample_data(500)
    test_data = data.iloc[300:]
    
    # Create env factory
    def env_factory(df):
        return TradingEnv(
            df=df,
            initial_balance=100_000,
            window_size=20,
        )
    
    # Create or load model
    if model_path:
        engine = BacktestEngine(
            model_path=model_path,
            algorithm_class=PPO,
            env_factory=env_factory,
            config=BacktestConfig(initial_balance=100_000),
        )
    else:
        # Train a quick model for testing
        train_data = data.iloc[:300]
        train_env = env_factory(train_data)
        
        model = PPO("MlpPolicy", train_env, verbose=0)
        model.learn(total_timesteps=2000)
        
        engine = BacktestEngine(
            model=model,
            env_factory=env_factory,
            config=BacktestConfig(initial_balance=100_000),
        )
        
        train_env.close()
    
    # Run backtest
    result = engine.run(test_data, symbols=["SYNTHETIC"])
    
    logger.info(f"Backtest Results:")
    logger.info(f"  Total Return: {result.metrics.total_return:.2%}")
    logger.info(f"  Sharpe Ratio: {result.metrics.sharpe_ratio:.2f}")
    logger.info(f"  Max Drawdown: {result.metrics.max_drawdown:.2%}")
    logger.info(f"  Win Rate: {result.metrics.win_rate:.1%}")
    logger.info(f"  Total Trades: {result.metrics.total_trades}")
    
    # Generate report
    output_dir = Path(__file__).parent.parent / "outputs"
    output_dir.mkdir(exist_ok=True)
    
    reporter = BacktestReporter(result, title="Test Backtest")
    reporter.save_html(output_dir / "test_backtest_report.html")
    
    logger.info(f"  Report saved to: {output_dir / 'test_backtest_report.html'}")
    
    reporter.close()
    
    return result


def test_metrics():
    """Test metrics calculations."""
    from backtesting.metrics import (
        sharpe_ratio,
        sortino_ratio,
        max_drawdown,
        total_return,
        MetricsCalculator,
    )
    
    logger.info("=" * 60)
    logger.info("Testing Metrics")
    logger.info("=" * 60)
    
    # Create sample equity curve
    np.random.seed(42)
    returns = np.random.randn(252) * 0.01 + 0.0005
    equity = 100_000 * np.cumprod(1 + returns)
    equity = np.insert(equity, 0, 100_000)
    
    # Calculate metrics
    calc = MetricsCalculator(equity)
    metrics = calc.calculate_all()
    
    logger.info(f"Metrics from simulated equity:")
    logger.info(f"  Total Return: {metrics.total_return:.2%}")
    logger.info(f"  Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
    logger.info(f"  Sortino Ratio: {metrics.sortino_ratio:.2f}")
    logger.info(f"  Max Drawdown: {metrics.max_drawdown:.2%}")
    logger.info(f"  Volatility: {metrics.volatility:.2%}")
    
    # Verify calculations
    assert metrics.total_return != 0, "Total return should not be zero"
    assert -10 < metrics.sharpe_ratio < 10, "Sharpe ratio out of reasonable range"
    assert 0 <= metrics.max_drawdown <= 1, "Max drawdown should be between 0 and 1"
    
    logger.info("✓ All metrics tests passed!")
    
    return metrics


def main():
    """Run all tests."""
    logger.info("Starting Training & Backtesting Integration Tests")
    logger.info("=" * 60)
    
    try:
        # Test metrics first (no dependencies)
        test_metrics()
        
        # Test training
        training_result = test_training_manager()
        
        # Test backtesting with trained model
        if training_result.model_path:
            test_backtesting(training_result.model_path)
        else:
            test_backtesting()
        
        logger.info("=" * 60)
        logger.info("✓ ALL TESTS PASSED!")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.exception(f"Test failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
