"""
Data preprocessing for trading.

Handles:
    - Missing value handling
    - Normalization
    - Train/test splitting
    - Feature engineering
"""
import logging
from typing import Tuple, Optional

import numpy as np
import pandas as pd

from ..utils.validators import validate_dataframe

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    Preprocess market data for RL training.

    Handles missing values, normalization, and splitting.

    Example:
        >>> preprocessor = DataPreprocessor()
        >>> train_df, test_df = preprocessor.split(df, test_ratio=0.2)
    """

    def __init__(
        self,
        fill_method: str = "ffill",
        normalize: bool = True,
    ):
        """
        Initialize preprocessor.

        Args:
            fill_method: Method to fill missing values (ffill, bfill, mean)
            normalize: Whether to normalize data
        """
        self.fill_method = fill_method
        self.normalize = normalize
        self._normalization_params: dict = {}

    def process(
        self,
        df: pd.DataFrame,
        fit: bool = True,
    ) -> pd.DataFrame:
        """
        Process DataFrame.

        Args:
            df: Input DataFrame
            fit: Whether to fit normalization params (True for train, False for test)

        Returns:
            Processed DataFrame
        """
        df = validate_dataframe(df, required_columns=["close"])
        df = df.copy()

        # Handle missing values
        df = self._handle_missing(df)

        # Normalize if requested
        if self.normalize:
            df = self._normalize(df, fit=fit)

        return df

    def _handle_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values."""
        # Count missing before
        missing_before = df.isnull().sum().sum()

        if self.fill_method == "ffill":
            df = df.ffill().bfill()
        elif self.fill_method == "bfill":
            df = df.bfill().ffill()
        elif self.fill_method == "mean":
            df = df.fillna(df.mean())
        else:
            raise ValueError(f"Unknown fill method: {self.fill_method}")

        # Count missing after
        missing_after = df.isnull().sum().sum()

        if missing_before > 0:
            logger.info(f"Filled {missing_before - missing_after} missing values")

        # Drop any remaining NaN rows (at edges)
        df = df.dropna()

        return df

    def _normalize(
        self,
        df: pd.DataFrame,
        fit: bool = True,
    ) -> pd.DataFrame:
        """Normalize numeric columns.

        Raises:
            RuntimeError: If ``fit=False`` is used before the preprocessor
                has been fitted (BUG-09 fix).
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_cols = [c for c in numeric_cols if c != "date"]

        if fit:
            # Calculate and store normalization params
            self._normalization_params = {}
            for col in numeric_cols:
                self._normalization_params[col] = {
                    "mean": df[col].mean(),
                    "std": df[col].std(),
                }
        else:
            if not self._normalization_params:
                raise RuntimeError(
                    "DataPreprocessor has not been fitted yet. "
                    "Call process(df, fit=True) on training data first."
                )

        # Apply normalization
        for col in numeric_cols:
            if col in self._normalization_params:
                params = self._normalization_params[col]
                std = params["std"] if params["std"] > 0 else 1.0
                df[col] = (df[col] - params["mean"]) / std

        return df

    def split(
        self,
        df: pd.DataFrame,
        test_ratio: float = 0.2,
        validation_ratio: float = 0.0,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
        """
        Split data into train/test (and optionally validation) sets.

        Args:
            df: Input DataFrame
            test_ratio: Fraction for test set
            validation_ratio: Fraction for validation set

        Returns:
            Tuple of (train_df, test_df, val_df or None)

        Note:
            Uses time-based split (no shuffling) to preserve temporal order.
        """
        df = validate_dataframe(df)
        n = len(df)

        # Calculate split indices
        test_size = int(n * test_ratio)
        val_size = int(n * validation_ratio)
        train_size = n - test_size - val_size

        if train_size < 1:
            raise ValueError("Not enough data for train set")

        # Split
        train_df = df.iloc[:train_size].copy()
        test_df = df.iloc[train_size : train_size + test_size].copy()

        val_df = None
        if validation_ratio > 0:
            val_df = df.iloc[train_size + test_size :].copy()

        logger.info(
            f"Split data: train={len(train_df)}, test={len(test_df)}, "
            f"val={len(val_df) if val_df is not None else 0}"
        )

        return train_df, test_df, val_df

    def inverse_normalize(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Inverse normalization to get original values.

        Args:
            df: Normalized DataFrame

        Returns:
            DataFrame with original scale
        """
        df = df.copy()

        for col, params in self._normalization_params.items():
            if col in df.columns:
                std = params["std"] if params["std"] > 0 else 1.0
                df[col] = df[col] * std + params["mean"]

        return df


def preprocess_data(
    df: pd.DataFrame,
    test_ratio: float = 0.2,
    normalize: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convenience function to preprocess data.

    Args:
        df: Input DataFrame
        test_ratio: Fraction for test set
        normalize: Whether to normalize

    Returns:
        Tuple of (train_df, test_df)

    Example:
        >>> train_df, test_df = preprocess_data(df, test_ratio=0.2)
    """
    preprocessor = DataPreprocessor(normalize=normalize)

    # Split first (before normalizing)
    train_df, test_df, _ = preprocessor.split(df, test_ratio=test_ratio)

    # Process train (fit normalization)
    train_df = preprocessor.process(train_df, fit=True)

    # Process test (use train normalization params)
    test_df = preprocessor.process(test_df, fit=False)

    return train_df, test_df
