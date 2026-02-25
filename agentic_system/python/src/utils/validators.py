"""
Input validation utilities.

Security:
    - All inputs validated before processing
    - Prevents invalid/malicious data from entering system
    - Clear error messages for debugging
"""
from typing import List, Optional, Any
import pandas as pd
import numpy as np


def validate_positive(
    value: float,
    name: str,
    allow_zero: bool = False,
) -> float:
    """
    Validate that a value is positive.

    Args:
        value: Value to validate
        name: Name for error message
        allow_zero: Whether to allow zero

    Returns:
        The validated value

    Raises:
        ValueError: If value is not positive

    Example:
        >>> validate_positive(100.0, "balance")
        100.0
        >>> validate_positive(-1, "balance")
        ValueError: balance must be positive, got -1
    """
    if allow_zero:
        if value < 0:
            raise ValueError(f"{name} must be non-negative, got {value}")
    else:
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}")
    return value


def validate_range(
    value: float,
    name: str,
    min_val: Optional[float] = None,
    max_val: Optional[float] = None,
    inclusive: bool = True,
) -> float:
    """
    Validate that a value is within a range.

    Args:
        value: Value to validate
        name: Name for error message
        min_val: Minimum allowed value (inclusive)
        max_val: Maximum allowed value (inclusive)
        inclusive: Whether bounds are inclusive

    Returns:
        The validated value

    Raises:
        ValueError: If value is out of range

    Example:
        >>> validate_range(0.5, "risk", min_val=0, max_val=1)
        0.5
    """
    if min_val is not None:
        if inclusive and value < min_val:
            raise ValueError(f"{name} must be >= {min_val}, got {value}")
        elif not inclusive and value <= min_val:
            raise ValueError(f"{name} must be > {min_val}, got {value}")

    if max_val is not None:
        if inclusive and value > max_val:
            raise ValueError(f"{name} must be <= {max_val}, got {value}")
        elif not inclusive and value >= max_val:
            raise ValueError(f"{name} must be < {max_val}, got {value}")

    return value


def validate_dataframe(
    df: pd.DataFrame,
    required_columns: Optional[List[str]] = None,
    min_rows: int = 1,
    name: str = "DataFrame",
) -> pd.DataFrame:
    """
    Validate a pandas DataFrame.

    Args:
        df: DataFrame to validate
        required_columns: List of required column names
        min_rows: Minimum number of rows required
        name: Name for error message

    Returns:
        The validated DataFrame

    Raises:
        ValueError: If DataFrame is invalid
        TypeError: If input is not a DataFrame

    Example:
        >>> df = pd.DataFrame({"close": [1, 2, 3]})
        >>> validate_dataframe(df, required_columns=["close"])
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"{name} must be a pandas DataFrame, got {type(df)}")

    if df.empty:
        raise ValueError(f"{name} cannot be empty")

    if len(df) < min_rows:
        raise ValueError(f"{name} must have at least {min_rows} rows, got {len(df)}")

    if required_columns:
        missing = set(required_columns) - set(df.columns)
        if missing:
            raise ValueError(f"{name} missing required columns: {missing}")

    return df


def validate_array(
    arr: np.ndarray,
    min_length: int = 1,
    name: str = "array",
) -> np.ndarray:
    """
    Validate a numpy array.

    Args:
        arr: Array to validate
        min_length: Minimum length required
        name: Name for error message

    Returns:
        The validated array

    Raises:
        ValueError: If array is invalid
        TypeError: If input is not an ndarray
    """
    if not isinstance(arr, np.ndarray):
        raise TypeError(f"{name} must be a numpy array, got {type(arr)}")

    if len(arr) < min_length:
        raise ValueError(f"{name} must have at least {min_length} elements, got {len(arr)}")

    if np.any(np.isnan(arr)):
        raise ValueError(f"{name} contains NaN values")

    if np.any(np.isinf(arr)):
        raise ValueError(f"{name} contains infinite values")

    return arr


def validate_symbol(symbol: str) -> str:
    """
    Validate a trading symbol.

    Args:
        symbol: Symbol to validate

    Returns:
        Cleaned symbol string

    Raises:
        ValueError: If symbol is invalid
    """
    if not symbol or not isinstance(symbol, str):
        raise ValueError(f"Symbol must be a non-empty string, got {symbol}")

    # Clean and uppercase
    symbol = symbol.strip().upper()

    if len(symbol) < 2:
        raise ValueError(f"Symbol too short: {symbol}")

    if len(symbol) > 20:
        raise ValueError(f"Symbol too long: {symbol}")

    # Check for invalid characters
    if not symbol.replace("-", "").replace("=", "").replace(".", "").isalnum():
        raise ValueError(f"Symbol contains invalid characters: {symbol}")

    return symbol


def validate_action(
    action: Any,
    min_val: float = -1.0,
    max_val: float = 1.0,
) -> np.ndarray:
    """
    Validate and normalize a trading action.

    Args:
        action: Action value(s), can be scalar or array
        min_val: Minimum allowed value
        max_val: Maximum allowed value

    Returns:
        Validated and clipped action as numpy array

    Example:
        >>> validate_action(1.5)  # Gets clipped to 1.0
        array([1.0])
    """
    action = np.atleast_1d(np.asarray(action, dtype=np.float32))
    action = np.clip(action, min_val, max_val)
    return action
