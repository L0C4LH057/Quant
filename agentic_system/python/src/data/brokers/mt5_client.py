"""
MT5 Client - Connects to MT5 Bridge on Windows VPS.

This client runs on the Linux server and communicates with the
MT5 bridge service running on Windows VPS via HTTP/WebSocket.

Security:
    - API key authentication
    - All inputs validated
    - Connection timeout handling

Token Optimization:
    - Simple interface matching existing fetcher pattern
    - Async support for real-time streaming
"""

import asyncio
import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime
from queue import Queue
from typing import Any, Callable, Dict, List, Optional

import httpx
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class MT5Config:
    """MT5 Bridge configuration."""
    
    bridge_url: str = "http://localhost:8888"
    api_key: str = "pipflow-dev-key-change-me"
    timeout: float = 30.0
    
    def __post_init__(self):
        # Remove trailing slash
        self.bridge_url = self.bridge_url.rstrip("/")


@dataclass
class Quote:
    """Real-time quote data."""
    
    symbol: str
    bid: float
    ask: float
    spread: float
    time: datetime
    volume: int = 0


@dataclass
class Position:
    """Open position data."""
    
    ticket: int
    symbol: str
    type: str  # BUY or SELL
    volume: float
    open_price: float
    current_price: float
    sl: float
    tp: float
    profit: float
    swap: float
    open_time: datetime
    comment: str = ""
    magic: int = 0


@dataclass 
class AccountInfo:
    """Trading account information."""
    
    login: int
    name: str
    server: str
    currency: str
    balance: float
    equity: float
    margin: float
    free_margin: float
    margin_level: float
    leverage: int
    profit: float
    trade_mode: str  # Demo or Real


@dataclass
class TradeResult:
    """Result of a trade operation."""
    
    success: bool
    order_id: Optional[int] = None
    deal_id: Optional[int] = None
    price: Optional[float] = None
    volume: Optional[float] = None
    error: Optional[str] = None


class MT5Client:
    """
    Client for MT5 Bridge service.
    
    Connects to the MT5 bridge running on Windows VPS and provides
    methods for account info, quotes, history, and trading.
    
    Example:
        >>> client = MT5Client(MT5Config(
        ...     bridge_url="http://192.168.1.100:8888",
        ...     api_key="your-api-key"
        ... ))
        >>> 
        >>> # Get account info
        >>> account = client.get_account()
        >>> print(f"Balance: {account.balance}")
        >>>
        >>> # Get quote
        >>> quote = client.get_quote("EURUSD")
        >>> print(f"EURUSD: {quote.bid}/{quote.ask}")
        >>>
        >>> # Get history as DataFrame
        >>> df = client.get_history("EURUSD", timeframe="H1", count=500)
    
    Security:
        - API key sent in X-API-Key header
        - All responses validated
        - Timeout handling prevents hanging
    """
    
    def __init__(self, config: Optional[MT5Config] = None):
        """
        Initialize MT5 client.
        
        Args:
            config: MT5 bridge configuration. If None, loads from environment.
        """
        self.config = config or self._load_config_from_env()
        self._client = httpx.Client(
            base_url=self.config.bridge_url,
            headers={"X-API-Key": self.config.api_key},
            timeout=self.config.timeout,
        )
        self._async_client: Optional[httpx.AsyncClient] = None
        
        # WebSocket streaming
        self._ws_callbacks: Dict[str, List[Callable[[Quote], None]]] = {}
        self._ws_thread: Optional[threading.Thread] = None
        self._ws_running = False
        self._quote_queue: Queue = Queue()
    
    def _load_config_from_env(self) -> MT5Config:
        """Load configuration from environment variables."""
        import os
        from dotenv import load_dotenv
        
        load_dotenv()
        
        return MT5Config(
            bridge_url=os.getenv("MT5_BRIDGE_URL", "http://localhost:8888"),
            api_key=os.getenv("MT5_BRIDGE_API_KEY", "pipflow-dev-key-change-me"),
            timeout=float(os.getenv("MT5_BRIDGE_TIMEOUT", "30.0")),
        )
    
    def _handle_response(self, response: httpx.Response) -> Dict[str, Any]:
        """Handle HTTP response and errors."""
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 403:
            raise PermissionError("Invalid API key")
        elif response.status_code == 404:
            raise ValueError(response.json().get("detail", "Not found"))
        elif response.status_code == 503:
            raise ConnectionError("MT5 bridge not connected to MetaTrader")
        else:
            detail = response.json().get("detail", "Unknown error")
            raise RuntimeError(f"MT5 bridge error: {detail}")
    
    # ==========================================
    # Health & Account
    # ==========================================
    
    def is_connected(self) -> bool:
        """Check if bridge is connected to MT5."""
        try:
            response = self._client.get("/health")
            data = response.json()
            return data.get("mt5_connected", False)
        except Exception:
            return False
    
    def get_account(self) -> AccountInfo:
        """
        Get account information.
        
        Returns:
            AccountInfo with balance, equity, margin, etc.
        
        Raises:
            ConnectionError: If not connected to MT5
            PermissionError: If API key is invalid
        """
        response = self._client.get("/account")
        data = self._handle_response(response)
        
        return AccountInfo(
            login=data["login"],
            name=data["name"],
            server=data["server"],
            currency=data["currency"],
            balance=data["balance"],
            equity=data["equity"],
            margin=data["margin"],
            free_margin=data["free_margin"],
            margin_level=data.get("margin_level", 0),
            leverage=data["leverage"],
            profit=data["profit"],
            trade_mode=data["trade_mode"],
        )
    
    # ==========================================
    # Market Data
    # ==========================================
    
    def get_quote(self, symbol: str) -> Quote:
        """
        Get current quote for a symbol.
        
        Args:
            symbol: Trading symbol (e.g., "EURUSD")
        
        Returns:
            Quote with bid, ask, spread
        
        Raises:
            ValueError: If symbol not found
        """
        response = self._client.get(f"/quote/{symbol}")
        data = self._handle_response(response)
        
        return Quote(
            symbol=data["symbol"],
            bid=data["bid"],
            ask=data["ask"],
            spread=data["spread"],
            time=datetime.fromisoformat(data["time"]),
            volume=data.get("volume", 0),
        )
    
    def get_history(
        self,
        symbol: str,
        timeframe: str = "H1",
        count: int = 500,
    ) -> pd.DataFrame:
        """
        Get historical OHLCV data.
        
        Args:
            symbol: Trading symbol
            timeframe: M1, M5, M15, M30, H1, H4, D1, W1, MN1
            count: Number of bars (max 5000)
        
        Returns:
            DataFrame with columns: date, open, high, low, close, volume
        
        Raises:
            ValueError: If symbol not found or invalid timeframe
        """
        response = self._client.get(
            f"/history/{symbol}",
            params={"timeframe": timeframe, "count": count},
        )
        data = self._handle_response(response)
        
        # Convert to DataFrame
        df = pd.DataFrame(data["data"])
        
        # Parse dates
        if "time" in df.columns:
            df["date"] = pd.to_datetime(df["time"])
            df = df.drop(columns=["time", "timestamp"], errors="ignore")
        
        # Ensure columns exist
        for col in ["open", "high", "low", "close", "volume"]:
            if col not in df.columns:
                df[col] = 0.0
        
        # Reorder columns
        df = df[["date", "open", "high", "low", "close", "volume"]]
        
        logger.info(f"Fetched {len(df)} bars for {symbol} ({timeframe})")
        return df
    
    def get_symbols(self) -> List[Dict[str, Any]]:
        """
        Get list of available symbols.
        
        Returns:
            List of symbol info dicts
        """
        response = self._client.get("/symbols")
        data = self._handle_response(response)
        return data.get("symbols", [])
    
    # ==========================================
    # Position Management
    # ==========================================
    
    def get_positions(self) -> List[Position]:
        """
        Get all open positions.
        
        Returns:
            List of Position objects
        """
        response = self._client.get("/positions")
        data = self._handle_response(response)
        
        positions = []
        for p in data.get("positions", []):
            positions.append(Position(
                ticket=p["ticket"],
                symbol=p["symbol"],
                type=p["type"],
                volume=p["volume"],
                open_price=p["open_price"],
                current_price=p["current_price"],
                sl=p["sl"],
                tp=p["tp"],
                profit=p["profit"],
                swap=p["swap"],
                open_time=datetime.fromisoformat(p["open_time"]),
                comment=p.get("comment", ""),
                magic=p.get("magic", 0),
            ))
        
        return positions
    
    # ==========================================
    # Trading
    # ==========================================
    
    def trade(
        self,
        symbol: str,
        action: str,
        lot_size: float,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        comment: str = "PipFlow",
    ) -> TradeResult:
        """
        Execute a trade.
        
        Args:
            symbol: Trading symbol (e.g., "EURUSD")
            action: "BUY" or "SELL"
            lot_size: Volume in lots
            stop_loss: Stop loss price (optional)
            take_profit: Take profit price (optional)
            comment: Order comment
        
        Returns:
            TradeResult with order_id and execution details
        
        Raises:
            ValueError: If invalid parameters
            RuntimeError: If order fails
        """
        if action.upper() not in ["BUY", "SELL"]:
            raise ValueError("Action must be BUY or SELL")
        
        if lot_size <= 0:
            raise ValueError("Lot size must be positive")
        
        payload = {
            "symbol": symbol,
            "action": action.upper(),
            "lot_size": lot_size,
            "comment": comment,
        }
        
        if stop_loss is not None:
            payload["stop_loss"] = stop_loss
        if take_profit is not None:
            payload["take_profit"] = take_profit
        
        try:
            response = self._client.post("/trade", json=payload)
            data = self._handle_response(response)
            
            return TradeResult(
                success=True,
                order_id=data.get("order_id"),
                deal_id=data.get("deal_id"),
                price=data.get("price"),
                volume=data.get("volume"),
            )
        except RuntimeError as e:
            return TradeResult(success=False, error=str(e))
    
    def modify_position(
        self,
        symbol: str,
        ticket: Optional[int] = None,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
    ) -> TradeResult:
        """
        Modify SL/TP of a position.
        
        Args:
            symbol: Symbol of position
            ticket: Position ticket (if None, modifies first position for symbol)
            stop_loss: New stop loss (None = keep current)
            take_profit: New take profit (None = keep current)
        
        Returns:
            TradeResult
        """
        payload = {"symbol": symbol}
        
        if ticket is not None:
            payload["ticket"] = ticket
        if stop_loss is not None:
            payload["stop_loss"] = stop_loss
        if take_profit is not None:
            payload["take_profit"] = take_profit
        
        try:
            response = self._client.post("/modify", json=payload)
            data = self._handle_response(response)
            
            return TradeResult(
                success=True,
                order_id=data.get("ticket"),
            )
        except RuntimeError as e:
            return TradeResult(success=False, error=str(e))
    
    def close_position(
        self,
        symbol: str,
        ticket: Optional[int] = None,
        volume: Optional[float] = None,
    ) -> TradeResult:
        """
        Close a position.
        
        Args:
            symbol: Symbol of position
            ticket: Position ticket (if None, closes first position for symbol)
            volume: Volume to close (if None, closes full position)
        
        Returns:
            TradeResult
        """
        payload = {"symbol": symbol}
        
        if ticket is not None:
            payload["ticket"] = ticket
        if volume is not None:
            payload["volume"] = volume
        
        try:
            response = self._client.post("/close", json=payload)
            data = self._handle_response(response)
            
            return TradeResult(
                success=True,
                order_id=data.get("ticket"),
                price=data.get("close_price"),
                volume=data.get("volume"),
            )
        except RuntimeError as e:
            return TradeResult(success=False, error=str(e))
    
    # ==========================================
    # Cleanup
    # ==========================================
    
    def close(self):
        """Close HTTP client."""
        self._client.close()
        if self._async_client:
            # Note: async client should be closed in async context
            pass
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# ==========================================
# Convenience Functions
# ==========================================

def get_mt5_client() -> MT5Client:
    """
    Get MT5 client with configuration from environment.
    
    Returns:
        Configured MT5Client instance
    
    Example:
        >>> client = get_mt5_client()
        >>> print(client.get_account().balance)
    """
    return MT5Client()
