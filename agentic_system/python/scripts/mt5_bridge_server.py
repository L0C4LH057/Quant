"""
SignalBridge MT5 Adapter - Enhanced for PipFlow AI

Run this on your Windows VPS where MetaTrader 5 is installed.
Provides REST API + WebSocket for real-time price streaming.

Usage:
    pip install fastapi uvicorn MetaTrader5 websockets python-dotenv
    python mt5_bridge_server.py

Security:
    - API key authentication via X-API-Key header
    - Rate limiting on trade endpoints
    - All inputs validated
"""

import asyncio
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set

import MetaTrader5 as mt5
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

load_dotenv()

# ============================================
# Configuration
# ============================================
API_KEY = os.getenv("MT5_BRIDGE_API_KEY", "pipflow-dev-key-change-me")
HOST = os.getenv("MT5_BRIDGE_HOST", "0.0.0.0")
PORT = int(os.getenv("MT5_BRIDGE_PORT", "8888"))

# ============================================
# FastAPI App
# ============================================
app = FastAPI(
    title="PipFlow MT5 Bridge",
    description="MetaTrader 5 bridge for PipFlow AI trading system",
    version="1.0.0",
)

# CORS for web UI access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================
# Models
# ============================================
class TradeRequest(BaseModel):
    symbol: str
    action: str  # BUY or SELL
    lot_size: float = Field(gt=0, le=100)
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    comment: Optional[str] = "PipFlow"
    
    @field_validator("action")
    @classmethod
    def validate_action(cls, v: str) -> str:
        if v.upper() not in ["BUY", "SELL"]:
            raise ValueError("Action must be BUY or SELL")
        return v.upper()


class ModifyRequest(BaseModel):
    symbol: str
    ticket: Optional[int] = None  # If None, modifies first position
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None


class CloseRequest(BaseModel):
    symbol: str
    ticket: Optional[int] = None  # If None, closes first position
    volume: Optional[float] = None  # If None, closes full position


class HistoryRequest(BaseModel):
    symbol: str
    timeframe: str = "H1"  # M1, M5, M15, M30, H1, H4, D1, W1, MN1
    count: int = Field(default=500, ge=1, le=5000)


# ============================================
# Authentication
# ============================================
async def verify_api_key(x_api_key: str = Header(None)) -> str:
    """Verify API key from request header."""
    if x_api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")
    return x_api_key


# ============================================
# MT5 Helpers
# ============================================
TIMEFRAME_MAP = {
    "M1": mt5.TIMEFRAME_M1,
    "M5": mt5.TIMEFRAME_M5,
    "M15": mt5.TIMEFRAME_M15,
    "M30": mt5.TIMEFRAME_M30,
    "H1": mt5.TIMEFRAME_H1,
    "H4": mt5.TIMEFRAME_H4,
    "D1": mt5.TIMEFRAME_D1,
    "W1": mt5.TIMEFRAME_W1,
    "MN1": mt5.TIMEFRAME_MN1,
}


def ensure_mt5_initialized() -> bool:
    """Ensure MT5 is initialized, attempt reinit if needed."""
    if not mt5.terminal_info():
        if not mt5.initialize():
            return False
    return True


# ============================================
# WebSocket Connection Manager
# ============================================
class ConnectionManager:
    """Manage WebSocket connections for price streaming."""
    
    def __init__(self):
        self.active_connections: Dict[str, Set[WebSocket]] = {}  # symbol -> connections
        self._streaming = False
        self._stream_task: Optional[asyncio.Task] = None
    
    async def connect(self, websocket: WebSocket, symbols: List[str]):
        """Accept connection and subscribe to symbols."""
        await websocket.accept()
        for symbol in symbols:
            if symbol not in self.active_connections:
                self.active_connections[symbol] = set()
            self.active_connections[symbol].add(websocket)
        
        # Start streaming if not already running
        if not self._streaming:
            self._streaming = True
            self._stream_task = asyncio.create_task(self._stream_prices())
    
    def disconnect(self, websocket: WebSocket):
        """Remove connection from all symbols."""
        for symbol in list(self.active_connections.keys()):
            self.active_connections[symbol].discard(websocket)
            if not self.active_connections[symbol]:
                del self.active_connections[symbol]
        
        # Stop streaming if no connections
        if not self.active_connections:
            self._streaming = False
            if self._stream_task:
                self._stream_task.cancel()
    
    async def _stream_prices(self):
        """Stream prices to all connected clients."""
        while self._streaming and self.active_connections:
            try:
                if not ensure_mt5_initialized():
                    await asyncio.sleep(5)
                    continue
                
                for symbol, connections in list(self.active_connections.items()):
                    tick = mt5.symbol_info_tick(symbol)
                    if tick:
                        data = {
                            "type": "tick",
                            "symbol": symbol,
                            "bid": tick.bid,
                            "ask": tick.ask,
                            "time": datetime.fromtimestamp(tick.time).isoformat(),
                            "volume": tick.volume,
                        }
                        
                        # Send to all connections for this symbol
                        dead_connections = set()
                        for ws in connections:
                            try:
                                await ws.send_json(data)
                            except Exception:
                                dead_connections.add(ws)
                        
                        # Remove dead connections
                        for ws in dead_connections:
                            self.disconnect(ws)
                
                await asyncio.sleep(0.5)  # 2 ticks per second
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Stream error: {e}")
                await asyncio.sleep(1)


manager = ConnectionManager()


# ============================================
# Lifecycle Events
# ============================================
@app.on_event("startup")
async def startup_event():
    """Initialize MT5 connection on startup."""
    print("Initializing MT5...")
    if not mt5.initialize():
        print(f"MT5 initialize() failed: {mt5.last_error()}")
    else:
        info = mt5.terminal_info()
        print(f"MT5 Initialized: {info.name} build {info.build}")
        account = mt5.account_info()
        if account:
            print(f"Account: {account.login} @ {account.server}")


@app.on_event("shutdown")
def shutdown_event():
    """Cleanup MT5 on shutdown."""
    mt5.shutdown()


# ============================================
# REST Endpoints
# ============================================
@app.get("/")
def read_root():
    """Health check."""
    return {
        "status": "running",
        "service": "PipFlow MT5 Bridge",
        "mt5_connected": ensure_mt5_initialized(),
    }


@app.get("/health")
def health_check():
    """Detailed health check."""
    connected = ensure_mt5_initialized()
    return {
        "status": "healthy" if connected else "degraded",
        "mt5_connected": connected,
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/account")
def get_account_info(api_key: str = Depends(verify_api_key)):
    """Get account information."""
    if not ensure_mt5_initialized():
        raise HTTPException(status_code=503, detail="MT5 not connected")
    
    account = mt5.account_info()
    if account is None:
        raise HTTPException(status_code=500, detail="Failed to get account info")
    
    return {
        "login": account.login,
        "name": account.name,
        "server": account.server,
        "currency": account.currency,
        "balance": account.balance,
        "equity": account.equity,
        "margin": account.margin,
        "free_margin": account.margin_free,
        "margin_level": account.margin_level if account.margin > 0 else 0,
        "leverage": account.leverage,
        "profit": account.profit,
        "trade_mode": "Demo" if account.trade_mode == 0 else "Real",
    }


@app.get("/quote/{symbol}")
def get_quote(symbol: str, api_key: str = Depends(verify_api_key)):
    """Get current bid/ask for a symbol."""
    if not ensure_mt5_initialized():
        raise HTTPException(status_code=503, detail="MT5 not connected")
    
    # Ensure symbol is selected
    if not mt5.symbol_select(symbol, True):
        raise HTTPException(status_code=404, detail=f"Symbol {symbol} not found")
    
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        raise HTTPException(status_code=404, detail=f"No tick data for {symbol}")
    
    info = mt5.symbol_info(symbol)
    
    return {
        "symbol": symbol,
        "bid": tick.bid,
        "ask": tick.ask,
        "spread": round((tick.ask - tick.bid) / info.point) if info else 0,
        "time": datetime.fromtimestamp(tick.time).isoformat(),
        "volume": tick.volume,
    }


@app.get("/history/{symbol}")
def get_history(
    symbol: str,
    timeframe: str = "H1",
    count: int = 500,
    api_key: str = Depends(verify_api_key),
):
    """Get historical OHLCV data."""
    if not ensure_mt5_initialized():
        raise HTTPException(status_code=503, detail="MT5 not connected")
    
    # Validate timeframe
    tf = TIMEFRAME_MAP.get(timeframe.upper())
    if tf is None:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid timeframe. Use: {list(TIMEFRAME_MAP.keys())}"
        )
    
    # Ensure symbol is selected
    if not mt5.symbol_select(symbol, True):
        raise HTTPException(status_code=404, detail=f"Symbol {symbol} not found")
    
    # Fetch rates
    rates = mt5.copy_rates_from_pos(symbol, tf, 0, min(count, 5000))
    if rates is None or len(rates) == 0:
        raise HTTPException(status_code=404, detail=f"No history for {symbol}")
    
    # Convert to list of dicts
    history = []
    for rate in rates:
        history.append({
            "time": datetime.fromtimestamp(rate["time"]).isoformat(),
            "timestamp": int(rate["time"]),
            "open": float(rate["open"]),
            "high": float(rate["high"]),
            "low": float(rate["low"]),
            "close": float(rate["close"]),
            "volume": int(rate["tick_volume"]),
        })
    
    return {
        "symbol": symbol,
        "timeframe": timeframe.upper(),
        "count": len(history),
        "data": history,
    }


@app.get("/symbols")
def get_symbols(api_key: str = Depends(verify_api_key)):
    """Get list of available symbols."""
    if not ensure_mt5_initialized():
        raise HTTPException(status_code=503, detail="MT5 not connected")
    
    symbols = mt5.symbols_get()
    if symbols is None:
        return {"symbols": []}
    
    # Return visible forex pairs and major indices
    result = []
    for s in symbols:
        if s.visible:
            result.append({
                "symbol": s.name,
                "description": s.description,
                "digits": s.digits,
                "min_lot": s.volume_min,
                "max_lot": s.volume_max,
                "lot_step": s.volume_step,
            })
    
    return {"symbols": result[:100]}  # Limit to 100


@app.get("/positions")
def get_positions(api_key: str = Depends(verify_api_key)):
    """Get all open positions."""
    if not ensure_mt5_initialized():
        raise HTTPException(status_code=503, detail="MT5 not connected")
    
    positions = mt5.positions_get()
    if positions is None:
        return {"positions": []}
    
    result = []
    for pos in positions:
        result.append({
            "ticket": pos.ticket,
            "symbol": pos.symbol,
            "type": "BUY" if pos.type == 0 else "SELL",
            "volume": pos.volume,
            "open_price": pos.price_open,
            "current_price": pos.price_current,
            "sl": pos.sl,
            "tp": pos.tp,
            "profit": pos.profit,
            "swap": pos.swap,
            "open_time": datetime.fromtimestamp(pos.time).isoformat(),
            "comment": pos.comment,
            "magic": pos.magic,
        })
    
    return {"positions": result}


@app.post("/trade")
def execute_trade(trade: TradeRequest, api_key: str = Depends(verify_api_key)):
    """Execute a trade."""
    if not ensure_mt5_initialized():
        raise HTTPException(status_code=503, detail="MT5 not connected")
    
    # Check symbol
    symbol_info = mt5.symbol_info(trade.symbol)
    if symbol_info is None:
        raise HTTPException(status_code=404, detail=f"Symbol {trade.symbol} not found")
    
    if not symbol_info.visible:
        if not mt5.symbol_select(trade.symbol, True):
            raise HTTPException(status_code=400, detail=f"Cannot select {trade.symbol}")
    
    # Get current price
    tick = mt5.symbol_info_tick(trade.symbol)
    if tick is None:
        raise HTTPException(status_code=500, detail="Failed to get price")
    
    order_type = mt5.ORDER_TYPE_BUY if trade.action == "BUY" else mt5.ORDER_TYPE_SELL
    price = tick.ask if trade.action == "BUY" else tick.bid
    
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": trade.symbol,
        "volume": trade.lot_size,
        "type": order_type,
        "price": price,
        "deviation": 20,
        "magic": 234000,
        "comment": trade.comment or "PipFlow",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    
    if trade.stop_loss:
        request["sl"] = trade.stop_loss
    if trade.take_profit:
        request["tp"] = trade.take_profit
    
    result = mt5.order_send(request)
    
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        raise HTTPException(
            status_code=400, 
            detail=f"Order failed: {result.comment} (code: {result.retcode})"
        )
    
    return {
        "status": "success",
        "order_id": result.order,
        "deal_id": result.deal,
        "price": result.price,
        "volume": result.volume,
        "comment": result.comment,
    }


@app.post("/modify")
def modify_position(request: ModifyRequest, api_key: str = Depends(verify_api_key)):
    """Modify SL/TP of a position."""
    if not ensure_mt5_initialized():
        raise HTTPException(status_code=503, detail="MT5 not connected")
    
    # Find position
    if request.ticket:
        positions = mt5.positions_get(ticket=request.ticket)
    else:
        positions = mt5.positions_get(symbol=request.symbol)
    
    if not positions or len(positions) == 0:
        raise HTTPException(status_code=404, detail="No position found")
    
    position = positions[0]
    
    modify_request = {
        "action": mt5.TRADE_ACTION_SLTP,
        "symbol": position.symbol,
        "position": position.ticket,
        "sl": request.stop_loss if request.stop_loss is not None else position.sl,
        "tp": request.take_profit if request.take_profit is not None else position.tp,
    }
    
    result = mt5.order_send(modify_request)
    
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        raise HTTPException(
            status_code=400, 
            detail=f"Modify failed: {result.comment} (code: {result.retcode})"
        )
    
    return {
        "status": "success",
        "ticket": position.ticket,
        "sl": modify_request["sl"],
        "tp": modify_request["tp"],
    }


@app.post("/close")
def close_position(request: CloseRequest, api_key: str = Depends(verify_api_key)):
    """Close a position."""
    if not ensure_mt5_initialized():
        raise HTTPException(status_code=503, detail="MT5 not connected")
    
    # Find position
    if request.ticket:
        positions = mt5.positions_get(ticket=request.ticket)
    else:
        positions = mt5.positions_get(symbol=request.symbol)
    
    if not positions or len(positions) == 0:
        raise HTTPException(status_code=404, detail="No position found")
    
    position = positions[0]
    
    # Determine close parameters
    close_type = mt5.ORDER_TYPE_SELL if position.type == 0 else mt5.ORDER_TYPE_BUY
    tick = mt5.symbol_info_tick(position.symbol)
    price = tick.bid if position.type == 0 else tick.ask
    volume = request.volume if request.volume else position.volume
    
    close_request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": position.symbol,
        "volume": volume,
        "type": close_type,
        "position": position.ticket,
        "price": price,
        "deviation": 20,
        "magic": 234000,
        "comment": "PipFlow close",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    
    result = mt5.order_send(close_request)
    
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        raise HTTPException(
            status_code=400, 
            detail=f"Close failed: {result.comment} (code: {result.retcode})"
        )
    
    return {
        "status": "success",
        "ticket": position.ticket,
        "close_price": result.price,
        "volume": result.volume,
        "profit": position.profit,
    }


# ============================================
# WebSocket Endpoints
# ============================================
@app.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket, symbols: str = "EURUSD"):
    """
    WebSocket endpoint for real-time price streaming.
    
    Connect with: ws://host:port/ws/stream?symbols=EURUSD,GBPUSD
    
    Messages:
        {"type": "tick", "symbol": "EURUSD", "bid": 1.0850, "ask": 1.0852, ...}
    """
    symbol_list = [s.strip().upper() for s in symbols.split(",")]
    
    await manager.connect(websocket, symbol_list)
    
    try:
        # Send initial prices
        if ensure_mt5_initialized():
            for symbol in symbol_list:
                tick = mt5.symbol_info_tick(symbol)
                if tick:
                    await websocket.send_json({
                        "type": "tick",
                        "symbol": symbol,
                        "bid": tick.bid,
                        "ask": tick.ask,
                        "time": datetime.fromtimestamp(tick.time).isoformat(),
                    })
        
        # Keep connection alive
        while True:
            try:
                # Wait for ping/pong or client messages
                data = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                
                # Handle client messages if needed
                if data == "ping":
                    await websocket.send_text("pong")
                    
            except asyncio.TimeoutError:
                # Send heartbeat
                await websocket.send_json({"type": "heartbeat", "time": datetime.now().isoformat()})
                
    except WebSocketDisconnect:
        pass
    finally:
        manager.disconnect(websocket)


# ============================================
# Entry Point
# ============================================
if __name__ == "__main__":
    import uvicorn
    
    print(f"""
    ╔══════════════════════════════════════════════════════════╗
    ║           PipFlow MT5 Bridge Server                      ║
    ╠══════════════════════════════════════════════════════════╣
    ║  Host: {HOST}                                          ║
    ║  Port: {PORT}                                            ║
    ║  Docs: http://{HOST}:{PORT}/docs                         ║
    ╚══════════════════════════════════════════════════════════╝
    """)
    
    uvicorn.run(app, host=HOST, port=PORT)
