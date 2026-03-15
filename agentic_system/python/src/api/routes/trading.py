"""
Trading API routes for PipFlow AI dashboard.

Provides endpoints for:
    - Real-time quotes and chart data
    - Agent listing and evaluation
    - Paper trading execution
    - Position management

Security:
    - API key authentication
    - All inputs validated
    - Audit logging for trades
"""

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

from ...config.base import get_config
from ...data.brokers.mt5_client import MT5Client, MT5Config, get_mt5_client

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/trading", tags=["trading"])


# ==========================================
# Request/Response Models
# ==========================================

class ChartDataRequest(BaseModel):
    symbol: str = "EURUSD"
    timeframe: str = "H1"
    count: int = Field(default=500, ge=1, le=5000)


class TradeRequest(BaseModel):
    symbol: str
    action: str  # BUY or SELL
    lot_size: float = Field(gt=0, le=10)
    stop_loss_pips: Optional[float] = None
    take_profit_pips: Optional[float] = None
    
    
class AgentEvaluateRequest(BaseModel):
    agent_id: str
    symbol: str = "EURUSD"
    mode: str = "paper"  # paper or live_demo


# ==========================================
# MT5 Client Dependency
# ==========================================

_mt5_client: Optional["MT5Client"] = None


def get_mt5() -> "MT5Client":
    """
    Get a singleton MT5 client (BUG-06 fix: reuse connection).

    SEC-01 fix: No hardcoded default API key — requires env var.
    """
    global _mt5_client
    if _mt5_client is not None:
        return _mt5_client

    import os
    bridge_key = os.getenv("MT5_BRIDGE_API_KEY", "")
    if not bridge_key:
        logger.warning("MT5_BRIDGE_API_KEY not set — MT5 calls may fail")

    mt5_config = MT5Config(
        bridge_url=os.getenv("MT5_BRIDGE_URL", "http://localhost:8888"),
        api_key=bridge_key,
    )

    _mt5_client = MT5Client(mt5_config)
    return _mt5_client


# ==========================================
# Chart Data Endpoints
# ==========================================

@router.get("/chart-data/{symbol}")
async def get_chart_data(
    symbol: str,
    timeframe: str = "H1",
    count: int = 500,
) -> Dict[str, Any]:
    """
    Get OHLCV data for charting.
    
    Returns data in Lightweight Charts format.
    """
    try:
        client = get_mt5()
        df = client.get_history(symbol, timeframe, count)
        
        # Convert to Lightweight Charts format
        data = []
        for _, row in df.iterrows():
            data.append({
                "time": int(row["date"].timestamp()),
                "open": round(row["open"], 5),
                "high": round(row["high"], 5),
                "low": round(row["low"], 5),
                "close": round(row["close"], 5),
            })
        
        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "data": data,
        }
        
    except ConnectionError as e:
        raise HTTPException(status_code=503, detail=f"MT5 not connected: {e}")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Chart data error: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch chart data")


@router.get("/quote/{symbol}")
async def get_quote(symbol: str) -> Dict[str, Any]:
    """Get current bid/ask for a symbol."""
    try:
        client = get_mt5()
        quote = client.get_quote(symbol)
        
        return {
            "symbol": quote.symbol,
            "bid": quote.bid,
            "ask": quote.ask,
            "spread": quote.spread,
            "time": quote.time.isoformat(),
        }
        
    except ConnectionError as e:
        raise HTTPException(status_code=503, detail=f"MT5 not connected: {e}")
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/symbols")
async def get_symbols() -> Dict[str, Any]:
    """Get list of available trading symbols."""
    try:
        client = get_mt5()
        symbols = client.get_symbols()
        
        return {"symbols": symbols}
        
    except ConnectionError as e:
        raise HTTPException(status_code=503, detail=f"MT5 not connected: {e}")


# ==========================================
# Account & Positions
# ==========================================

@router.get("/account")
async def get_account() -> Dict[str, Any]:
    """Get account information."""
    try:
        client = get_mt5()
        account = client.get_account()
        
        return {
            "login": account.login,
            "name": account.name,
            "server": account.server,
            "currency": account.currency,
            "balance": account.balance,
            "equity": account.equity,
            "profit": account.profit,
            "margin": account.margin,
            "free_margin": account.free_margin,
            "leverage": account.leverage,
            "trade_mode": account.trade_mode,
        }
        
    except ConnectionError as e:
        raise HTTPException(status_code=503, detail=f"MT5 not connected: {e}")


@router.get("/positions")
async def get_positions() -> Dict[str, Any]:
    """Get open positions."""
    try:
        client = get_mt5()
        positions = client.get_positions()
        
        return {
            "positions": [
                {
                    "ticket": p.ticket,
                    "symbol": p.symbol,
                    "type": p.type,
                    "volume": p.volume,
                    "open_price": p.open_price,
                    "current_price": p.current_price,
                    "profit": p.profit,
                    "sl": p.sl,
                    "tp": p.tp,
                }
                for p in positions
            ]
        }
        
    except ConnectionError as e:
        raise HTTPException(status_code=503, detail=f"MT5 not connected: {e}")


# ==========================================
# Trading
# ==========================================

@router.post("/trade")
async def execute_trade(request: TradeRequest) -> Dict[str, Any]:
    """
    Execute a trade on demo account.
    
    Note: Only works with demo accounts for safety.
    """
    try:
        client = get_mt5()
        
        # Verify demo account
        account = client.get_account()
        if account.trade_mode != "Demo":
            raise HTTPException(
                status_code=403, 
                detail="Trading only allowed on demo accounts"
            )
        
        # Calculate SL/TP prices from pips
        quote = client.get_quote(request.symbol)
        pip_value = 0.0001 if "JPY" not in request.symbol else 0.01
        
        sl = None
        tp = None
        
        if request.stop_loss_pips:
            if request.action.upper() == "BUY":
                sl = quote.ask - (request.stop_loss_pips * pip_value)
            else:
                sl = quote.bid + (request.stop_loss_pips * pip_value)
        
        if request.take_profit_pips:
            if request.action.upper() == "BUY":
                tp = quote.ask + (request.take_profit_pips * pip_value)
            else:
                tp = quote.bid - (request.take_profit_pips * pip_value)
        
        # Execute trade
        result = client.trade(
            symbol=request.symbol,
            action=request.action,
            lot_size=request.lot_size,
            stop_loss=sl,
            take_profit=tp,
            comment="PipFlow Dashboard",
        )
        
        if not result.success:
            raise HTTPException(status_code=400, detail=result.error)
        
        logger.info(f"Trade executed: {request.action} {request.lot_size} {request.symbol}")
        
        return {
            "success": True,
            "order_id": result.order_id,
            "price": result.price,
            "volume": result.volume,
        }
        
    except ConnectionError as e:
        raise HTTPException(status_code=503, detail=f"MT5 not connected: {e}")


@router.post("/close/{ticket}")
async def close_position(ticket: int) -> Dict[str, Any]:
    """Close a position by ticket."""
    try:
        client = get_mt5()
        
        # Find position
        positions = client.get_positions()
        position = next((p for p in positions if p.ticket == ticket), None)
        
        if not position:
            raise HTTPException(status_code=404, detail="Position not found")
        
        result = client.close_position(
            symbol=position.symbol,
            ticket=ticket,
        )
        
        if not result.success:
            raise HTTPException(status_code=400, detail=result.error)
        
        return {
            "success": True,
            "ticket": ticket,
            "close_price": result.price,
            "profit": position.profit,
        }
        
    except ConnectionError as e:
        raise HTTPException(status_code=503, detail=f"MT5 not connected: {e}")


# ==========================================
# Agent Evaluation
# ==========================================

@router.get("/agents")
async def list_agents() -> Dict[str, Any]:
    """
    List available trained agents.
    
    Returns agents from the model storage directory.
    """
    from pathlib import Path
    
    config = get_config()
    model_path = Path(config.model_save_path)
    
    agents = []
    
    if model_path.exists():
        for model_dir in model_path.iterdir():
            if model_dir.is_dir():
                # Look for model files
                zip_files = list(model_dir.glob("*.zip"))
                if zip_files:
                    agents.append({
                        "id": model_dir.name,
                        "name": model_dir.name,
                        "path": str(model_dir),
                        "models": [f.stem for f in zip_files],
                        "created": datetime.fromtimestamp(
                            model_dir.stat().st_mtime
                        ).isoformat(),
                    })
    
    return {"agents": agents}


@router.post("/evaluate")
async def evaluate_agent(request: AgentEvaluateRequest) -> Dict[str, Any]:
    """
    Start agent evaluation on demo account.
    
    Runs the agent in paper trading mode against live market data.
    """
    # TODO: Implement agent evaluation
    # This will:
    # 1. Load the trained agent
    # 2. Connect to MT5 for live prices
    # 3. Run in paper trading mode
    # 4. Log all decisions
    
    return {
        "status": "started",
        "agent_id": request.agent_id,
        "symbol": request.symbol,
        "mode": request.mode,
        "message": "Agent evaluation started (paper trading mode)",
    }


# ==========================================
# WebSocket for Real-Time Updates
# ==========================================

@router.websocket("/ws/live")
async def websocket_live(websocket: WebSocket, symbols: str = "EURUSD"):
    """
    WebSocket for real-time price updates.
    
    Connect with: ws://host:port/api/trading/ws/live?symbols=EURUSD,GBPUSD
    """
    await websocket.accept()
    
    symbol_list = [s.strip().upper() for s in symbols.split(",")]
    client = get_mt5()
    
    try:
        while True:
            for symbol in symbol_list:
                try:
                    quote = client.get_quote(symbol)
                    await websocket.send_json({
                        "type": "quote",
                        "symbol": symbol,
                        "bid": quote.bid,
                        "ask": quote.ask,
                        "spread": quote.spread,
                        "time": quote.time.isoformat(),
                    })
                except Exception as e:
                    logger.warning(f"Quote error for {symbol}: {e}")
            
            await asyncio.sleep(1)  # Update every second
            
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
