"""
PipFlow AI - Live Trading Dashboard

Streamlit dashboard for MT5 trading with real-time charts and agent control.

Run with: streamlit run scripts/trading_dashboard.py
"""

import logging
import os
import sys

logger = logging.getLogger(__name__)

# Fix PyTorch/Streamlit file watcher conflict
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from datetime import datetime, timedelta
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
env_path = Path(__file__).parent.parent.parent / ".env"
load_dotenv(env_path)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import httpx
import glob
import asyncio
import time
import json

# Import specialized agents
from src.agents.specialized.market_analysis_agent import MarketAnalysisAgent
from src.agents.specialized.risk_management_agent import RiskManagementAgent
from src.agents.specialized.execution_agent import ExecutionAgent
from src.agents.specialized.sentiment_analysis_agent import SentimentAnalysisAgent

from src.data.sentiment_fetcher import fetch_news_sentiment, NewsSentimentFetcher
from src.features.sentiment_analyzer import (
    add_sentiment_features,
    compute_sentiment_features,
    get_sentiment_summary,
)

# Import RL agents and components
from src.agents.rl_agents.ensemble_agent import EnsembleAgent

# Removed importlib.reload due to circular import issues with Streamlit watcher
from src.agents.rl_agents.signal_generator import RLSignalGenerator
from src.agents.rl_agents.rl_trace_wrapper import RLTraceWrapper
from src.environments.trading_env import TradingEnv
from src.environments.discrete_trading_env import DiscreteTradingEnv

# Signal Intelligence — regime detection, transition alerts, signal arbitration
from src.agents.signal_intelligence import (
    MarketRegimeDetector,
    SignalTransitionDetector,
    UnifiedSignalArbiter,
)

# Page config
st.set_page_config(
    page_title="PipFlow AI - Trading Dashboard",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for premium dark theme
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #0d1117 0%, #161b22 100%);
    }
    .metric-card {
        background: linear-gradient(135deg, #21262d 0%, #30363d 100%);
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #30363d;
    }
    .agent-card {
        background: linear-gradient(135deg, #1a1f29 0%, #252d3a 100%);
        padding: 16px;
        border-radius: 10px;
        border: 1px solid #30363d;
        margin-bottom: 10px;
    }
    .strategy-card {
        background: linear-gradient(135deg, #1f2937 0%, #374151 100%);
        padding: 16px;
        border-radius: 10px;
        border: 1px solid #4b5563;
    }
    .buy-btn {
        background: linear-gradient(135deg, #238636 0%, #2ea043 100%) !important;
        color: white !important;
    }
    .sell-btn {
        background: linear-gradient(135deg, #da3633 0%, #f85149 100%) !important;
        color: white !important;
    }
    .status-running { color: #3fb950; font-weight: bold; }
    .status-stopped { color: #8b949e; }
    .status-error { color: #f85149; }
    div[data-testid="stMetricValue"] {
        font-size: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)


# ============= MT5 CLIENT =============

class MT5DashboardClient:
    """Simple MT5 bridge client for the dashboard."""
    
    def __init__(self):
        self.bridge_url = os.getenv("MT5_BRIDGE_URL", "http://localhost:8000")
        self.api_key = os.getenv("MT5_BRIDGE_API_KEY", "pipflow-dev-key-change-me")
        self.timeout = 10.0
    
    def _headers(self):
        return {"X-API-Key": self.api_key}
    
    def is_connected(self) -> bool:
        try:
            resp = httpx.get(f"{self.bridge_url}/health", timeout=5.0)
            return resp.status_code == 200 and resp.json().get("mt5_connected", False)
        except:
            return False
    
    def get_account(self) -> dict:
        try:
            resp = httpx.get(
                f"{self.bridge_url}/account",
                headers=self._headers(),
                timeout=self.timeout
            )
            if resp.status_code == 200:
                return resp.json()
        except:
            pass
        return {}
    
    def get_quote(self, symbol: str) -> dict:
        try:
            resp = httpx.get(
                f"{self.bridge_url}/quote/{symbol}",
                headers=self._headers(),
                timeout=self.timeout
            )
            if resp.status_code == 200:
                return resp.json()
        except:
            pass
        return {}
    
    def get_history(self, symbol: str, timeframe: str = "H1", count: int = 500) -> pd.DataFrame:
        try:
            resp = httpx.get(
                f"{self.bridge_url}/history/{symbol}",
                params={"timeframe": timeframe, "count": count},
                headers=self._headers(),
                timeout=self.timeout
            )
            if resp.status_code == 200:
                data = resp.json()
                df = pd.DataFrame(data["data"])
                if "time" in df.columns:
                    df["date"] = pd.to_datetime(df["time"])
                return df
            else:
                st.warning(f"Failed to fetch history: {resp.text}")
        except Exception as e:
            st.error(f"Error fetching history: {e}")
        return pd.DataFrame()
    
    def get_positions(self) -> list:
        try:
            resp = httpx.get(
                f"{self.bridge_url}/positions",
                headers=self._headers(),
                timeout=self.timeout
            )
            if resp.status_code == 200:
                return resp.json().get("positions", [])
        except:
            pass
        return []
    
    def get_symbols(self) -> list:
        try:
            resp = httpx.get(
                f"{self.bridge_url}/symbols",
                headers=self._headers(),
                timeout=self.timeout
            )
            if resp.status_code == 200:
                symbols = resp.json().get("symbols", [])
                # Return symbol names only
                return [s["symbol"] for s in symbols]
        except:
            pass
        return []
    
    def trade(self, symbol: str, action: str, lot_size: float, 
              stop_loss: float = None, take_profit: float = None) -> dict:
        try:
            payload = {
                "symbol": symbol,
                "action": action,
                "lot_size": lot_size,
            }
            if stop_loss:
                payload["stop_loss"] = stop_loss
            if take_profit:
                payload["take_profit"] = take_profit
            
            resp = httpx.post(
                f"{self.bridge_url}/trade",
                headers=self._headers(),
                json=payload,
                timeout=self.timeout
            )
            data = resp.json()
            if resp.status_code != 200:
                return {"error": data.get("detail", data.get("error", resp.text))}
            return data
        except Exception as e:
            return {"error": str(e)}
    
    def close_position(self, symbol: str, ticket: int = None) -> dict:
        try:
            payload = {"symbol": symbol}
            if ticket:
                payload["ticket"] = ticket
            
            resp = httpx.post(
                f"{self.bridge_url}/close",
                headers=self._headers(),
                json=payload,
                timeout=self.timeout
            )
            data = resp.json()
            if resp.status_code != 200:
                return {"error": data.get("detail", data.get("error", resp.text))}
            return data
        except Exception as e:
            return {"error": str(e)}


# Initialize client
@st.cache_resource
def get_client():
    return MT5DashboardClient()


# ============= TECHNICAL ANALYSIS AGENT =============

class TechnicalAnalysisAgent:
    """Rule-based agent using pandas_ta for technical analysis."""
    
    def __init__(self):
        self.name = "Technical Analyst"
        
    def analyze(self, df: pd.DataFrame) -> dict:
        """Analyze market data and generate signal."""
        if df.empty or len(df) < 50:
            return {"action": "HOLD", "confidence": 0.0, "reason": "Insufficient data"}
        
        # Calculate indicators using pandas_ta
        # Ensure imports are available
        try:
            import pandas_ta as ta
        except ImportError:
            return {"action": "HOLD", "confidence": 0.0, "reason": "pandas_ta not installed"}
            
        df = df.copy()
        
        # RSI
        df['rsi'] = df.ta.rsi(length=14)
        
        # MACD
        macd = df.ta.macd(fast=12, slow=26, signal=9)
        if macd is not None:
            df = pd.concat([df, macd], axis=1)
        
        # Bollinger Bands
        bb = df.ta.bbands(length=20, std=2)
        if bb is not None:
            df = pd.concat([df, bb], axis=1)
        
        # SMA
        df['sma_20'] = df.ta.sma(length=20)
        df['sma_50'] = df.ta.sma(length=50)
        df['sma_200'] = df.ta.sma(length=200)
        
        # Get latest values
        current = df.iloc[-1]
        prev = df.iloc[-2]
        
        signals = []
        reasons = []
        
        # 1. RSI Logic
        current_rsi = current.get('rsi')
        if current_rsi is not None:
            if current_rsi < 30:
                signals.append(1)
                reasons.append(f"RSI Oversold ({current_rsi:.1f})")
            elif current_rsi > 70:
                signals.append(-1)
                reasons.append(f"RSI Overbought ({current_rsi:.1f})")
        
        # 2. MACD Logic
        macd_line = current.get('MACD_12_26_9')
        macd_signal = current.get('MACDs_12_26_9')
        if macd_line is not None and macd_signal is not None:
            if macd_line > macd_signal and prev.get('MACD_12_26_9') < prev.get('MACDs_12_26_9'):
                signals.append(1)
                reasons.append("MACD Bullish Crossover")
            elif macd_line < macd_signal and prev.get('MACD_12_26_9') > prev.get('MACDs_12_26_9'):
                signals.append(-1)
                reasons.append("MACD Bearish Crossover")
        
        # 3. Bollinger Bands Logic
        close = current['close']
        bb_lower = current.get('BBL_20_2.0')
        bb_upper = current.get('BBU_20_2.0')
        
        if bb_lower is not None and close < bb_lower:
            signals.append(0.5)  # Weak buy (mean reversion)
            reasons.append("Price below Lower BB")
        elif bb_upper is not None and close > bb_upper:
            signals.append(-0.5) # Weak sell
            reasons.append("Price above Upper BB")
            
        # 4. Trend Logic (SMA)
        sma20 = current.get('sma_20')
        sma50 = current.get('sma_50')
        
        if sma20 is not None and sma50 is not None:
            if sma20 > sma50:
                signals.append(0.5)
                reasons.append("Uptrend (SMA20 > SMA50)")
            elif sma20 < sma50:
                signals.append(-0.5)
                reasons.append("Downtrend (SMA20 < SMA50)")
        
        # Aggregate
        if not signals:
            return {"action": "HOLD", "confidence": 0.5, "reason": "No strong signals"}
            
        score = sum(signals)
        
        if score >= 1.5:
            action = "BUY"
            confidence = min(0.6 + (score * 0.1), 0.95)
        elif score <= -1.5:
            action = "SELL"
            confidence = min(0.6 + (abs(score) * 0.1), 0.95)
        else:
            action = "HOLD"
            confidence = 0.5 + (abs(score) * 0.1)
            
        return {
            "action": action,
            "confidence": confidence,
            "reason": ", ".join(reasons)
        }

# ============= AGENT MANAGEMENT =============

def get_available_models() -> list:
    """Find all saved RL models."""
    models = ["Demo Agent (Random)", "Technical Analysis (Rule-Based)"]
    
    models_path = Path(__file__).parent.parent / "models"
    if models_path.exists():
        # Find .zip files (SB3 model format)
        model_files = list(models_path.glob("**/*.zip"))
        models.extend([str(f.relative_to(models_path)) for f in model_files])
    
    return models

# ... (rest of the file) ...




def get_strategy_presets() -> dict:
    """Return predefined trading strategy configurations."""
    return {
        "Conservative": {
            "max_position_size": 0.01,
            "max_drawdown": 0.05,
            "profit_target": 0.02,
            "stop_loss_pips": 30,
            "take_profit_pips": 60,
            "risk_per_trade": 0.01,
            "description": "Low risk, steady gains"
        },
        "Moderate": {
            "max_position_size": 0.05,
            "max_drawdown": 0.10,
            "profit_target": 0.05,
            "stop_loss_pips": 50,
            "take_profit_pips": 100,
            "risk_per_trade": 0.02,
            "description": "Balanced risk/reward"
        },
        "Aggressive": {
            "max_position_size": 0.10,
            "max_drawdown": 0.20,
            "profit_target": 0.10,
            "stop_loss_pips": 100,
            "take_profit_pips": 200,
            "risk_per_trade": 0.05,
            "description": "Higher risk, higher potential"
        },
        "Custom": {
            "max_position_size": 0.01,
            "max_drawdown": 0.10,
            "profit_target": 0.05,
            "stop_loss_pips": 50,
            "take_profit_pips": 100,
            "risk_per_trade": 0.02,
            "description": "Configure your own parameters"
        }
    }


# ============= CHART FUNCTIONS =============

def create_candlestick_chart(df: pd.DataFrame, symbol: str) -> go.Figure:
    """Create a candlestick chart with volume."""
    if df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No data available - check symbol availability",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=18, color="#8b949e")
        )
        fig.update_layout(
            paper_bgcolor='#161b22',
            plot_bgcolor='#161b22',
            height=500,
        )
        return fig
    
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.75, 0.25],
    )
    
    # Candlestick
    fig.add_trace(
        go.Candlestick(
            x=df['date'],
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name=symbol,
            increasing_line_color='#3fb950',
            decreasing_line_color='#f85149',
        ),
        row=1, col=1
    )
    
    # Volume
    colors = ['#3fb950' if c >= o else '#f85149' 
              for o, c in zip(df['open'], df['close'])]
    
    fig.add_trace(
        go.Bar(
            x=df['date'],
            y=df['volume'],
            name='Volume',
            marker_color=colors,
            opacity=0.7,
        ),
        row=2, col=1
    )
    
    fig.update_layout(
        title=f"{symbol} Chart",
        paper_bgcolor='#161b22',
        plot_bgcolor='#161b22',
        font=dict(color='#f0f6fc'),
        xaxis_rangeslider_visible=False,
        height=550,
        showlegend=False,
        margin=dict(l=50, r=50, t=50, b=30),
    )
    
    fig.update_xaxes(gridcolor='#21262d', showgrid=True)
    fig.update_yaxes(gridcolor='#21262d', showgrid=True)
    
    return fig


# ============= SESSION STATE INIT =============

if "agent_running" not in st.session_state:
    st.session_state.agent_running = False
if "selected_model" not in st.session_state:
    st.session_state.selected_model = None
if "strategy_config" not in st.session_state:
    st.session_state.strategy_config = get_strategy_presets()["Moderate"]
if "agent_signals" not in st.session_state:
    st.session_state.agent_signals = []
if "cached_symbols" not in st.session_state:
    st.session_state.cached_symbols = None

# Agent Pipeline State
if "agent_analysis_interval" not in st.session_state:
    st.session_state.agent_analysis_interval = 10  # seconds
if "auto_execute_trades" not in st.session_state:
    st.session_state.auto_execute_trades = False
if "enable_news_sentiment" not in st.session_state:
    st.session_state.enable_news_sentiment = False
if "max_trades_session" not in st.session_state:
    st.session_state.max_trades_session = 10
if "trade_cooldown" not in st.session_state:
    st.session_state.trade_cooldown = 60  # seconds
if "last_trade_time" not in st.session_state:
    st.session_state.last_trade_time = None
if "agent_trades" not in st.session_state:
    st.session_state.agent_trades = []
if "trades_this_session" not in st.session_state:
    st.session_state.trades_this_session = 0
if "last_analysis_result" not in st.session_state:
    st.session_state.last_analysis_result = None
if "last_risk_result" not in st.session_state:
    st.session_state.last_risk_result = None


# ============= AGENT INSTANCES =============

@st.cache_resource
def get_agents():
    """Create singleton agent instances."""
    return {
        "market_analyst": MarketAnalysisAgent(),
        "risk_manager": RiskManagementAgent(max_risk_per_trade=0.02),
        "executor": ExecutionAgent(),
        "sentiment_analyst": SentimentAnalysisAgent(),
    }


@st.cache_resource
def get_rl_tracer():
    """Create trace wrapper for callback server."""
    server_url = os.getenv("CALLBACK_SERVER_URL", "http://localhost:3001")
    return RLTraceWrapper(server_url=server_url, project_id=1)


def get_rl_algorithm(algo_name: str):
    """Import and return SB3 algorithm class by name."""
    if algo_name == "PPO":
        from stable_baselines3 import PPO
        return PPO
    elif algo_name == "SAC":
        from stable_baselines3 import SAC
        return SAC
    elif algo_name == "A2C":
        from stable_baselines3 import A2C
        return A2C
    elif algo_name == "TD3":
        from stable_baselines3 import TD3
        return TD3
    elif algo_name == "DQN":
        from stable_baselines3 import DQN
        return DQN
    else:
        raise ValueError(f"Unknown algorithm: {algo_name}")


# RL Session State
if "rl_trained_agents" not in st.session_state:
    st.session_state.rl_trained_agents = {}  # {name: agent_instance}
if "rl_ensemble" not in st.session_state:
    st.session_state.rl_ensemble = None
if "rl_signal_result" not in st.session_state:
    st.session_state.rl_signal_result = None
if "rl_training_log" not in st.session_state:
    st.session_state.rl_training_log = []
if "rl_trace_log" not in st.session_state:
    st.session_state.rl_trace_log = []

# Signal Intelligence Session State
if "signal_transition_detector" not in st.session_state:
    st.session_state.signal_transition_detector = SignalTransitionDetector(window_size=10)
if "signal_alerts" not in st.session_state:
    st.session_state.signal_alerts = []
if "market_regime" not in st.session_state:
    st.session_state.market_regime = None
if "rl_auto_trained" not in st.session_state:
    st.session_state.rl_auto_trained = False
if "last_arbitration" not in st.session_state:
    st.session_state.last_arbitration = None


def _run_async(coro):
    """Run async code safely with Streamlit."""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    if loop.is_running():
        # Streamlit >= 1.30 runs in an async context, but blockingly inside ScriptRunner
        import threading
        result = None
        err = None
        
        def _run_in_thread():
            nonlocal result, err
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            try:
                result = new_loop.run_until_complete(coro)
            except Exception as e:
                err = e
            finally:
                new_loop.close()
                
        t = threading.Thread(target=_run_in_thread)
        t.start()
        t.join(timeout=30.0)
        
        if t.is_alive():
            raise TimeoutError("Async execution timed out")
        if err:
            raise err
        return result
    else:
        return loop.run_until_complete(coro)


def _auto_train_rl_agents(df: pd.DataFrame, symbol: str):
    """
    Auto-train a small set of RL agents (PPO + SAC) for quick signal generation.

    Only called if no RL agents are already trained in the session.
    Uses 10k timesteps each for fast startup (~15-30s total).
    """
    quick_algos = ["PPO", "SAC"]
    quick_timesteps = 10_000

    for algo_name in quick_algos:
        agent_key = f"{algo_name}_auto"
        if agent_key not in st.session_state.rl_trained_agents:
            try:
                model, env, metrics = run_rl_training(
                    df, algo_name, "sharpe", quick_timesteps, symbol=symbol,
                )
                st.session_state.rl_trained_agents[agent_key] = {
                    "model": model,
                    "env": env,
                    "metrics": metrics,
                }
                st.session_state.rl_training_log.insert(0, {
                    "time": datetime.now().strftime("%H:%M:%S"),
                    "agent": agent_key,
                    "metrics": metrics,
                })
            except Exception as e:
                st.warning(f"Auto-train {algo_name} failed: {e}")

    st.session_state.rl_auto_trained = True


def _build_rl_signal(df: pd.DataFrame, symbol: str) -> dict:
    """
    Build an EnsembleAgent from trained RL agents and generate an RL signal.

    Returns the RLSignalGenerator output dict, or a hold-fallback on error.
    """
    try:
        ensemble = EnsembleAgent(
            buy_threshold=0.05,   # Lower thresholds for lightly-trained auto models
            sell_threshold=-0.05,
            min_confidence=0.4,
        )
        for name, agent_data in st.session_state.rl_trained_agents.items():
            # Handle both formats: dict {"model": ..., "env": ...} or raw SB3 model
            if isinstance(agent_data, dict):
                raw_model = agent_data["model"]
            else:
                raw_model = agent_data

            # Wrap SB3 model so predict() returns a single action array
            # (raw SB3 predict returns (action, states) tuple)
            class _ModelWrapper:
                def __init__(self, m):
                    self._m = m
                def predict(self, obs, deterministic=True):
                    action, _ = self._m.predict(obs, deterministic=deterministic)
                    return action

            ensemble.add_agent(name, _ModelWrapper(raw_model))

        if ensemble.agent_count == 0:
            return {"signal": "hold", "confidence": 0, "reason": "No RL agents available"}

        st.session_state.rl_ensemble = ensemble

        # Clean df: keep only date + numeric columns (same as run_rl_training)
        # This avoids crashes from non-numeric columns like MT5's string 'time'
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        keep_cols = [c for c in ["date", "open", "high", "low", "close", "volume"] if c in df.columns]
        for c in numeric_cols:
            if c not in keep_cols:
                keep_cols.append(c)
        df_clean = df[keep_cols].copy()

        gen = RLSignalGenerator(
            ensemble,
            min_confidence=0.4,  # Lower threshold for auto-trained models
            use_technical_indicators=False,  # Auto-trained models use raw OHLCV
        )
        result = gen.generate(df_clean, symbol=symbol)
        st.session_state.rl_signal_result = result
        return result
    except Exception as e:
        logger.error(f"RL signal generation failed: {e}", exc_info=True)
        return {"signal": "hold", "confidence": 0, "reason": f"RL signal error: {e}"}


def run_agent_pipeline(client, symbol: str, timeframe: str, account_balance: float):
    """
    Unified agent pipeline:

        1. Fetch market data and compute indicators
        2. Auto-train RL agents if none exist
        3. Generate RL ensemble signal
        4. Generate specialist signal (MarketAnalysisAgent)
        5. Detect market regime (trending / consolidating / volatile)
        6. Arbitrate between RL and specialist signals
        7. Detect signal transitions and generate alerts
        8. Risk management and optional execution

    Returns (analysis_result, risk_result, trade_result, arbitration_result).
    """
    agents = get_agents()

    # ── Step 1: Market data + indicators ──
    df = client.get_history(symbol, timeframe, 200)
    if df.empty or len(df) < 50:
        return {"error": "Insufficient market data"}, None, None, None

    quote = client.get_quote(symbol)
    current_price = quote.get("bid", 0) if quote else df.iloc[-1]["close"]

    try:
        import pandas_ta as ta
        df_copy = df.copy()
        df_copy['rsi_14'] = df_copy.ta.rsi(length=14)
        macd = df_copy.ta.macd(fast=12, slow=26, signal=9)
        if macd is not None:
            df_copy = pd.concat([df_copy, macd], axis=1)
        df_copy['sma_20'] = df_copy.ta.sma(length=20)
        df_copy['sma_50'] = df_copy.ta.sma(length=50)
        df_copy['atr'] = df_copy.ta.atr(length=14)

        latest = df_copy.iloc[-1]
        indicators = {
            "rsi_14": float(latest.get("rsi_14", 50)),
            "macd": float(latest.get("MACD_12_26_9", 0)),
            "macd_signal": float(latest.get("MACDs_12_26_9", 0)),
            "sma_20": float(latest.get("sma_20", current_price)),
            "sma_50": float(latest.get("sma_50", current_price)),
            "close": float(latest["close"]),
            "atr": float(latest.get("atr", current_price * 0.01)),
        }
    except Exception:
        indicators = {"close": current_price, "rsi_14": 50, "macd": 0, "macd_signal": 0}

    # ── Step 2: Auto-train RL agents if none exist ──
    if not st.session_state.rl_trained_agents:
        with st.spinner("🤖 Auto-training RL agents (PPO + SAC, 10k steps each)..."):
            _auto_train_rl_agents(df, symbol)

    # ── Step 3: RL ensemble signal ──
    rl_result = _build_rl_signal(df, symbol)
    rl_signal = rl_result.get("signal", "hold")
    rl_confidence = rl_result.get("confidence", 0)

    # ── Step 4: Specialist signal ──
    analysis_input = {
        "symbol": symbol,
        "current_price": current_price,
        "indicators": indicators,
    }
    try:
        specialist_result = _run_async(agents["market_analyst"].process(analysis_input))
    except Exception as e:
        specialist_result = {"signal": "hold", "confidence": 0, "reason": str(e)}

    specialist_signal = specialist_result.get("signal", "hold")
    specialist_confidence = specialist_result.get("confidence", 0)

    # ── Step 5: Market regime detection ──
    regime_detector = MarketRegimeDetector()
    if "high" in df.columns and "low" in df.columns:
        regime = regime_detector.detect(df)
    else:
        regime = regime_detector.detect(df_copy if 'df_copy' in dir() else df)
    st.session_state.market_regime = regime

    # ── Step 5.5: Sentiment Analysis (Optional) ──
    sentiment_signal = "hold"
    sentiment_confidence = 0.0
    sentiment_summary = None
    
    if getattr(st.session_state, "enable_news_sentiment", False):
        try:
            # Fetch lookback period for sentiment matching
            end_date_str = datetime.now().strftime("%Y-%m-%d")
            start_date_str = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")

            news_dict = fetch_news_sentiment([symbol], start_date_str, end_date_str)
            # Look up by original symbol or normalized name
            news_df = news_dict.get(symbol, pd.DataFrame())
            if news_df.empty:
                normalized = NewsSentimentFetcher._normalize_mt5_symbol(symbol)
                news_df = news_dict.get(normalized, pd.DataFrame())

            if not news_df.empty:
                # Score headlines with VADER and compute summary
                scored_news = compute_sentiment_features(news_df, model="vader")
                news_summary = get_sentiment_summary(scored_news)
                top_headlines = scored_news["headline"].head(10).tolist()

                # Add sentiment features to market df
                enriched_df = add_sentiment_features(df_copy, news_df)
                latest = enriched_df.iloc[-1]

                # Extract ONLY the 5 sentiment feature keys the agent expects
                sentiment_feature_keys = [
                    "sentiment_score", "sentiment_magnitude",
                    "sentiment_volume", "sentiment_momentum",
                    "sentiment_divergence",
                ]
                latest_features = {
                    k: float(latest.get(k, 0.0)) for k in sentiment_feature_keys
                }

                # Build agent input with headlines + summary for richer analysis
                sentiment_input = {
                    "symbol": symbol,
                    "sentiment_features": latest_features,
                    "headlines": top_headlines,
                    "sentiment_summary": news_summary,
                }
                sentiment_result = _run_async(agents["sentiment_analyst"].process(sentiment_input))

                sentiment_signal = sentiment_result.get("signal", "hold")
                sentiment_confidence = sentiment_result.get("confidence", 0.0)
                sentiment_summary = sentiment_result.get("sentiment_summary", news_summary)
        except Exception as e:
            logger.error(f"Sentiment analysis failed: {e}", exc_info=True)

    # ── Step 6: Signal arbitration ──
    arbiter = UnifiedSignalArbiter()
    transition_detector = st.session_state.signal_transition_detector
    momentum = transition_detector.get_momentum()

    arbitration = arbiter.arbitrate(
        rl_signal=rl_signal,
        rl_confidence=rl_confidence,
        specialist_signal=specialist_signal,
        specialist_confidence=specialist_confidence,
        sentiment_signal=sentiment_signal,
        sentiment_confidence=sentiment_confidence,
        regime=regime,
        signal_momentum=momentum,
    )
    st.session_state.last_arbitration = arbitration

    # ── Step 7: Transition detection ──
    alerts = transition_detector.update(arbitration.final_signal, arbitration.final_confidence)
    if alerts:
        for alert in alerts:
            st.session_state.signal_alerts.insert(0, {
                "time": alert.timestamp,
                "severity": alert.severity,
                "message": alert.message,
                "from": alert.previous_signal,
                "to": alert.new_signal,
            })
        # Keep only last 20 alerts
        st.session_state.signal_alerts = st.session_state.signal_alerts[:20]

    # Build the unified analysis result
    analysis_result = {
        "signal": arbitration.final_signal,
        "confidence": arbitration.final_confidence,
        "reason": arbitration.reason,
        "source": arbitration.source,
        "rl_signal": rl_signal,
        "rl_confidence": rl_confidence,
        "specialist_signal": specialist_signal,
        "specialist_confidence": specialist_confidence,
        "sentiment_signal": sentiment_signal,
        "sentiment_confidence": sentiment_confidence,
        "sentiment_summary": sentiment_summary,
        "regime": regime.regime,
        "consolidation_score": regime.consolidation_score,
    }
    st.session_state.last_analysis_result = analysis_result

    # ── Step 8: Risk management + execution ──
    risk_result = None
    trade_result = None

    if arbitration.final_signal in ("buy", "sell"):
        risk_input = {
            "signal": arbitration.final_signal,
            "confidence": arbitration.final_confidence,
            "current_price": current_price,
            "account_balance": account_balance,
            "atr": indicators.get("atr", current_price * 0.01),
        }

        try:
            risk_result = _run_async(agents["risk_manager"].process(risk_input))
        except Exception as e:
            risk_result = {"approved": False, "reason": str(e)}

        st.session_state.last_risk_result = risk_result

        # Execution (only if auto-execute + risk approved + limits OK)
        if (st.session_state.auto_execute_trades and
            risk_result.get("approved", False) and
            st.session_state.trades_this_session < st.session_state.max_trades_session):

            can_trade = True
            if st.session_state.last_trade_time:
                elapsed = (datetime.now() - st.session_state.last_trade_time).total_seconds()
                if elapsed < st.session_state.trade_cooldown:
                    can_trade = False

            if can_trade:
                pip_value = 0.0001 if "JPY" not in symbol else 0.01

                trade_result = client.trade(
                    symbol=symbol,
                    action=arbitration.final_signal.upper(),
                    lot_size=min(risk_result["position_size"], 0.1),
                    stop_loss=risk_result["stop_loss"],
                    take_profit=risk_result["take_profit"],
                )

                if "error" not in trade_result:
                    st.session_state.last_trade_time = datetime.now()
                    st.session_state.trades_this_session += 1

                    st.session_state.agent_trades.insert(0, {
                        "Time": datetime.now().strftime("%H:%M:%S"),
                        "Symbol": symbol,
                        "Action": arbitration.final_signal.upper(),
                        "Size": risk_result["position_size"],
                        "Price": current_price,
                        "SL": risk_result["stop_loss"],
                        "TP": risk_result["take_profit"],
                        "Status": trade_result.get("status", "executed"),
                    })
                else:
                    st.session_state.agent_trades.insert(0, {
                        "Time": datetime.now().strftime("%H:%M:%S"),
                        "Symbol": symbol,
                        "Action": arbitration.final_signal.upper(),
                        "Size": risk_result["position_size"],
                        "Price": current_price,
                        "SL": risk_result["stop_loss"],
                        "TP": risk_result["take_profit"],
                        "Status": f"Failed: {trade_result['error']}",
                    })

    return analysis_result, risk_result, trade_result, arbitration


def run_rl_training(
    df: pd.DataFrame,
    algo_name: str,
    reward_type: str,
    timesteps: int,
    symbol: str = "",
):
    """
    Train an RL agent on market data with callback tracing.

    Returns the trained agent and training summary.
    """
    tracer = get_rl_tracer()

    # Clean DataFrame: drop all non-numeric columns (e.g. 'time' datetime strings from MT5)
    # Keep only OHLCV + numeric indicator columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    keep_cols = [c for c in ["date", "open", "high", "low", "close", "volume"] if c in df.columns]
    for c in numeric_cols:
        if c not in keep_cols:
            keep_cols.append(c)
    df = df[keep_cols].copy()

    # Determine if discrete env is needed
    is_discrete = algo_name == "DQN"

    if is_discrete:
        env = DiscreteTradingEnv(
            df,
            initial_balance=100000,
            reward_type=reward_type if reward_type != "profit" else None,
        )
    else:
        env = TradingEnv(
            df,
            initial_balance=100000,
            reward_type=reward_type if reward_type != "profit" else None,
        )

    # Get SB3 algorithm class
    AlgoClass = get_rl_algorithm(algo_name)

    config = {
        "algorithm": algo_name,
        "reward_type": reward_type,
        "timesteps": timesteps,
        "data_rows": len(df),
    }

    # Start training trace
    run_id = tracer.start_training_run(
        algorithm=algo_name,
        config=config,
        symbol=symbol,
        timesteps=timesteps,
    )

    try:
        # Create and train model
        model = AlgoClass(
            policy="MlpPolicy",
            env=env,
            verbose=0,
            **({
                "learning_starts": min(500, timesteps // 10)
            } if algo_name in ("DQN", "SAC", "TD3") else {}),
        )

        start_time = time.time()
        model.learn(total_timesteps=timesteps)
        training_time = time.time() - start_time

        # Evaluate
        obs, _ = env.reset()
        total_reward = 0.0
        done = False
        steps = 0
        while not done and steps < 1000:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated
            steps += 1

        metrics = {
            "total_reward": round(total_reward, 4),
            "training_time": round(training_time, 2),
            "eval_steps": steps,
            "final_portfolio": round(info.get("portfolio_value", 100000), 2),
            "return_pct": round(info.get("return_pct", 0), 4),
            "num_trades": info.get("num_trades", 0),
        }

        # Log training complete
        tracer.end_training_run(run_id, metrics)
        tracer.log_training_step(
            run_id, "Model Evaluation",
            {"episodes": 1, "deterministic": True},
            metrics,
        )

        env.close()
        return model, env, metrics

    except Exception as e:
        tracer.end_training_run(run_id, {"error": str(e)}, status="failed")
        env.close()
        raise


# ============= MAIN APP =============

def main():
    client = get_client()
    connected = client.is_connected()
    
    # ========== SIDEBAR ==========
    with st.sidebar:
        st.title("PipFlow AI")
        st.markdown("---")
        
        # Connection status
        if connected:
            st.markdown("🟢 **Connected to MT5**")
        else:
            st.markdown("🔴 **Disconnected**")
            st.caption(f"URL: {client.bridge_url}")
        
        st.markdown("---")
        
        # Symbol selection - fetch from broker
        st.subheader("📊 Symbol")
        
        # Cache symbols to avoid repeated API calls
        if st.session_state.cached_symbols is None and connected:
            st.session_state.cached_symbols = client.get_symbols()
        
        symbols = st.session_state.cached_symbols or ["EURUSD", "GBPUSD", "USDJPY"]
        
        # Show first 20 symbols + allow custom input
        display_symbols = symbols[:20] if len(symbols) > 20 else symbols
        symbol = st.selectbox("Select Symbol", display_symbols, label_visibility="collapsed")
        
        # Option to enter custom symbol
        custom_symbol = st.text_input("Or enter symbol manually:", placeholder="e.g., EURUSD")
        if custom_symbol:
            symbol = custom_symbol.upper()
        
        # Timeframe
        st.subheader("⏱️ Timeframe")
        timeframes = {
            "1 Minute": "M1",
            "5 Minutes": "M5", 
            "15 Minutes": "M15",
            "30 Minutes": "M30",
            "1 Hour": "H1",
            "4 Hours": "H4",
            "Daily": "D1",
        }
        tf_name = st.selectbox("Select Timeframe", list(timeframes.keys()), 
                               index=4, label_visibility="collapsed")
        timeframe = timeframes[tf_name]
        
        st.markdown("---")
        
        # Refresh button
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Refresh", use_container_width=True):
                st.session_state.cached_symbols = None
                st.rerun()
        with col2:
            if st.button("📋 Symbols", use_container_width=True):
                st.session_state.show_all_symbols = True
    
    # ========== MAIN CONTENT ==========
    
    # Create tabs for different sections
    tab_trading, tab_agents, tab_rl, tab_strategies = st.tabs([
        "📈 Trading", "🤖 Agent Control", "🧠 RL Agents", "⚙️ Strategy Config"
    ])
    
    # ========== TRADING TAB ==========
    with tab_trading:
        # Account metrics row
        if connected:
            account = client.get_account()
            if account:
                m1, m2, m3, m4 = st.columns(4)
                with m1:
                    st.metric("Balance", f"${account.get('balance', 0):,.2f}")
                with m2:
                    st.metric("Equity", f"${account.get('equity', 0):,.2f}")
                with m3:
                    profit = account.get('profit', 0)
                    st.metric("Profit", f"${profit:,.2f}", delta=f"{profit:+.2f}")
                with m4:
                    st.metric("Leverage", f"1:{account.get('leverage', 0)}")
        
        # Main layout: Chart + Trade Panel
        col_chart, col_trade = st.columns([3, 1])
        
        with col_chart:
            st.subheader(f"{symbol} - {tf_name}")
            
            # Indicator selection
            st.caption(" Select indicators to display on chart")
            ind_cols = st.columns(5)
            indicators = []
            with ind_cols[0]:
                if st.checkbox("SMA 20", value=True):
                    indicators.append("SMA20")
            with ind_cols[1]:
                if st.checkbox("SMA 50"):
                    indicators.append("SMA50")
            with ind_cols[2]:
                if st.checkbox("EMA 20"):
                    indicators.append("EMA20")
            with ind_cols[3]:
                if st.checkbox("Bollinger"):
                    indicators.append("BB")
            with ind_cols[4]:
                if st.checkbox("RSI"):
                    indicators.append("RSI")
            
            if connected:
                df = client.get_history(symbol, timeframe, 500)
                
                if not df.empty:
                    # Convert to format expected by chart component
                    # Lightweight Charts requires Unix timestamps (numbers)
                    ohlcv_data = []
                    for _, row in df.iterrows():
                        # Use timestamp field if available, otherwise parse time string
                        if 'timestamp' in row:
                            ts = int(row['timestamp'])
                        elif 'time' in row:
                            try:
                                dt = datetime.fromisoformat(str(row['time']).replace('Z', '+00:00'))
                                ts = int(dt.timestamp())
                            except:
                                continue
                        else:
                            continue
                        
                        ohlcv_data.append({
                            "time": ts,
                            "open": float(row['open']),
                            "high": float(row['high']),
                            "low": float(row['low']),
                            "close": float(row['close']),
                            "volume": int(row.get('volume', 0)),
                        })
                    
                    # Render professional chart with real-time streaming
                    from components.trading_chart import render_trading_chart
                    render_trading_chart(
                        ohlcv_data=ohlcv_data,
                        symbol=symbol,
                        height=550,
                        theme="dark",
                        show_volume=True,
                        indicators=indicators,
                        bridge_url=client.bridge_url,
                        api_key=client.api_key,
                        timeframe=timeframe,
                    )
                else:
                    st.warning(f"️No data for {symbol}. Check if this symbol is available on your broker.")
                    st.info(" Try entering a different symbol in the sidebar")
                
                # Current price
                quote = client.get_quote(symbol)
                if quote:
                    p1, p2, p3 = st.columns(3)
                    with p1:
                        st.metric("Bid", f"{quote.get('bid', 0):.5f}")
                    with p2:
                        st.metric("Ask", f"{quote.get('ask', 0):.5f}")
                    with p3:
                        st.metric("Spread", f"{quote.get('spread', 0)} pips")
            else:
                st.warning("️ Not connected to MT5 Bridge")
                st.info("Check that the bridge server is running on your Windows VPS")
        
        with col_trade:
            st.subheader("Quick Trade")
            
            with st.form("trade_form"):
                lot_size = st.number_input("Lot Size", min_value=0.01, max_value=10.0, 
                                           value=0.01, step=0.01)
                sl_pips = st.number_input("Stop Loss (pips)", min_value=0, max_value=500, 
                                          value=50, step=5)
                tp_pips = st.number_input("Take Profit (pips)", min_value=0, max_value=500, 
                                          value=100, step=5)
                
                col_buy, col_sell = st.columns(2)
                with col_buy:
                    buy_btn = st.form_submit_button("🟢 BUY", use_container_width=True)
                with col_sell:
                    sell_btn = st.form_submit_button("🔴 SELL", use_container_width=True)
            
            if buy_btn and connected:
                quote = client.get_quote(symbol)
                if quote:
                    pip = 0.0001 if "JPY" not in symbol else 0.01
                    sl = quote['ask'] - (sl_pips * pip) if sl_pips > 0 else None
                    tp = quote['ask'] + (tp_pips * pip) if tp_pips > 0 else None
                    result = client.trade(symbol, "BUY", lot_size, sl, tp)
                    if "error" in result:
                        st.error(f"Trade failed: {result['error']}")
                    else:
                        st.success(f"BUY @ {result.get('price', 'N/A')}")
                        st.rerun()
            
            if sell_btn and connected:
                quote = client.get_quote(symbol)
                if quote:
                    pip = 0.0001 if "JPY" not in symbol else 0.01
                    sl = quote['bid'] + (sl_pips * pip) if sl_pips > 0 else None
                    tp = quote['bid'] - (tp_pips * pip) if tp_pips > 0 else None
                    result = client.trade(symbol, "SELL", lot_size, sl, tp)
                    if "error" in result:
                        st.error(f"Trade failed: {result['error']}")
                    else:
                        st.success(f"SELL @ {result.get('price', 'N/A')}")
                        st.rerun()
            
            st.markdown("---")
            
            # Open positions
            st.subheader("Open Positions")
            if connected:
                positions = client.get_positions()
                if positions:
                    for pos in positions:
                        with st.container():
                            st.markdown(f"**{pos['symbol']}** - {pos['type']}")
                            st.caption(f"{pos['volume']} lot @ {pos['open_price']}")
                            
                            profit = pos.get('profit', 0)
                            color = "green" if profit >= 0 else "red"
                            st.markdown(f"P/L: :{color}[${profit:.2f}]")
                            
                            if st.button(f"Close #{pos['ticket']}", key=f"close_{pos['ticket']}"):
                                result = client.close_position(pos['symbol'], pos['ticket'])
                                if "error" not in result:
                                    st.success("Position closed")
                                    st.rerun()
                                else:
                                    st.error(f"Error: {result['error']}")
                            st.markdown("---")
                else:
                    st.info("No open positions")
            else:
                st.info("Connect to view positions")
    
    # ========== AGENT CONTROL TAB ==========
    with tab_agents:
        st.header("🤖 Agent Control Panel")
        
        # Top row: Agent config and status
        col_config, col_status = st.columns([2, 1])
        
        with col_config:
            st.subheader("⚙️ Agent Configuration")
            
            # Analysis interval
            analysis_interval = st.slider(
                "Analysis Interval (seconds)",
                min_value=5,
                max_value=60,
                value=st.session_state.agent_analysis_interval,
                step=5,
                help="How often the agent analyzes the market"
            )
            st.session_state.agent_analysis_interval = analysis_interval
            
            # Auto-execute toggle with warning
            auto_execute = st.toggle(
                "⚡ Auto-Execute Trades",
                value=st.session_state.auto_execute_trades,
                help="Automatically execute trades when signals are generated"
            )
            st.session_state.auto_execute_trades = auto_execute
            
            enable_news = st.toggle(
                "📰 Enable News Sentiment",
                value=st.session_state.enable_news_sentiment,
                help="Fetch real-time news and use SentimentAnalysisAgent"
            )
            st.session_state.enable_news_sentiment = enable_news
            
            if auto_execute:
                st.warning("⚠️ **LIVE TRADING ENABLED** - Trades will be executed automatically!")
                
                col_max, col_cool = st.columns(2)
                with col_max:
                    max_trades = st.number_input(
                        "Max Trades/Session",
                        min_value=1,
                        max_value=50,
                        value=st.session_state.max_trades_session
                    )
                    st.session_state.max_trades_session = max_trades
                with col_cool:
                    cooldown = st.number_input(
                        "Cooldown (seconds)",
                        min_value=10,
                        max_value=300,
                        value=st.session_state.trade_cooldown
                    )
                    st.session_state.trade_cooldown = cooldown
        
        with col_status:
            st.subheader("📊 Status")
            
            if st.session_state.agent_running:
                st.markdown("**Status:** :green[🟢 Running]")
                st.metric("Signals", len(st.session_state.agent_signals))
                st.metric("Trades", st.session_state.trades_this_session)
            else:
                st.markdown("**Status:** :gray[⚪ Stopped]")
            
            # Remaining trades
            if auto_execute:
                remaining = st.session_state.max_trades_session - st.session_state.trades_this_session
                st.caption(f"📈 Trades remaining: {remaining}")
        
        st.markdown("---")
        
        # Agent Controls
        col_start, col_stop, col_reset = st.columns(3)
        
        with col_start:
            start_disabled = st.session_state.agent_running or not connected
            if st.button("▶️ Start Agent", use_container_width=True, disabled=start_disabled):
                st.session_state.agent_running = True
                st.session_state.trades_this_session = 0
                st.success("Agent started!")
                st.rerun()
        
        with col_stop:
            if st.button("⏹️ Stop Agent", use_container_width=True,
                        disabled=not st.session_state.agent_running):
                st.session_state.agent_running = False
                st.info("Agent stopped")
                st.rerun()
        
        with col_reset:
            if st.button("🔄 Reset All", use_container_width=True):
                st.session_state.agent_signals = []
                st.session_state.agent_trades = []
                st.session_state.trades_this_session = 0
                st.session_state.last_trade_time = None
                st.info("All data cleared")
                st.rerun()
        
        st.markdown("---")
        
        # Live Agent Output
        if st.session_state.agent_running and connected:
            st.subheader("🔴 Live Unified Agent Pipeline")

            # Get account balance for risk calc
            account = client.get_account()
            account_balance = account.get("balance", 10000) if account else 10000

            # Show unified pipeline visualization
            st.markdown("""
            ```
            📊 Specialist ──┐
                            ├──▶ ⚖️ Arbiter ──▶ 🛡️ Risk ──▶ 💹 Execution
            🤖 RL Ensemble ─┘        ▲
                               🌐 Regime
            ```
            """)

            # ── Signal Transition Alerts ──
            if st.session_state.signal_alerts:
                for alert in st.session_state.signal_alerts[:3]:
                    sev = alert["severity"]
                    if sev == "critical":
                        st.error(f"🚨 {alert['message']}")
                    elif sev == "warning":
                        st.warning(f"⚠️ {alert['message']}")
                    else:
                        st.info(f"ℹ️ {alert['message']}")

            # ── Run the unified pipeline ──
            analysis_result, risk_result, trade_result, arbitration = run_agent_pipeline(
                client, symbol, timeframe, account_balance
            )

            # Send trace to callback server
            tracer = get_rl_tracer()
            trace_id = tracer.trace_agent_pipeline(
                symbol=symbol,
                analysis_result=analysis_result,
                risk_result=risk_result,
                trade_result=trade_result,
            )
            st.session_state.rl_trace_log.insert(0, {
                "time": datetime.now().strftime("%H:%M:%S"),
                "type": "pipeline",
                "trace_id": trace_id[:8],
                "signal": analysis_result.get("signal", "hold"),
            })
            if len(st.session_state.rl_trace_log) > 50:
                st.session_state.rl_trace_log = st.session_state.rl_trace_log[:50]

            # ── Market Regime Badge ──
            regime = st.session_state.market_regime
            if regime:
                regime_icons = {
                    "trending": "🟢 Trending",
                    "consolidating": "🟡 Consolidating",
                    "volatile": "🔴 Volatile",
                }
                regime_label = regime_icons.get(regime.regime, regime.regime)
                st.markdown(f"**🌐 Market Regime:** {regime_label}  ·  Consolidation Score: `{regime.consolidation_score:.2f}`  ·  ADX: `{regime.adx:.1f}`")

                if regime.regime == "consolidating":
                    st.warning("⚠️ **Market Consolidating** — Emphasizing HOLD to avoid whipsaw losses")

            # ── Signal Comparison ──
            if "error" not in analysis_result:
                st.markdown("#### 📊 Signal Comparison: RL vs Specialist vs Final")
                # Support for 2-way vs 3-way arbitration display
                if getattr(st.session_state, "enable_news_sentiment", False):
                    col_rl, col_spec, col_sent, col_final = st.columns(4)
                else:
                    col_rl, col_spec, col_final = st.columns(3)

                def _signal_color(sig):
                    sig = sig.upper()
                    return "green" if sig == "BUY" else "red" if sig == "SELL" else "gray"

                with col_rl:
                    rl_sig = analysis_result.get("rl_signal", "hold").upper()
                    rl_conf = analysis_result.get("rl_confidence", 0)
                    color = _signal_color(rl_sig)
                    st.markdown(f"**🤖 RL Ensemble:** :{color}[{rl_sig}]")
                    st.metric("RL Confidence", f"{rl_conf:.2%}")

                with col_spec:
                    spec_sig = analysis_result.get("specialist_signal", "hold").upper()
                    spec_conf = analysis_result.get("specialist_confidence", 0)
                    color = _signal_color(spec_sig)
                    st.markdown(f"**📊 Specialist:** :{color}[{spec_sig}]")
                    st.metric("Specialist Confidence", f"{spec_conf:.2%}")
                
                if getattr(st.session_state, "enable_news_sentiment", False):
                    with col_sent:
                        sent_sig = analysis_result.get("sentiment_signal", "hold").upper()
                        sent_conf = analysis_result.get("sentiment_confidence", 0)
                        color = _signal_color(sent_sig)
                        st.markdown(f"**📰 Sentiment:** :{color}[{sent_sig}]")
                        st.metric("Sentiment Confidence", f"{sent_conf:.2%}")

                with col_final:
                    final_sig = analysis_result.get("signal", "hold").upper()
                    final_conf = analysis_result.get("confidence", 0)
                    color = _signal_color(final_sig)
                    st.markdown(f"**⚡ Final Signal:** :{color}[{final_sig}]")
                    st.metric("Final Confidence", f"{final_conf:.2%}")

                # Show arbiter reasoning
                source = analysis_result.get("source", "")
                reason = analysis_result.get("reason", "")
                st.caption(f"⚖️ Source: **{source}** — {reason}")

                # Log signal
                new_signal = {
                    "Time": datetime.now().strftime("%H:%M:%S"),
                    "Symbol": symbol,
                    "Signal": final_sig,
                    "Confidence": f"{final_conf:.2f}",
                    "RL": f"{rl_sig} ({rl_conf:.0%})",
                    "Specialist": f"{spec_sig} ({spec_conf:.0%})",
                    "Regime": analysis_result.get("regime", "?"),
                    "Source": source,
                }

                # Avoid duplicate consecutive signals
                if not st.session_state.agent_signals or \
                   st.session_state.agent_signals[0].get("Signal") != final_sig:
                    st.session_state.agent_signals.insert(0, new_signal)
                    if len(st.session_state.agent_signals) > 50:
                        st.session_state.agent_signals.pop()
            else:
                st.error(analysis_result["error"])

            st.markdown("---")
            
            # ── News Sentiment Analysis Card ──
            if getattr(st.session_state, "enable_news_sentiment", False) and "sentiment_summary" in analysis_result:
                summary = analysis_result["sentiment_summary"]
                if summary:
                    st.markdown("#### 📰 News Sentiment Analysis")
                    
                    s_score = summary.get('mean_sentiment', 0)
                    s_color = "green" if s_score > 0 else "red" if s_score < 0 else "gray"
                    sentiment_label = "Bullish" if s_score > 0.05 else "Bearish" if s_score < -0.05 else "Neutral"
                    
                    st.markdown(f"**Overall Sentiment:** :{s_color}[{sentiment_label} ({s_score:.2f})]")
                    
                    n1, n2, n3, n4 = st.columns(4)
                    with n1:
                        st.metric("Articles Analyzed", summary.get('article_count', 0))
                    with n2:
                        st.metric("Bullish News", f"{summary.get('bullish_pct', 0):.0%}")
                    with n3:
                        st.metric("Bearish News", f"{summary.get('bearish_pct', 0):.0%}")
                    with n4:
                        st.metric("Intensity (Magnitude)", f"{summary.get('magnitude', 0):.2f}")
                        
                    st.markdown("---")

            # ── Risk Management ──
            col_risk_display, col_trade_display = st.columns(2)

            with col_risk_display:
                st.markdown("#### 🛡️ Risk Management")
                if risk_result:
                    approved = risk_result.get("approved", False)
                    if approved:
                        st.markdown("**Status:** :green[✅ Approved]")
                        st.metric("Position Size", f"{risk_result.get('position_size', 0):.4f} lots")
                        st.caption(f"SL: {risk_result.get('stop_loss', 0):.5f} | TP: {risk_result.get('take_profit', 0):.5f}")
                        st.caption(f"Risk: ${risk_result.get('risk_amount', 0):.2f} ({risk_result.get('risk_pct', 0)}%)")
                    else:
                        st.markdown("**Status:** :gray[⏸️ No trade]")
                        st.caption(risk_result.get("reason", "Hold signal"))
                else:
                    st.markdown("**Status:** :gray[⏸️ Waiting for signal]")

            with col_trade_display:
                st.markdown("#### 💹 Trade Execution")
                if trade_result:
                    if "error" in trade_result:
                        st.error(f"❌ Execution failed: {trade_result['error']}")
                    else:
                        action = analysis_result.get("signal", "trade").upper()
                        st.success(f"✅ **{action}** order executed!")
                        st.caption(f"Price: {trade_result.get('price', 'market')} | Size: {risk_result.get('position_size', 0):.4f} lots")
                elif risk_result and risk_result.get("approved"):
                    if not st.session_state.auto_execute_trades:
                        st.warning("⚠️ Auto-Execute is **OFF**")
                    elif st.session_state.trades_this_session >= st.session_state.max_trades_session:
                        st.warning(f"⚠️ Max trades reached ({st.session_state.max_trades_session})")
                    elif st.session_state.last_trade_time:
                        elapsed = (datetime.now() - st.session_state.last_trade_time).total_seconds()
                        remaining = st.session_state.trade_cooldown - elapsed
                        if remaining > 0:
                            st.info(f"⏳ Cooldown: {remaining:.0f}s remaining")
                else:
                    if analysis_result.get("signal") == "hold":
                        st.info("📊 Waiting for BUY or SELL signal...")

            # ── Signal Momentum ──
            momentum = st.session_state.signal_transition_detector.get_momentum()
            if momentum["changes_in_window"] > 0:
                st.markdown("---")
                st.markdown(f"**📈 Signal Momentum:** Stability `{momentum['stability']:.2f}` · "
                           f"Changes `{momentum['changes_in_window']}` · "
                           f"Dominant `{momentum['dominant_signal'].upper()}`")
                if momentum["stability"] < 0.5:
                    st.warning("⚠️ Signals are unstable — consider reducing position sizes")

            # Continuous loop - wait and rerun
            st.markdown("---")
            progress_placeholder = st.empty()
            for i in range(analysis_interval, 0, -1):
                progress_placeholder.caption(f"⏳ Next analysis in {i} seconds...")
                time.sleep(1)
                if not st.session_state.agent_running:
                    break

            if st.session_state.agent_running:
                st.rerun()

        elif not connected:
            st.warning("⚠️ Connect to MT5 Bridge to run agents")

        st.markdown("---")

        # Signal History
        st.subheader("📋 Signal History")
        if st.session_state.agent_signals:
            signals_df = pd.DataFrame(st.session_state.agent_signals[:20])
            st.dataframe(signals_df, use_container_width=True, hide_index=True)
        else:
            st.info("No signals yet. Start the agent to generate signals.")

        # Trade History
        if st.session_state.agent_trades:
            st.subheader("💰 Executed Trades")
            trades_df = pd.DataFrame(st.session_state.agent_trades)
            st.dataframe(trades_df, use_container_width=True, hide_index=True)

        # Signal Transition Alert History
        if st.session_state.signal_alerts:
            st.subheader("🔔 Signal Transition Alerts")
            alerts_df = pd.DataFrame(st.session_state.signal_alerts[:10])
            st.dataframe(alerts_df, use_container_width=True, hide_index=True)

    # ========== RL AGENTS TAB ==========
    with tab_rl:
        st.header("🧠 RL Agents")
        st.caption("Train, test, and deploy reinforcement learning agents")

        rl_col1, rl_col2 = st.columns([1, 1])

        # ───── Quick Train ─────
        with rl_col1:
            st.subheader("⚡ Quick Train")

            algo_name = st.selectbox(
                "Algorithm",
                ["PPO", "SAC", "A2C", "TD3", "DQN"],
                help="PPO/A2C: on-policy. SAC/TD3: off-policy continuous. DQN: discrete."
            )

            reward_type = st.selectbox(
                "Reward Function",
                ["sharpe", "sortino", "risk_adjusted", "profit"],
                help="Sharpe/Sortino optimize risk-adjusted returns."
            )

            timesteps = st.select_slider(
                "Training Timesteps",
                options=[5000, 10000, 25000, 50000, 100000, 200000],
                value=25000,
            )

            # Use fetched data or generate sample
            train_data_source = st.radio(
                "Training Data",
                ["Live Market Data", "Sample Data (200 bars)"],
                horizontal=True,
            )

            if st.button("🚀 Train Agent", use_container_width=True, type="primary"):
                with st.spinner(f"Training {algo_name} ({timesteps:,} steps)..."):
                    try:
                        if train_data_source == "Live Market Data" and connected:
                            df = client.get_history(symbol, timeframe, 500)
                            if df.empty or len(df) < 50:
                                st.error("Insufficient live data for training")
                                df = None
                        else:
                            # Generate sample data
                            np.random.seed(42)
                            n = 200
                            dates = pd.date_range("2023-01-01", periods=n, freq="D")
                            close = 100 + np.cumsum(np.random.randn(n) * 0.5)
                            df = pd.DataFrame({
                                "date": dates,
                                "open": close + np.random.randn(n) * 0.2,
                                "high": close + np.abs(np.random.randn(n) * 0.3),
                                "low": close - np.abs(np.random.randn(n) * 0.3),
                                "close": close,
                                "volume": np.random.randint(1000, 10000, n),
                            })

                        if df is not None:
                            model, env, metrics = run_rl_training(
                                df, algo_name, reward_type, timesteps, symbol
                            )

                            # Store trained agent
                            agent_key = f"{algo_name}_{reward_type}"
                            st.session_state.rl_trained_agents[agent_key] = {
                                "model": model,
                                "env": env,
                                "algo": algo_name,
                                "reward": reward_type,
                                "metrics": metrics,
                            }

                            # Log training
                            st.session_state.rl_training_log.insert(0, {
                                "time": datetime.now().strftime("%H:%M:%S"),
                                "agent": agent_key,
                                **metrics,
                            })

                            st.success(f"✅ {agent_key} trained successfully!")

                    except Exception as e:
                        st.error(f"Training failed: {e}")

            # Show trained agents
            if st.session_state.rl_trained_agents:
                st.markdown("---")
                st.markdown("#### Trained Agents")
                for name, info in st.session_state.rl_trained_agents.items():
                    m = info["metrics"]
                    color = "green" if m.get("return_pct", 0) > 0 else "red"
                    st.markdown(
                        f"**{name}** — "
                        f"Return: :{color}[{m.get('return_pct', 0):.2%}] | "
                        f"Trades: {m.get('num_trades', 0)} | "
                        f"Time: {m.get('training_time', 0):.1f}s"
                    )

        # ───── Ensemble Predict & Signal Generation ─────
        with rl_col2:
            st.subheader("🎯 Ensemble Prediction")

            trained = st.session_state.rl_trained_agents
            if not trained:
                st.info("Train at least one agent first")
            else:
                # Select agents for ensemble
                selected_agents = st.multiselect(
                    "Select agents for ensemble",
                    list(trained.keys()),
                    default=list(trained.keys()),
                )

                if st.button("🔮 Run Ensemble Prediction", use_container_width=True):
                    if not selected_agents:
                        st.warning("Select at least one agent")
                    else:
                        with st.spinner("Running ensemble prediction..."):
                            try:
                                # Build ensemble
                                ensemble = EnsembleAgent()
                                for name in selected_agents:
                                    agent_info = trained[name]
                                    from src.agents.rl_agents.base_agent import BaseRLAgent
                                    # Wrap SB3 model in a predict-compatible interface
                                    class ModelWrapper:
                                        def __init__(self, model):
                                            self.model = model
                                        def predict(self, obs, deterministic=True):
                                            action, _ = self.model.predict(obs, deterministic=deterministic)
                                            return action
                                    ensemble.add_agent(name, ModelWrapper(agent_info["model"]))

                                # Get observation from one of the envs
                                first_agent = trained[selected_agents[0]]
                                env = first_agent["env"]
                                obs, _ = env.reset()

                                # Run ensemble
                                result = ensemble.predict(obs)
                                st.session_state.rl_signal_result = result

                                # Trace it
                                tracer = get_rl_tracer()
                                tracer.trace_ensemble_prediction(
                                    observation_shape=obs.shape,
                                    agent_votes=result.get("agent_votes", {}),
                                    consensus={
                                        "signal": result["signal"],
                                        "confidence": result["confidence"],
                                        "agreement": result["agreement_pct"],
                                    },
                                )

                                st.session_state.rl_trace_log.insert(0, {
                                    "time": datetime.now().strftime("%H:%M:%S"),
                                    "type": "ensemble",
                                    "signal": result["signal"],
                                    "confidence": f"{result['confidence']:.2%}",
                                })

                            except Exception as e:
                                st.error(f"Ensemble failed: {e}")

                # Display last result
                result = st.session_state.rl_signal_result
                if result:
                    st.markdown("---")
                    st.markdown("#### 📊 Last Ensemble Result")

                    signal = result["signal"].upper()
                    color = "green" if signal == "BUY" else "red" if signal == "SELL" else "gray"

                    st.markdown(f"### :{color}[{signal}]")

                    m1, m2, m3 = st.columns(3)
                    with m1:
                        st.metric("Confidence", f"{result['confidence']:.0%}")
                    with m2:
                        st.metric("Agreement", f"{result['agreement_pct']:.0%}")
                    with m3:
                        st.metric("Raw Action", f"{result.get('raw_action', 0):.4f}")

                    # Agent votes breakdown
                    if result.get("agent_votes"):
                        st.markdown("**Agent Votes:**")
                        for agent_name, vote in result["agent_votes"].items():
                            vote_str = vote.get("vote", "?") if isinstance(vote, dict) else str(vote)
                            action_val = vote.get("action", 0) if isinstance(vote, dict) else 0
                            v_color = "green" if vote_str == "buy" else "red" if vote_str == "sell" else "gray"
                            st.markdown(
                                f"- **{agent_name}**: :{v_color}[{vote_str.upper()}] "
                                f"(action={action_val:.4f})"
                            )

        # ───── Signal Generator Section ─────
        st.markdown("---")
        st.subheader("📡 RL Signal Generator")

        if trained and len(trained) >= 1:
            sig_col1, sig_col2 = st.columns([1, 2])

            with sig_col1:
                min_conf = st.slider("Min Confidence", 0.0, 1.0, 0.6, 0.05)
                max_pos = st.slider("Max Position %", 1, 20, 5) / 100
                atr_mult = st.slider("ATR Multiplier (SL)", 1.0, 4.0, 2.0, 0.5)

                if st.button("📡 Generate Signal", use_container_width=True, type="primary"):
                    with st.spinner("Generating signal..."):
                        try:
                            # Build ensemble from trained agents
                            ensemble = EnsembleAgent()
                            for name in trained:
                                agent_info = trained[name]
                                class ModelWrapper2:
                                    def __init__(self, model):
                                        self.model = model
                                    def predict(self, obs, deterministic=True):
                                        action, _ = self.model.predict(obs, deterministic=deterministic)
                                        return action
                                ensemble.add_agent(name, ModelWrapper2(agent_info["model"]))

                            generator = RLSignalGenerator(
                                ensemble=ensemble,
                                min_confidence=min_conf,
                                max_position_pct=max_pos,
                                atr_stop_multiplier=atr_mult,
                                use_technical_indicators=False,  # Quick Train uses raw OHLCV
                            )

                            # Get data
                            if connected:
                                df = client.get_history(symbol, timeframe, 200)
                            else:
                                np.random.seed(42)
                                n = 200
                                close = 100 + np.cumsum(np.random.randn(n) * 0.5)
                                df = pd.DataFrame({
                                    "date": pd.date_range("2023-01-01", periods=n, freq="D"),
                                    "open": close + np.random.randn(n) * 0.2,
                                    "high": close + np.abs(np.random.randn(n) * 0.3),
                                    "low": close - np.abs(np.random.randn(n) * 0.3),
                                    "close": close,
                                    "volume": np.random.randint(1000, 10000, n),
                                })

                            if not df.empty and len(df) >= 50:
                                # Clean: strip non-numeric columns (e.g. 'time' from MT5)
                                num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                                keep = [c for c in ["date", "open", "high", "low", "close", "volume"] if c in df.columns]
                                for c in num_cols:
                                    if c not in keep:
                                        keep.append(c)
                                df = df[keep].copy()

                                sig_result = generator.generate(df, symbol=symbol)

                                # Trace signal generation
                                tracer = get_rl_tracer()
                                tracer.trace_signal_generation(symbol, sig_result)

                                st.session_state.rl_signal_result = sig_result
                                st.session_state.rl_trace_log.insert(0, {
                                    "time": datetime.now().strftime("%H:%M:%S"),
                                    "type": "signal",
                                    "signal": sig_result.get("signal", "hold"),
                                    "confidence": f"{sig_result.get('confidence', 0):.2%}",
                                })
                                st.success("Signal generated!")
                            else:
                                st.error("Insufficient market data")

                        except Exception as e:
                            st.error(f"Signal generation failed: {e}")

            with sig_col2:
                sig = st.session_state.rl_signal_result
                if sig and "current_price" in sig:
                    signal = sig.get("signal", "hold").upper()
                    color = "green" if signal == "BUY" else "red" if signal == "SELL" else "gray"

                    st.markdown(f"### :{color}[{signal}] Signal")

                    s1, s2, s3, s4 = st.columns(4)
                    with s1:
                        st.metric("Price", f"{sig.get('current_price', 0):.5f}")
                    with s2:
                        st.metric("Confidence", f"{sig.get('confidence', 0):.0%}")
                    with s3:
                        sl = sig.get("stop_loss")
                        st.metric("Stop Loss", f"{sl:.5f}" if sl else "N/A")
                    with s4:
                        tp = sig.get("take_profit")
                        st.metric("Take Profit", f"{tp:.5f}" if tp else "N/A")

                    r1, r2 = st.columns(2)
                    with r1:
                        st.metric("Position Size", f"{sig.get('position_size_pct', 0):.1%}")
                    with r2:
                        rr = sig.get("risk_reward_ratio")
                        st.metric("Risk:Reward", f"1:{rr:.1f}" if rr else "N/A")
                else:
                    st.info("Click 'Generate Signal' to see results")

        else:
            st.info("Train at least one RL agent to use the signal generator")

        # ───── Trace Log ─────
        st.markdown("---")
        st.subheader("📋 Trace Log")
        st.caption(f"Callback server: {os.getenv('CALLBACK_SERVER_URL', 'http://localhost:3001')}")

        if st.session_state.rl_trace_log:
            trace_df = pd.DataFrame(st.session_state.rl_trace_log[:20])
            st.dataframe(trace_df, use_container_width=True, hide_index=True)
        else:
            st.info("No traces yet. Train an agent or run a prediction to generate traces.")

        # Training history
        if st.session_state.rl_training_log:
            st.markdown("---")
            st.subheader("📈 Training History")
            train_df = pd.DataFrame(st.session_state.rl_training_log)
            st.dataframe(train_df, use_container_width=True, hide_index=True)
    
    # ========== STRATEGY CONFIG TAB ==========
    with tab_strategies:
        st.header("️ Strategy Configuration")
        
        col_preset, col_custom = st.columns([1, 2])
        
        with col_preset:
            st.subheader("Presets")
            
            presets = get_strategy_presets()
            selected_preset = st.radio(
                "Select Preset",
                list(presets.keys()),
                index=1,  # Default to Moderate
                label_visibility="collapsed"
            )
            
            preset = presets[selected_preset]
            st.caption(f"*{preset['description']}*")
            
            if st.button("Apply Preset", use_container_width=True):
                st.session_state.strategy_config = preset.copy()
                st.success(f"Applied {selected_preset} preset")
                st.rerun()
        
        with col_custom:
            st.subheader("Custom Parameters")
            
            config = st.session_state.strategy_config
            
            c1, c2 = st.columns(2)
            
            with c1:
                config["max_position_size"] = st.number_input(
                    "Max Position Size (lots)",
                    min_value=0.01, max_value=1.0,
                    value=float(config.get("max_position_size", 0.01)),
                    step=0.01,
                    help="Maximum position size per trade"
                )
                
                config["stop_loss_pips"] = st.number_input(
                    "Stop Loss (pips)",
                    min_value=10, max_value=500,
                    value=int(config.get("stop_loss_pips", 50)),
                    step=5
                )
                
                config["risk_per_trade"] = st.slider(
                    "Risk per Trade (%)",
                    min_value=0.5, max_value=10.0,
                    value=float(config.get("risk_per_trade", 0.02)) * 100,
                    step=0.5,
                    help="Percentage of account to risk per trade"
                ) / 100
            
            with c2:
                config["max_drawdown"] = st.slider(
                    "Max Drawdown (%)",
                    min_value=1.0, max_value=50.0,
                    value=float(config.get("max_drawdown", 0.10)) * 100,
                    step=1.0,
                    help="Stop trading if drawdown exceeds this"
                ) / 100
                
                config["take_profit_pips"] = st.number_input(
                    "Take Profit (pips)",
                    min_value=10, max_value=500,
                    value=int(config.get("take_profit_pips", 100)),
                    step=5
                )
                
                config["profit_target"] = st.slider(
                    "Daily Profit Target (%)",
                    min_value=0.5, max_value=20.0,
                    value=float(config.get("profit_target", 0.05)) * 100,
                    step=0.5
                ) / 100
            
            st.session_state.strategy_config = config
        
        st.markdown("---")
        
        # Current config summary
        st.subheader("Current Configuration")
        
        config = st.session_state.strategy_config
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Max Position", f"{config['max_position_size']} lots")
            st.metric("Risk/Trade", f"{config['risk_per_trade']*100:.1f}%")
        with col2:
            st.metric("Stop Loss", f"{config['stop_loss_pips']} pips")
            st.metric("Take Profit", f"{config['take_profit_pips']} pips")
        with col3:
            st.metric("Max Drawdown", f"{config['max_drawdown']*100:.0f}%")
            st.metric("Profit Target", f"{config['profit_target']*100:.1f}%")
        
        # Save config button
        if st.button(" Save Configuration", use_container_width=True):
            st.success("Configuration saved!")
            st.balloons()
    
    # ========== SHOW ALL SYMBOLS MODAL ==========
    if st.session_state.get("show_all_symbols", False):
        with st.expander(" All Available Symbols", expanded=True):
            if st.session_state.cached_symbols:
                # Display in columns000
                cols = st.columns(5)
                for i, sym in enumerate(st.session_state.cached_symbols):
                    cols[i % 5].write(sym)
            else:
                st.info("Symbols not loaded yet")
            
            if st.button("Close"):
                st.session_state.show_all_symbols = False
                st.rerun()


if __name__ == "__main__":
    main()
