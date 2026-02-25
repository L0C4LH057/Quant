"""
PipFlow AI - Agent Testing Dashboard

Interactive Streamlit interface for testing trading agents
and strategies before deployment.

Run with: streamlit run scripts/testing_dashboard.py
"""
import asyncio
import sys
import os
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import List, Dict, Any

# Setup path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from dotenv import load_dotenv
env_path = os.path.join(os.path.dirname(project_root), '.env')
load_dotenv(env_path)

import streamlit as st
import pandas as pd
import numpy as np

# Page config
st.set_page_config(
    page_title="PipFlow AI - Testing Dashboard",
    page_icon="📈",
    layout="wide",
)

# Custom CSS
st.markdown("""
<style>
    .stMetric {
        background: linear-gradient(135deg, #1e3a5f 0%, #2d5a87 100%);
        padding: 15px;
        border-radius: 10px;
    }
    .signal-buy { color: #00ff88; font-weight: bold; }
    .signal-sell { color: #ff4444; font-weight: bold; }
    .signal-hold { color: #ffaa00; font-weight: bold; }
    .trace-step {
        background: #1a1a2e;
        border-left: 3px solid #4ecdc4;
        padding: 10px;
        margin: 5px 0;
        border-radius: 5px;
    }
    .token-badge {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 5px 10px;
        border-radius: 20px;
        font-size: 0.8em;
    }
</style>
""", unsafe_allow_html=True)


# ============= AGENT TRACER =============
@dataclass
class TraceStep:
    """Single step in agent execution trace."""
    timestamp: datetime
    agent: str
    action: str
    input_summary: str
    output_summary: str
    tokens_used: int = 0
    duration_ms: float = 0


class AgentTracer:
    """Track agent execution for debugging and monitoring."""

    def __init__(self):
        self.traces: List[TraceStep] = []
        self.total_tokens = 0
        self.start_time = None

    def start(self):
        """Start new trace session."""
        self.traces = []
        self.total_tokens = 0
        self.start_time = datetime.now()

    def add_step(
        self,
        agent: str,
        action: str,
        input_data: Dict[str, Any],
        output_data: Dict[str, Any],
        tokens: int = 0,
        duration_ms: float = 0
    ):
        """Add a step to the trace."""
        # Create summary (max 100 chars)
        input_summary = str(input_data)[:100]
        output_summary = str(output_data)[:100]

        step = TraceStep(
            timestamp=datetime.now(),
            agent=agent,
            action=action,
            input_summary=input_summary,
            output_summary=output_summary,
            tokens_used=tokens,
            duration_ms=duration_ms,
        )
        self.traces.append(step)
        self.total_tokens += tokens

    def get_summary(self) -> Dict[str, Any]:
        """Get trace summary."""
        return {
            "total_steps": len(self.traces),
            "total_tokens": self.total_tokens,
            "agents_used": list(set(t.agent for t in self.traces)),
            "total_duration_ms": sum(t.duration_ms for t in self.traces),
        }


# Global tracer instance
if "tracer" not in st.session_state:
    st.session_state.tracer = AgentTracer()

if "token_history" not in st.session_state:
    st.session_state.token_history = []


def run_async(coro):
    """Helper to run async code in Streamlit."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def get_cached_llm():
    """Get LLM provider (not cached to track tokens)."""
    from src.llm.provider import DeepSeekProvider
    return DeepSeekProvider()


def get_agents(llm=None):
    """Get agent instances with provided LLM."""
    from src.agents.specialized import (
        MarketAnalysisAgent,
        RiskManagementAgent,
        PortfolioOptimizationAgent,
        PerformanceMonitorAgent,
    )

    if llm is None:
        llm = get_cached_llm()

    return {
        "market_analysis": MarketAnalysisAgent(llm_provider=llm),
        "risk_management": RiskManagementAgent(llm_provider=llm),
        "portfolio": PortfolioOptimizationAgent(llm_provider=llm),
        "monitor": PerformanceMonitorAgent(llm_provider=llm),
    }


def display_traces():
    """Display agent execution traces in sidebar."""
    tracer = st.session_state.tracer

    if tracer.traces:
        st.sidebar.divider()
        st.sidebar.subheader("🔍 Agent Traces")

        # Token summary
        col1, col2 = st.sidebar.columns(2)
        col1.metric("Total Tokens", tracer.total_tokens)
        col2.metric("Steps", len(tracer.traces))

        # Show each trace step
        for i, step in enumerate(tracer.traces[-5:], 1):  # Last 5 steps
            with st.sidebar.expander(f"Step {i}: {step.agent}", expanded=False):
                st.caption(f"⏱️ {step.duration_ms:.0f}ms | 🎯 {step.tokens_used} tokens")
                st.markdown(f"**Action:** {step.action}")
                st.markdown(f"**Input:** `{step.input_summary}...`")
                st.markdown(f"**Output:** `{step.output_summary}...`")


def main():
    st.title("📈 PipFlow AI - Testing Dashboard")
    st.markdown("*Test trading agents and strategies before deployment*")

    # Get LLM for token tracking
    llm = get_cached_llm()
    tracer = st.session_state.tracer

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")

        use_llm = st.checkbox("Use DeepSeek LLM", value=True)
        st.caption("Enable for AI-enhanced analysis")

        st.divider()

        symbol = st.selectbox(
            "Symbol",
            ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD"],
            index=0,
        )

        account_balance = st.number_input(
            "Account Balance ($)",
            min_value=1000,
            max_value=10000000,
            value=100000,
            step=10000,
        )

        st.divider()
        api_status = "✅ Connected" if llm.api_key else "❌ Not configured"
        st.caption(f"DeepSeek API: {api_status}")
        st.caption(f"Model: `{llm.model}`")

        # Show token usage
        st.divider()
        st.subheader("📊 Token Usage")
        st.metric("Session Tokens", llm.total_tokens_used)

        if st.button("🗑️ Clear Traces"):
            tracer.start()
            st.rerun()

    # Display traces in sidebar
    display_traces()

    # Main tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🔍 Market Analysis",
        "⚠️ Risk Management",
        "📊 Full Workflow",
        "📈 Backtest Simulator",
        "📋 Trace Log",
    ])

    # Tab 1: Market Analysis
    with tab1:
        st.header("Market Analysis Agent")
        st.markdown("Test market analysis with custom indicators")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📊 Input Indicators")

            current_price = st.number_input("Current Price", value=1.0850, format="%.5f")
            rsi = st.slider("RSI (14)", 0, 100, 45)
            macd = st.number_input("MACD", value=0.0015, format="%.5f")
            macd_signal = st.number_input("MACD Signal", value=-0.001, format="%.5f")
            sma_20 = st.number_input("SMA 20", value=current_price - 0.003, format="%.5f")
            sma_50 = st.number_input("SMA 50", value=current_price - 0.007, format="%.5f")

        with col2:
            st.subheader("🎯 Analysis Result")

            if st.button("Run Analysis", type="primary", key="analyze"):
                tracer.start()
                start_time = datetime.now()

                with st.spinner("Analyzing..."):
                    input_data = {
                        "symbol": symbol,
                        "current_price": current_price,
                        "indicators": {
                            "rsi_14": rsi,
                            "macd": macd,
                            "macd_signal": macd_signal,
                            "sma_20": sma_20,
                            "sma_50": sma_50,
                            "close": current_price,
                        },
                        "use_llm": use_llm,
                    }

                    agents = get_agents(llm)
                    tokens_before = llm.total_tokens_used

                    result = run_async(agents["market_analysis"].process(input_data))

                    tokens_used = llm.total_tokens_used - tokens_before
                    duration = (datetime.now() - start_time).total_seconds() * 1000

                    # Add to trace
                    tracer.add_step(
                        agent="MarketAnalysis",
                        action="analyze",
                        input_data=input_data,
                        output_data=result,
                        tokens=tokens_used,
                        duration_ms=duration,
                    )

                    # Display result
                    signal = result["signal"].upper()
                    signal_color = {"BUY": "🟢", "SELL": "🔴", "HOLD": "🟡"}.get(signal, "⚪")

                    st.markdown(f"### {signal_color} Signal: **{signal}**")

                    col_a, col_b, col_c = st.columns(3)
                    col_a.metric("Confidence", f"{result['confidence']:.1%}")
                    col_b.metric("Analysis Type", result.get("analysis_type", "rule_based"))
                    col_c.metric("Tokens Used", tokens_used)

                    st.markdown(f"**Reason:** {result['reason']}")

                    st.success(f"✅ Completed in {duration:.0f}ms")

                    with st.expander("Raw Result"):
                        st.json(result)

    # Tab 2: Risk Management
    with tab2:
        st.header("Risk Management Agent")
        st.markdown("Calculate position sizes and risk parameters")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📥 Trade Parameters")

            signal_input = st.selectbox("Signal", ["buy", "sell", "hold"])
            confidence_input = st.slider("Confidence", 0.0, 1.0, 0.75)
            price_input = st.number_input("Entry Price", value=1.0850, format="%.5f")
            atr_input = st.number_input("ATR (Volatility)", value=0.0050, format="%.5f")

        with col2:
            st.subheader("📤 Risk Calculation")

            if st.button("Calculate Risk", type="primary", key="risk"):
                start_time = datetime.now()

                with st.spinner("Calculating..."):
                    input_data = {
                        "signal": signal_input,
                        "confidence": confidence_input,
                        "current_price": price_input,
                        "account_balance": account_balance,
                        "atr": atr_input,
                    }

                    agents = get_agents(llm)
                    tokens_before = llm.total_tokens_used

                    result = run_async(agents["risk_management"].process(input_data))

                    tokens_used = llm.total_tokens_used - tokens_before
                    duration = (datetime.now() - start_time).total_seconds() * 1000

                    # Add to trace
                    tracer.add_step(
                        agent="RiskManagement",
                        action="calculate_risk",
                        input_data=input_data,
                        output_data=result,
                        tokens=tokens_used,
                        duration_ms=duration,
                    )

                    if result.get("approved"):
                        st.success("✅ Trade Approved")

                        col_a, col_b, col_c = st.columns(3)
                        col_a.metric("Position Size", f"{result['position_size']:.4f}")
                        col_b.metric("Stop Loss", f"{result['stop_loss']:.5f}")
                        col_c.metric("Take Profit", f"{result['take_profit']:.5f}")

                        st.metric("Risk Amount", f"${result['risk_amount']:.2f}")
                        st.metric("Risk %", f"{result.get('risk_pct', 0):.2f}%")
                    else:
                        st.warning(f"⚠️ Trade Not Approved: {result.get('reason', 'Unknown')}")

                    st.caption(f"⏱️ {duration:.0f}ms | 🎯 {tokens_used} tokens")

                    with st.expander("Raw Result"):
                        st.json(result)

    # Tab 3: Full Workflow
    with tab3:
        st.header("Full Agent Workflow")
        st.markdown("Run the complete trading workflow through all agents")

        col1, col2 = st.columns([1, 2])

        with col1:
            st.subheader("📊 Market Data")

            wf_price = st.number_input("Price", value=1.0850, format="%.5f", key="wf_price")
            wf_rsi = st.slider("RSI", 0, 100, 35, key="wf_rsi")
            wf_macd = st.number_input("MACD", value=0.002, format="%.5f", key="wf_macd")

            run_workflow = st.button("🚀 Run Full Workflow", type="primary")

        with col2:
            if run_workflow:
                tracer.start()
                start_time = datetime.now()

                with st.spinner("Running workflow..."):
                    from src.agents.orchestration import TradingWorkflow

                    workflow_llm = llm if use_llm else None
                    workflow = TradingWorkflow(llm_provider=workflow_llm)

                    tokens_before = llm.total_tokens_used

                    input_data = {
                        "symbol": symbol,
                        "price": wf_price,
                        "indicators": {
                            "rsi_14": wf_rsi,
                            "macd": wf_macd,
                            "macd_signal": wf_macd - 0.001,
                            "sma_20": wf_price - 0.002,
                            "sma_50": wf_price - 0.005,
                            "close": wf_price,
                            "atr_14": 0.005,
                        },
                        "balance": account_balance,
                    }

                    result = run_async(workflow.run(input_data))

                    tokens_used = llm.total_tokens_used - tokens_before
                    duration = (datetime.now() - start_time).total_seconds() * 1000

                    # Show workflow steps
                    st.subheader("Workflow Results")

                    steps = [
                        ("Market Analysis", result.analysis_result),
                        ("Risk Management", result.risk_result),
                        ("Portfolio", result.allocation_result),
                        ("Execution", result.execution_result),
                        ("Monitoring", result.monitoring_result),
                    ]

                    for i, (step_name, step_result) in enumerate(steps, 1):
                        if step_result:
                            st.markdown(f"✅ **Step {i}: {step_name}**")
                            tracer.add_step(
                                agent=step_name,
                                action="process",
                                input_data={"step": i},
                                output_data=step_result,
                                duration_ms=duration / len(steps),
                            )
                        else:
                            st.markdown(f"⏭️ Step {i}: {step_name} - Skipped")

                    st.divider()

                    # Summary
                    col_a, col_b, col_c, col_d = st.columns(4)
                    col_a.metric("Signal", result.market.signal.upper())
                    col_b.metric("Confidence", f"{result.market.confidence:.1%}")
                    col_c.metric("Trade Executed", "Yes" if result.should_trade else "No")
                    col_d.metric("Total Tokens", tokens_used)

                    st.success(f"✅ Workflow completed in {duration:.0f}ms")

                    if result.monitoring_result:
                        with st.expander("Performance Metrics"):
                            st.json(result.monitoring_result.get("metrics", {}))

    # Tab 4: Backtest Simulator
    with tab4:
        st.header("Backtest Simulator")
        st.markdown("Simulate strategy on historical data")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("⚙️ Backtest Settings")

            bt_days = st.slider("Days of Data", 30, 365, 100)
            bt_initial = st.number_input("Initial Balance", value=100000, step=10000)

            strategy = st.selectbox(
                "Strategy",
                ["RSI Mean Reversion", "SMA Crossover", "Random Baseline"]
            )

        with col2:
            if st.button("▶️ Run Backtest", type="primary"):
                with st.spinner("Running backtest simulation..."):
                    # Generate synthetic data for demo
                    np.random.seed(42)
                    dates = pd.date_range(end=datetime.now(), periods=bt_days, freq='D')
                    prices = 1.08 + np.cumsum(np.random.randn(bt_days) * 0.001)

                    # Simple strategy simulation
                    balance = bt_initial
                    position = 0
                    trades = []
                    equity_curve = [balance]

                    for i in range(30, len(prices)):
                        price = prices[i]

                        # RSI-like signal
                        returns = np.diff(prices[i-14:i+1])
                        gains = returns[returns > 0].sum()
                        losses = abs(returns[returns < 0].sum())
                        rsi = 100 - (100 / (1 + gains / (losses + 1e-10)))

                        if strategy == "RSI Mean Reversion":
                            if rsi < 30 and position == 0:  # Buy
                                position = balance * 0.1 / price
                                balance -= position * price
                                trades.append(("buy", price))
                            elif rsi > 70 and position > 0:  # Sell
                                balance += position * price
                                position = 0
                                trades.append(("sell", price))

                        elif strategy == "SMA Crossover":
                            sma_10 = prices[i-10:i].mean()
                            sma_30 = prices[i-30:i].mean()
                            if sma_10 > sma_30 and position == 0:
                                position = balance * 0.1 / price
                                balance -= position * price
                            elif sma_10 < sma_30 and position > 0:
                                balance += position * price
                                position = 0

                        equity_curve.append(balance + position * price)

                    final_equity = equity_curve[-1]
                    total_return = (final_equity - bt_initial) / bt_initial

                    # Show results
                    st.subheader("📊 Results")

                    col_a, col_b, col_c = st.columns(3)
                    col_a.metric("Final Equity", f"${final_equity:,.2f}")
                    col_b.metric("Total Return", f"{total_return:.2%}")
                    col_c.metric("Total Trades", len(trades))

                    # Equity curve
                    st.line_chart(pd.DataFrame({
                        "Equity": equity_curve
                    }))

                    st.caption(f"Simulated {bt_days} days with {strategy} strategy")

    # Tab 5: Trace Log
    with tab5:
        st.header("📋 Agent Execution Log")
        st.markdown("Detailed log of all agent executions")

        if tracer.traces:
            # Summary metrics
            summary = tracer.get_summary()

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Steps", summary["total_steps"])
            col2.metric("Total Tokens", summary["total_tokens"])
            col3.metric("Total Duration", f"{summary['total_duration_ms']:.0f}ms")
            col4.metric("Agents Used", len(summary["agents_used"]))

            st.divider()

            # Detailed trace table
            trace_data = []
            for step in tracer.traces:
                trace_data.append({
                    "Time": step.timestamp.strftime("%H:%M:%S"),
                    "Agent": step.agent,
                    "Action": step.action,
                    "Tokens": step.tokens_used,
                    "Duration (ms)": f"{step.duration_ms:.0f}",
                    "Input": step.input_summary[:50] + "...",
                    "Output": step.output_summary[:50] + "...",
                })

            st.dataframe(pd.DataFrame(trace_data), use_container_width=True)

            # Export button
            if st.button("📥 Export Traces as JSON"):
                import json
                traces_json = json.dumps([{
                    "timestamp": t.timestamp.isoformat(),
                    "agent": t.agent,
                    "action": t.action,
                    "tokens": t.tokens_used,
                    "duration_ms": t.duration_ms,
                    "input": t.input_summary,
                    "output": t.output_summary,
                } for t in tracer.traces], indent=2)

                st.download_button(
                    "Download JSON",
                    traces_json,
                    "agent_traces.json",
                    "application/json"
                )
        else:
            st.info("No traces yet. Run some agent actions to see the execution log.")


if __name__ == "__main__":
    main()
