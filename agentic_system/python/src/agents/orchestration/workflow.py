"""
LangGraph workflow for agent orchestration.

Defines the flow: RL Signal -> Market Analysis -> Risk -> Portfolio -> Execution -> Monitor

The RL signal node is optional — when an RLSignalGenerator is provided,
it runs first and its output is merged with the rule-based market analysis.
RL signals override when confidence > 0.8 (high ensemble agreement).

Token Optimization:
    - Simple graph structure
    - Conditional routing
"""
import logging
from typing import Any, Dict, Literal, Optional

from langgraph.graph import StateGraph, END

from .state import AgentState
from ..specialized import (
    MarketAnalysisAgent,
    RiskManagementAgent,
    PortfolioOptimizationAgent,
    ExecutionAgent,
    PerformanceMonitorAgent,
)

logger = logging.getLogger(__name__)


class TradingWorkflow:
    """
    Multi-agent trading workflow using LangGraph.

    Flow:
        1. Market Analysis -> Generate signal
        2. Risk Management -> Calculate position size
        3. Portfolio Optimization -> Check allocation
        4. Execution -> Execute trade (if approved)
        5. Performance Monitor -> Update metrics

    Example:
        >>> workflow = TradingWorkflow()
        >>> result = await workflow.run({
        ...     "symbol": "EURUSD",
        ...     "price": 1.0850,
        ...     "indicators": {...}
        ... })
    """

    def __init__(
        self,
        llm_provider: Optional[Any] = None,
        rl_signal_generator: Optional[Any] = None,
    ):
        """
        Initialize workflow with agents.

        Args:
            llm_provider: LLM provider for agents (optional)
            rl_signal_generator: RLSignalGenerator for RL-based signals (optional).
                When provided, RL signals are merged with rule-based analysis.
                RL overrides rule-based when confidence > 0.8.
        """
        # Create agents
        self.market_agent = MarketAnalysisAgent(llm_provider)
        self.risk_agent = RiskManagementAgent(llm_provider=llm_provider)
        self.portfolio_agent = PortfolioOptimizationAgent(llm_provider=llm_provider)
        self.execution_agent = ExecutionAgent(llm_provider=llm_provider)
        self.monitor_agent = PerformanceMonitorAgent(llm_provider=llm_provider)

        # Optional RL signal generator
        self.rl_signal_generator = rl_signal_generator

        # Build graph
        self.graph = self._build_graph()

        agent_count = "5 + RL" if rl_signal_generator else "5"
        logger.info(f"TradingWorkflow initialized with {agent_count} agents")

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow."""
        # Create graph with state type
        graph = StateGraph(AgentState)

        # Add RL signal node if generator is available
        if self.rl_signal_generator:
            graph.add_node("rl_signal", self._rl_signal_node)

        # Add nodes for each agent
        graph.add_node("market_analysis", self._market_analysis_node)
        graph.add_node("risk_management", self._risk_management_node)
        graph.add_node("portfolio_optimization", self._portfolio_node)
        graph.add_node("execution", self._execution_node)
        graph.add_node("monitoring", self._monitoring_node)

        # Set entry point
        if self.rl_signal_generator:
            graph.set_entry_point("rl_signal")
            graph.add_edge("rl_signal", "market_analysis")
        else:
            graph.set_entry_point("market_analysis")

        # Add edges
        graph.add_conditional_edges(
            "market_analysis",
            self._should_continue_to_risk,
            {
                "continue": "risk_management",
                "skip": "monitoring",
            },
        )

        graph.add_conditional_edges(
            "risk_management",
            self._should_execute,
            {
                "execute": "portfolio_optimization",
                "skip": "monitoring",
            },
        )

        graph.add_edge("portfolio_optimization", "execution")
        graph.add_edge("execution", "monitoring")
        graph.add_edge("monitoring", END)

        return graph.compile()

    async def _rl_signal_node(self, state: AgentState) -> AgentState:
        """
        RL signal generation step.

        Runs the RLSignalGenerator ensemble to produce RL-based predictions.
        These are stored in state.rl_signal_result and merged with
        rule-based analysis in _market_analysis_node.
        """
        logger.debug("Running RL signal generation...")

        try:
            # Build a minimal DataFrame from state indicators
            import pandas as pd
            import numpy as np

            indicators = state.market.indicators
            price = state.market.current_price

            # Create enough data points for observation
            n_points = max(60, self.rl_signal_generator.window_size + 30)
            df = pd.DataFrame({
                "close": np.full(n_points, price),
                "open": np.full(n_points, price),
                "high": np.full(n_points, price * 1.001),
                "low": np.full(n_points, price * 0.999),
                "volume": np.full(n_points, 10000),
            })

            result = self.rl_signal_generator.generate(
                df=df,
                current_price=price,
                symbol=state.market.symbol,
            )

            state.rl_signal_result = result
            logger.info(
                f"RL signal: {result['signal']} "
                f"(confidence={result['confidence']:.2%})"
            )

        except Exception as e:
            logger.warning(f"RL signal generation failed: {e}")
            state.rl_signal_result = {
                "signal": "hold",
                "confidence": 0.0,
                "error": str(e),
            }

        return state

    async def _market_analysis_node(self, state: AgentState) -> AgentState:
        """Market analysis step — merges with RL signal if available."""
        logger.debug("Running market analysis...")

        result = await self.market_agent.process({
            "symbol": state.market.symbol,
            "current_price": state.market.current_price,
            "indicators": state.market.indicators,
        })

        # Merge with RL signal if available
        rl_result = state.rl_signal_result
        if rl_result and rl_result.get("confidence", 0) > 0:
            rl_confidence = rl_result.get("confidence", 0)
            rule_confidence = result.get("confidence", 0.5)

            # RL overrides rule-based when confidence > 0.8
            if rl_confidence > 0.8:
                result["signal"] = rl_result["signal"]
                result["confidence"] = rl_confidence
                result["source"] = "rl_override"
                logger.info(
                    f"RL override: {rl_result['signal']} "
                    f"(rl={rl_confidence:.2%} > 0.8)"
                )
            else:
                # Weighted average: 40% rule-based + 60% RL
                blended_conf = 0.4 * rule_confidence + 0.6 * rl_confidence
                # Use RL signal if it agrees, else keep rule-based
                if rl_result.get("signal") == result.get("signal", "hold"):
                    result["confidence"] = blended_conf
                    result["source"] = "hybrid_agree"
                else:
                    # Disagreement — use higher confidence signal
                    if rl_confidence > rule_confidence:
                        result["signal"] = rl_result["signal"]
                        result["confidence"] = blended_conf
                        result["source"] = "hybrid_rl"
                    else:
                        result["source"] = "hybrid_rule"

        state.analysis_result = result
        state.market.signal = result.get("signal", "hold")
        state.market.confidence = result.get("confidence", 0.5)
        state.step = 1

        return state

    async def _risk_management_node(self, state: AgentState) -> AgentState:
        """Risk management step."""
        logger.debug("Running risk management...")

        result = await self.risk_agent.process({
            "signal": state.market.signal,
            "confidence": state.market.confidence,
            "current_price": state.market.current_price,
            "account_balance": state.portfolio.balance,
            "atr": state.market.indicators.get("atr_14", state.market.current_price * 0.01),
        })

        state.risk_result = result
        state.should_trade = result.get("approved", False)
        state.step = 2

        return state

    async def _portfolio_node(self, state: AgentState) -> AgentState:
        """Portfolio optimization step."""
        logger.debug("Running portfolio optimization...")

        result = await self.portfolio_agent.process({
            "assets": [state.market.symbol],
            "current_allocations": state.portfolio.holdings,
        })

        state.allocation_result = result
        state.step = 3

        return state

    async def _execution_node(self, state: AgentState) -> AgentState:
        """Execution step."""
        logger.debug("Running execution...")

        if not state.should_trade:
            state.execution_result = {"status": "skipped", "reason": "Not approved"}
            return state

        result = await self.execution_agent.process({
            "action": "buy" if state.market.signal == "buy" else "sell",
            "symbol": state.market.symbol,
            "size": state.risk_result.get("position_size", 0),
            "price": state.market.current_price,
            "stop_loss": state.risk_result.get("stop_loss"),
            "take_profit": state.risk_result.get("take_profit"),
        })

        state.execution_result = result
        state.step = 4

        return state

    async def _monitoring_node(self, state: AgentState) -> AgentState:
        """Performance monitoring step."""
        logger.debug("Running performance monitoring...")

        result = await self.monitor_agent.process({
            "current_balance": state.portfolio.total_value,
        })

        state.monitoring_result = result
        state.step = 5

        return state

    def _should_continue_to_risk(self, state: AgentState) -> Literal["continue", "skip"]:
        """Decide whether to continue to risk management."""
        signal = state.market.signal
        confidence = state.market.confidence

        # Skip if hold signal or low confidence
        if signal == "hold" or confidence < 0.6:
            return "skip"
        return "continue"

    def _should_execute(self, state: AgentState) -> Literal["execute", "skip"]:
        """Decide whether to execute trade."""
        if state.should_trade:
            return "execute"
        return "skip"

    async def run(self, input_data: Dict[str, Any]) -> AgentState:
        """
        Run the complete workflow.

        Args:
            input_data: {
                "symbol": str,
                "price": float,
                "indicators": {...},
                "balance": float
            }

        Returns:
            Final AgentState with all results
        """
        # Initialize state
        state = AgentState()
        state.market.symbol = input_data.get("symbol", "")
        state.market.current_price = input_data.get("price", 0)
        state.market.indicators = input_data.get("indicators", {})
        state.portfolio.balance = input_data.get("balance", 100000)
        state.portfolio.total_value = state.portfolio.balance

        # Run graph
        final_state = await self.graph.ainvoke(state)

        logger.info(
            f"Workflow complete: signal={final_state.market.signal}, "
            f"traded={final_state.should_trade}"
        )

        return final_state


def create_trading_graph(
    llm_provider: Optional[Any] = None,
    rl_signal_generator: Optional[Any] = None,
) -> TradingWorkflow:
    """
    Factory function to create trading workflow.

    Args:
        llm_provider: Optional LLM provider
        rl_signal_generator: Optional RLSignalGenerator for RL-based signals

    Returns:
        Configured TradingWorkflow instance
    """
    return TradingWorkflow(
        llm_provider=llm_provider,
        rl_signal_generator=rl_signal_generator,
    )
