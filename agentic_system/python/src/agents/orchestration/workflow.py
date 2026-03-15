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

from .state import AgentState, create_initial_state
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

    async def _rl_signal_node(self, state: AgentState) -> Dict[str, Any]:
        """
        RL signal generation step.

        Runs the RLSignalGenerator ensemble to produce RL-based predictions.
        These are stored in state.rl_signal_result and merged with
        rule-based analysis in _market_analysis_node.

        BUG-01 fix: Uses real OHLCV data from state["market"]["ohlcv_df"]
        instead of generating a flat constant-price DataFrame which would
        zero out every technical indicator.
        """
        logger.debug("Running RL signal generation...")

        try:
            import pandas as pd
            import numpy as np

            market = state.get("market", {})
            price = market.get("current_price", 0.0)
            ohlcv_df = market.get("ohlcv_df")

            if ohlcv_df is not None and isinstance(ohlcv_df, pd.DataFrame) and not ohlcv_df.empty:
                df = ohlcv_df.copy()
            else:
                # Fallback: generate synthetic data with *some* variance so
                # indicators don't degenerate.  Still not ideal — callers
                # should supply real data via state["market"]["ohlcv_df"].
                n_points = max(60, self.rl_signal_generator.window_size + 30)
                noise = np.random.default_rng(42).normal(0, price * 0.005, n_points).cumsum()
                prices = price + noise
                df = pd.DataFrame({
                    "close": prices,
                    "open": prices * (1 + np.random.default_rng(42).normal(0, 0.001, n_points)),
                    "high": prices * (1 + abs(np.random.default_rng(42).normal(0, 0.002, n_points))),
                    "low": prices * (1 - abs(np.random.default_rng(42).normal(0, 0.002, n_points))),
                    "volume": np.random.default_rng(42).integers(5000, 50000, n_points),
                })
                logger.warning(
                    "No OHLCV data in state — using synthetic fallback. "
                    "Pass real data via state['market']['ohlcv_df'] for accurate signals."
                )

            result = self.rl_signal_generator.generate(
                df=df,
                current_price=price,
                symbol=market.get("symbol", ""),
            )

            logger.info(
                f"RL signal: {result['signal']} "
                f"(confidence={result['confidence']:.2%})"
            )

            return {"rl_signal_result": result}

        except Exception as e:
            logger.warning(f"RL signal generation failed: {e}")
            return {
                "rl_signal_result": {
                    "signal": "hold",
                    "confidence": 0.0,
                    "error": str(e),
                }
            }

    async def _market_analysis_node(self, state: AgentState) -> Dict[str, Any]:
        """Market analysis step — merges with RL signal if available."""
        logger.debug("Running market analysis...")

        market = state.get("market", {})

        result = await self.market_agent.process({
            "symbol": market.get("symbol", ""),
            "current_price": market.get("current_price", 0.0),
            "indicators": market.get("indicators", {}),
        })

        # Merge with RL signal if available
        rl_result = state.get("rl_signal_result", {})
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

        signal = result.get("signal", "hold")
        confidence = result.get("confidence", 0.5)

        return {
            "analysis_result": result,
            "market": {
                **market,
                "signal": signal,
                "confidence": confidence,
            },
            "step": 1,
        }

    async def _risk_management_node(self, state: AgentState) -> Dict[str, Any]:
        """Risk management step."""
        logger.debug("Running risk management...")

        market = state.get("market", {})
        portfolio = state.get("portfolio", {})

        result = await self.risk_agent.process({
            "signal": market.get("signal", "hold"),
            "confidence": market.get("confidence", 0.5),
            "current_price": market.get("current_price", 0.0),
            "account_balance": portfolio.get("balance", 100_000),
            "atr": market.get("indicators", {}).get(
                "atr_14", market.get("current_price", 0.0) * 0.01
            ),
        })

        return {
            "risk_result": result,
            "should_trade": result.get("approved", False),
            "step": 2,
        }

    async def _portfolio_node(self, state: AgentState) -> Dict[str, Any]:
        """Portfolio optimization step."""
        logger.debug("Running portfolio optimization...")

        market = state.get("market", {})
        portfolio = state.get("portfolio", {})

        result = await self.portfolio_agent.process({
            "assets": [market.get("symbol", "")],
            "current_allocations": portfolio.get("holdings", {}),
        })

        return {
            "allocation_result": result,
            "step": 3,
        }

    async def _execution_node(self, state: AgentState) -> Dict[str, Any]:
        """Execution step."""
        logger.debug("Running execution...")

        if not state.get("should_trade", False):
            return {
                "execution_result": {"status": "skipped", "reason": "Not approved"},
                "step": 4,
            }

        market = state.get("market", {})
        risk_result = state.get("risk_result", {})

        result = await self.execution_agent.process({
            "action": "buy" if market.get("signal") == "buy" else "sell",
            "symbol": market.get("symbol", ""),
            "size": risk_result.get("position_size", 0),
            "price": market.get("current_price", 0.0),
            "stop_loss": risk_result.get("stop_loss"),
            "take_profit": risk_result.get("take_profit"),
        })

        return {
            "execution_result": result,
            "step": 4,
        }

    async def _monitoring_node(self, state: AgentState) -> Dict[str, Any]:
        """Performance monitoring step."""
        logger.debug("Running performance monitoring...")

        portfolio = state.get("portfolio", {})

        result = await self.monitor_agent.process({
            "current_balance": portfolio.get("total_value", 0.0),
        })

        return {
            "monitoring_result": result,
            "step": 5,
        }

    def _should_continue_to_risk(self, state: AgentState) -> Literal["continue", "skip"]:
        """Decide whether to continue to risk management."""
        market = state.get("market", {})
        signal = market.get("signal", "hold")
        confidence = market.get("confidence", 0.5)

        # Skip if hold signal or low confidence
        if signal == "hold" or confidence < 0.6:
            return "skip"
        return "continue"

    def _should_execute(self, state: AgentState) -> Literal["execute", "skip"]:
        """Decide whether to execute trade."""
        if state.get("should_trade", False):
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
                "balance": float,
                "ohlcv_df": Optional[pd.DataFrame]  # real OHLCV data for RL
            }

        Returns:
            Final AgentState with all results
        """
        # Initialize state using the helper (TypedDict-compatible)
        state = create_initial_state(
            symbol=input_data.get("symbol", ""),
            price=input_data.get("price", 0.0),
            indicators=input_data.get("indicators", {}),
            balance=input_data.get("balance", 100_000),
            ohlcv_df=input_data.get("ohlcv_df"),
        )

        # Run graph
        final_state = await self.graph.ainvoke(state)

        market = final_state.get("market", {})
        logger.info(
            f"Workflow complete: signal={market.get('signal', 'hold')}, "
            f"traded={final_state.get('should_trade', False)}"
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
