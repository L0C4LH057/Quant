"""
RL Agent trace wrapper for the Callback Server.

Sends structured trace events (training runs, ensemble predictions, signal
generation) to the callback server so that every RL step is visible in the
trace UI alongside LLM/chain/tool traces.

Since RL agents aren't LangChain chains, we manually construct trace events
that mirror the same schema used by CallbackServerHandler.
"""
import uuid
import time
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)


class RLTraceWrapper:
    """
    Wraps RL agent operations to emit traces to the callback server.

    Example:
        >>> tracer = RLTraceWrapper("http://localhost:3001", project_id=1)
        >>> with tracer.trace_training("PPO", params) as run_id:
        ...     agent.train(50000)
        ...     tracer.log_training_metrics(run_id, {"reward": 150.0})
    """

    def __init__(
        self,
        server_url: str = "http://localhost:3001",
        project_id: Optional[int] = None,
        session_id: Optional[str] = None,
    ):
        self.server_url = server_url
        self.project_id = project_id
        self.session_id = session_id or str(uuid.uuid4())

    def _post(self, endpoint: str, data: Dict[str, Any]) -> bool:
        """Send trace data to callback server. Fails silently."""
        try:
            # Strip None values so the server doesn't overwrite existing fields
            clean_data = {k: v for k, v in data.items() if v is not None}
            resp = requests.post(
                f"{self.server_url}/api/{endpoint}",
                json=clean_data,
                timeout=5,
            )
            return resp.status_code < 400
        except Exception as e:
            logger.debug(f"Trace send failed (non-blocking): {e}")
            return False

    # ─── Training Traces ─────────────────────────────────────

    def start_training_run(
        self,
        algorithm: str,
        config: Dict[str, Any],
        symbol: str = "",
        timesteps: int = 0,
    ) -> str:
        """Start a training run trace. Returns run_id."""
        run_id = str(uuid.uuid4())

        self._post("runs", {
            "id": run_id,
            "name": f"RL Training: {algorithm}",
            "type": "agent",
            "status": "running",
            "startTime": datetime.now().isoformat(),
            "inputs": {
                "algorithm": algorithm,
                "symbol": symbol,
                "timesteps": timesteps,
                "config": config,
            },
            "metadata": {"component": "rl_training"},
            "projectId": self.project_id,
            "sessionId": self.session_id,
        })

        return run_id

    def end_training_run(
        self,
        run_id: str,
        metrics: Dict[str, Any],
        status: str = "completed",
    ):
        """End a training run trace with metrics."""
        payload = {
            "id": run_id,
            "status": status,
            "endTime": datetime.now().isoformat(),
            "outputs": metrics,
        }
        self._post("runs", payload)

    def log_training_step(
        self,
        run_id: str,
        step_name: str,
        inputs: Dict[str, Any],
        outputs: Optional[Dict[str, Any]] = None,
        status: str = "completed",
    ):
        """Log a sub-step within a training run."""
        step_id = str(uuid.uuid4())

        self._post("steps", {
            "id": step_id,
            "runId": run_id,
            "parentId": run_id,
            "type": "tool",
            "name": step_name,
            "status": status,
            "startTime": datetime.now().isoformat(),
            "endTime": datetime.now().isoformat(),
            "inputs": inputs,
            "outputs": outputs or {},
        })

    # ─── Ensemble Prediction Traces ──────────────────────────

    def trace_ensemble_prediction(
        self,
        observation_shape: tuple,
        agent_votes: Dict[str, Any],
        consensus: Dict[str, Any],
    ) -> str:
        """Trace an ensemble prediction with per-agent votes."""
        run_id = str(uuid.uuid4())

        # Create parent run
        self._post("runs", {
            "id": run_id,
            "name": "Ensemble Prediction",
            "type": "agent",
            "status": "running",
            "startTime": datetime.now().isoformat(),
            "inputs": {"observation_shape": str(observation_shape)},
            "metadata": {"component": "ensemble"},
            "projectId": self.project_id,
            "sessionId": self.session_id,
        })

        # Log each agent's vote as a step
        for agent_name, vote_info in agent_votes.items():
            step_id = str(uuid.uuid4())
            self._post("steps", {
                "id": step_id,
                "runId": run_id,
                "parentId": run_id,
                "type": "tool",
                "name": f"Agent: {agent_name}",
                "status": "completed",
                "startTime": datetime.now().isoformat(),
                "endTime": datetime.now().isoformat(),
                "inputs": {"agent": agent_name},
                "outputs": vote_info if isinstance(vote_info, dict) else {"vote": str(vote_info)},
            })

        # Log consensus
        self._post("runs", {
            "id": run_id,
            "status": "completed",
            "endTime": datetime.now().isoformat(),
            "outputs": consensus,
        })

        return run_id

    # ─── Signal Generation Traces ────────────────────────────

    def trace_signal_generation(
        self,
        symbol: str,
        signal_result: Dict[str, Any],
    ) -> str:
        """Trace a full signal generation pipeline."""
        run_id = str(uuid.uuid4())

        self._post("runs", {
            "id": run_id,
            "name": f"RL Signal: {symbol}",
            "type": "agent",
            "status": "running",
            "startTime": datetime.now().isoformat(),
            "inputs": {"symbol": symbol},
            "metadata": {"component": "rl_signal"},
            "projectId": self.project_id,
            "sessionId": self.session_id,
        })

        # Log indicator computation step
        indicators = signal_result.get("indicators", {})
        self.log_training_step(
            run_id,
            "Compute Indicators",
            {"symbol": symbol},
            {"indicators": {k: round(v, 6) if isinstance(v, float) else v for k, v in indicators.items()}},
        )

        # Log ensemble prediction step
        self.log_training_step(
            run_id,
            "Ensemble Prediction",
            {"agent_count": len(signal_result.get("agent_votes", {}))},
            {
                "signal": signal_result.get("signal", "hold"),
                "confidence": signal_result.get("confidence", 0),
                "agent_votes": signal_result.get("agent_votes", {}),
            },
        )

        # Log risk calculation step
        self.log_training_step(
            run_id,
            "Risk Calculation",
            {"current_price": signal_result.get("current_price", 0)},
            {
                "stop_loss": signal_result.get("stop_loss"),
                "take_profit": signal_result.get("take_profit"),
                "position_size_pct": signal_result.get("position_size_pct", 0),
                "risk_reward_ratio": signal_result.get("risk_reward_ratio"),
            },
        )

        # Final output
        safe_result = {}
        for k, v in signal_result.items():
            if isinstance(v, (str, int, bool, type(None))):
                safe_result[k] = v
            elif isinstance(v, float):
                safe_result[k] = round(v, 6)
            elif isinstance(v, dict):
                safe_result[k] = str(v)[:200]
            else:
                safe_result[k] = str(v)[:200]

        self._post("runs", {
            "id": run_id,
            "status": "completed",
            "endTime": datetime.now().isoformat(),
            "outputs": safe_result,
        })

        return run_id

    # ─── Agent Pipeline Traces ───────────────────────────────

    def trace_agent_pipeline(
        self,
        symbol: str,
        analysis_result: Dict[str, Any],
        risk_result: Optional[Dict[str, Any]] = None,
        trade_result: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Trace the full specialized agent pipeline."""
        run_id = str(uuid.uuid4())

        self._post("runs", {
            "id": run_id,
            "name": f"Agent Pipeline: {symbol}",
            "type": "agent",
            "status": "running",
            "startTime": datetime.now().isoformat(),
            "inputs": {"symbol": symbol},
            "metadata": {"component": "agent_pipeline"},
            "projectId": self.project_id,
            "sessionId": self.session_id,
        })

        # Market Analysis step
        self.log_training_step(
            run_id,
            "Market Analysis",
            {"symbol": symbol},
            analysis_result,
        )

        # Risk Management step
        if risk_result:
            self.log_training_step(
                run_id,
                "Risk Management",
                {"signal": analysis_result.get("signal", "hold")},
                risk_result,
            )

        # Execution step
        if trade_result:
            self.log_training_step(
                run_id,
                "Trade Execution",
                {"action": analysis_result.get("signal", "hold")},
                trade_result,
            )

        self._post("runs", {
            "id": run_id,
            "status": "completed",
            "endTime": datetime.now().isoformat(),
            "outputs": {
                "signal": analysis_result.get("signal", "hold"),
                "confidence": analysis_result.get("confidence", 0),
                "traded": trade_result is not None,
            },
        })

        return run_id
