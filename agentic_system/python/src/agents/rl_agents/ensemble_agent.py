"""
Ensemble agent for robust trading signal generation.

Design Decision:
    No single RL algorithm dominates all market regimes.
    PPO performs well in trending markets, SAC adapts to volatility,
    TD3 provides low-variance signals, A2C is fast to adapt, and
    DQN provides clear discrete decisions.

    The EnsembleAgent collects predictions from all trained models
    and uses weighted majority voting to produce a consensus signal.
    Confidence is measured by the agreement ratio among agents.

    This approach is proven in quantitative finance:
    - FinRL uses ensemble methods across PPO/A2C/DDPG
    - Academic research shows ensemble trading outperforms single models
      (Yang et al., 2020, "Deep Reinforcement Learning for Automated
       Stock Trading: An Ensemble Strategy")
"""
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import gymnasium as gym
import numpy as np

from .base_agent import BaseRLAgent

logger = logging.getLogger(__name__)


# Thresholds for signal classification
BUY_THRESHOLD = 0.15
SELL_THRESHOLD = -0.15


class EnsembleAgent:
    """
    Ensemble agent that combines predictions from multiple RL agents.

    Uses weighted majority voting for robust signal generation.
    Confidence = agreement ratio (5/5 agree = 1.0, 3/5 = 0.6).

    Example:
        >>> ensemble = EnsembleAgent()
        >>> ensemble.add_agent("PPO", ppo_agent, weight=1.0)
        >>> ensemble.add_agent("SAC", sac_agent, weight=1.2)
        >>> result = ensemble.predict(observation)
        >>> print(result["signal"], result["confidence"])
    """

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        buy_threshold: float = BUY_THRESHOLD,
        sell_threshold: float = SELL_THRESHOLD,
        min_confidence: float = 0.5,
    ):
        """
        Initialize ensemble agent.

        Args:
            weights: Agent name -> weight mapping (default: equal weights)
            buy_threshold: Action threshold for buy signal (default: 0.15)
            sell_threshold: Action threshold for sell signal (default: -0.15)
            min_confidence: Minimum confidence to emit non-hold signal
        """
        self.default_weights = weights or {}
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.min_confidence = min_confidence

        self._agents: Dict[str, Dict[str, Any]] = {}

        logger.info("EnsembleAgent initialized")

    def add_agent(
        self,
        name: str,
        agent: BaseRLAgent,
        weight: float = 1.0,
    ) -> None:
        """
        Add an RL agent to the ensemble.

        Args:
            name: Agent identifier (e.g., "PPO", "SAC")
            agent: Trained BaseRLAgent instance
            weight: Voting weight (higher = more influence)
        """
        if weight <= 0:
            raise ValueError(f"Weight must be positive, got {weight}")

        self._agents[name] = {
            "agent": agent,
            "weight": self.default_weights.get(name, weight),
        }
        logger.info(f"Added agent '{name}' with weight {weight:.2f}")

    def remove_agent(self, name: str) -> None:
        """Remove an agent from the ensemble."""
        if name in self._agents:
            del self._agents[name]
            logger.info(f"Removed agent '{name}'")

    @property
    def agent_count(self) -> int:
        """Number of agents in ensemble."""
        return len(self._agents)

    @property
    def agent_names(self) -> List[str]:
        """Names of agents in ensemble."""
        return list(self._agents.keys())

    def predict(
        self,
        observation: np.ndarray,
        deterministic: bool = True,
    ) -> Dict[str, Any]:
        """
        Get ensemble prediction with confidence voting.

        Each agent produces an action [-1, 1]. Actions are classified:
            action > buy_threshold  → BUY vote
            action < sell_threshold → SELL vote
            otherwise              → HOLD vote

        Final signal = weighted majority vote.
        Confidence = weighted agreement / total weight.

        Args:
            observation: Current market observation vector
            deterministic: Use deterministic policy for each agent

        Returns:
            {
                "signal": "buy" | "sell" | "hold",
                "confidence": float (0.0-1.0),
                "raw_action": float (weighted average),
                "agent_votes": {name: {"action": float, "vote": str}},
                "agreement_pct": float
            }

        Raises:
            RuntimeError: If no agents are loaded
        """
        if not self._agents:
            raise RuntimeError("No agents in ensemble. Add agents first.")

        # Collect votes
        votes: Dict[str, Dict[str, Any]] = {}
        weighted_actions: List[Tuple[float, float]] = []

        for name, agent_info in self._agents.items():
            agent: BaseRLAgent = agent_info["agent"]
            weight: float = agent_info["weight"]

            try:
                action = agent.predict(observation, deterministic=deterministic)
                action_val = float(np.atleast_1d(action)[0])

                # Classify vote
                if action_val > self.buy_threshold:
                    vote = "buy"
                elif action_val < self.sell_threshold:
                    vote = "sell"
                else:
                    vote = "hold"

                votes[name] = {
                    "action": action_val,
                    "vote": vote,
                    "weight": weight,
                }
                weighted_actions.append((action_val, weight))

            except Exception as e:
                logger.warning(f"Agent '{name}' prediction failed: {e}")
                votes[name] = {
                    "action": 0.0,
                    "vote": "hold",
                    "weight": 0.0,
                    "error": str(e),
                }

        # Calculate weighted votes
        buy_weight = sum(
            v["weight"] for v in votes.values()
            if v["vote"] == "buy" and v["weight"] > 0
        )
        sell_weight = sum(
            v["weight"] for v in votes.values()
            if v["vote"] == "sell" and v["weight"] > 0
        )
        hold_weight = sum(
            v["weight"] for v in votes.values()
            if v["vote"] == "hold" and v["weight"] > 0
        )
        total_weight = buy_weight + sell_weight + hold_weight

        if total_weight == 0:
            return {
                "signal": "hold",
                "confidence": 0.0,
                "raw_action": 0.0,
                "agent_votes": votes,
                "agreement_pct": 0.0,
            }

        # Determine winning signal
        max_weight = max(buy_weight, sell_weight, hold_weight)
        if max_weight == buy_weight:
            signal = "buy"
            confidence = buy_weight / total_weight
        elif max_weight == sell_weight:
            signal = "sell"
            confidence = sell_weight / total_weight
        else:
            signal = "hold"
            confidence = hold_weight / total_weight

        # If confidence too low, default to hold
        if confidence < self.min_confidence and signal != "hold":
            signal = "hold"
            confidence = 1.0 - confidence  # Invert: low agreement = uncertain

        # Weighted average action
        raw_action = (
            sum(a * w for a, w in weighted_actions)
            / sum(w for _, w in weighted_actions)
            if weighted_actions
            else 0.0
        )

        # Agreement percentage (what fraction of agents agree on the signal)
        agreeing = sum(
            1 for v in votes.values()
            if v["vote"] == signal and v["weight"] > 0
        )
        total_voting = sum(1 for v in votes.values() if v["weight"] > 0)
        agreement_pct = agreeing / total_voting if total_voting > 0 else 0.0

        result = {
            "signal": signal,
            "confidence": round(confidence, 4),
            "raw_action": round(raw_action, 6),
            "agent_votes": votes,
            "agreement_pct": round(agreement_pct, 4),
        }

        logger.info(
            f"Ensemble: {signal} (confidence={confidence:.2%}, "
            f"agreement={agreement_pct:.0%}, agents={total_voting})"
        )

        return result

    def evaluate(
        self,
        env: gym.Env,
        n_episodes: int = 5,
        deterministic: bool = True,
    ) -> Dict[str, float]:
        """
        Evaluate ensemble performance on an environment.

        Args:
            env: Trading environment to evaluate on
            n_episodes: Number of evaluation episodes
            deterministic: Use deterministic policy

        Returns:
            {
                "mean_reward": float,
                "std_reward": float,
                "mean_return_pct": float,
                "mean_trades": float,
            }
        """
        episode_rewards = []
        episode_returns = []
        episode_trades = []

        for _ in range(n_episodes):
            obs, info = env.reset()
            done = False
            total_reward = 0.0

            while not done:
                result = self.predict(obs, deterministic=deterministic)

                # Map signal back to continuous action
                if result["signal"] == "buy":
                    action = np.array([max(result["raw_action"], 0.5)])
                elif result["signal"] == "sell":
                    action = np.array([min(result["raw_action"], -0.5)])
                else:
                    action = np.array([0.0])

                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                done = terminated or truncated

            episode_rewards.append(total_reward)
            episode_returns.append(info.get("return_pct", 0.0))
            episode_trades.append(info.get("num_trades", 0))

        return {
            "mean_reward": float(np.mean(episode_rewards)),
            "std_reward": float(np.std(episode_rewards)),
            "mean_return_pct": float(np.mean(episode_returns)),
            "mean_trades": float(np.mean(episode_trades)),
        }
