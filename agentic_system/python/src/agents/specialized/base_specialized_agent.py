"""
Base class for LLM-powered specialized agents.

Token Optimization:
    - Concise prompt templates
    - Structured output for parsing
    - State management for context
"""
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class AgentMessage:
    """
    Message format for inter-agent communication.

    Token Optimization:
        - Minimal structure
        - Clear type markers
    """

    sender: str
    receiver: str
    content: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    message_type: str = "info"  # info, request, response, alert

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "sender": self.sender,
            "receiver": self.receiver,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "message_type": self.message_type,
        }


class BaseSpecializedAgent(ABC):
    """
    Base class for specialized trading agents.

    Each agent has:
        - Unique role and responsibilities
        - State management
        - LLM integration for reasoning
        - Communication with other agents

    Token Optimization:
        - Minimal system prompts
        - Structured inputs/outputs
        - Context window management
    """

    def __init__(
        self,
        name: str,
        role: str,
        llm_provider: Optional[Any] = None,
    ):
        """
        Initialize agent.

        Args:
            name: Agent identifier
            role: Agent role description
            llm_provider: LLM provider instance
        """
        self.name = name
        self.role = role
        self.llm_provider = llm_provider

        # State
        self.state: Dict[str, Any] = {}
        self.message_history: List[AgentMessage] = []

        logger.info(f"Initialized {name} agent with role: {role}")

    @property
    @abstractmethod
    def system_prompt(self) -> str:
        """System prompt for LLM (keep concise for token optimization)."""
        pass

    @abstractmethod
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process input and produce output.

        Args:
            input_data: Input data dictionary

        Returns:
            Output data dictionary
        """
        pass

    def update_state(self, key: str, value: Any) -> None:
        """Update agent state."""
        self.state[key] = value

    def get_state(self, key: str, default: Any = None) -> Any:
        """Get value from agent state."""
        return self.state.get(key, default)

    def send_message(
        self,
        receiver: str,
        content: Dict[str, Any],
        message_type: str = "info",
    ) -> AgentMessage:
        """
        Create a message to another agent.

        Args:
            receiver: Receiver agent name
            content: Message content
            message_type: Type of message

        Returns:
            Created message
        """
        message = AgentMessage(
            sender=self.name,
            receiver=receiver,
            content=content,
            message_type=message_type,
        )
        self.message_history.append(message)
        return message

    def receive_message(self, message: AgentMessage) -> None:
        """Process received message."""
        self.message_history.append(message)
        logger.debug(f"{self.name} received message from {message.sender}")

    async def _call_llm(
        self,
        user_prompt: str,
        max_tokens: int = 500,
    ) -> str:
        """
        Call LLM with prompt.

        Token Optimization:
            - Uses concise system prompt
            - Limits max_tokens
            - Structured output format
        """
        if self.llm_provider is None:
            logger.warning(f"{self.name}: No LLM provider configured")
            return ""

        try:
            response = await self.llm_provider.generate(
                system_prompt=self.system_prompt,
                user_prompt=user_prompt,
                max_tokens=max_tokens,
            )
            return response
        except Exception as e:
            logger.error(f"{self.name} LLM error: {e}")
            return ""

    def get_context_summary(self, max_messages: int = 5) -> str:
        """
        Get summary of recent context.

        Token Optimization:
            - Limits history
            - Concise format
        """
        recent = self.message_history[-max_messages:]
        if not recent:
            return "No recent messages."

        lines = []
        for msg in recent:
            lines.append(f"[{msg.sender}->{msg.receiver}] {msg.message_type}")
        return "\n".join(lines)
