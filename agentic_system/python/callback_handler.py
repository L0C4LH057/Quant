import uuid
import requests
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

class CallbackServerHandler(BaseCallbackHandler):
    """Callback Handler that sends traces to the Callback Server."""

    def __init__(self, server_url: str = "http://localhost:3001", project_id: Optional[int] = None, session_id: Optional[str] = None):
        self.server_url = server_url
        self.project_id = project_id
        self.session_id = session_id
        self.run_id = None
        self.step_ids = {}

    def _post(self, endpoint: str, data: Dict[str, Any]):
        try:
            requests.post(f"{self.server_url}/api/{endpoint}", json=data)
        except Exception as e:
            # Fail silently to not disrupt the agent
            print(f"Failed to send trace to callback server: {e}")

    def on_chain_start(
        self, serialized: Dict[str, Any], inputs: Dict[str, Any], *, run_id: uuid.UUID, parent_run_id: Optional[uuid.UUID] = None, **kwargs: Any
    ) -> Any:
        """Run when chain starts running."""
        # If this is the root chain, create a Run
        if parent_run_id is None:
            self.run_id = str(run_id)
            self._post("runs", {
                "id": self.run_id,
                "name": serialized.get("name", "Agent Run"),
                "type": "agent",
                "status": "running",
                "startTime": datetime.now().isoformat(),
                "inputs": inputs,
                "metadata": kwargs.get("metadata"),
                "projectId": self.project_id,
                "sessionId": self.session_id
            })
        else:
            # It's a step (sub-chain)
            self._post("steps", {
                "id": str(run_id),
                "runId": self.run_id,
                "parentId": str(parent_run_id),
                "type": "chain",
                "name": serialized.get("name", "Chain"),
                "status": "running",
                "startTime": datetime.now().isoformat(),
                "inputs": inputs,
                "metadata": kwargs.get("metadata")
            })

    def on_chain_end(self, outputs: Dict[str, Any], *, run_id: uuid.UUID, parent_run_id: Optional[uuid.UUID] = None, **kwargs: Any) -> Any:
        """Run when chain ends running."""
        if parent_run_id is None:
            self._post("runs", {
                "id": str(run_id),
                "status": "completed",
                "endTime": datetime.now().isoformat(),
                "outputs": outputs
            })
        else:
            self._post("steps", {
                "id": str(run_id),
                "status": "completed",
                "endTime": datetime.now().isoformat(),
                "outputs": outputs
            })

    def on_chain_error(self, error: Union[Exception, KeyboardInterrupt], *, run_id: uuid.UUID, parent_run_id: Optional[uuid.UUID] = None, **kwargs: Any) -> Any:
        """Run when chain errors."""
        if parent_run_id is None:
            self._post("runs", {
                "id": str(run_id),
                "status": "failed",
                "endTime": datetime.now().isoformat(),
                "error": str(error)
            })
        else:
            self._post("steps", {
                "id": str(run_id),
                "status": "failed",
                "endTime": datetime.now().isoformat(),
                "error": str(error)
            })

    def on_tool_start(
        self, serialized: Dict[str, Any], input_str: str, *, run_id: uuid.UUID, parent_run_id: Optional[uuid.UUID] = None, **kwargs: Any
    ) -> Any:
        """Run when tool starts running."""
        self._post("steps", {
            "id": str(run_id),
            "runId": self.run_id,
            "parentId": str(parent_run_id),
            "type": "tool",
            "name": serialized.get("name", "Tool"),
            "status": "running",
            "startTime": datetime.now().isoformat(),
            "inputs": {"input": input_str},
            "metadata": kwargs.get("metadata")
        })

    def on_tool_end(self, output: str, *, run_id: uuid.UUID, parent_run_id: Optional[uuid.UUID] = None, **kwargs: Any) -> Any:
        """Run when tool ends running."""
        self._post("steps", {
            "id": str(run_id),
            "status": "completed",
            "endTime": datetime.now().isoformat(),
            "outputs": {"output": output}
        })

    def on_tool_error(self, error: Union[Exception, KeyboardInterrupt], *, run_id: uuid.UUID, parent_run_id: Optional[uuid.UUID] = None, **kwargs: Any) -> Any:
        """Run when tool errors."""
        self._post("steps", {
            "id": str(run_id),
            "status": "failed",
            "endTime": datetime.now().isoformat(),
            "error": str(error)
        })

    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], *, run_id: uuid.UUID, parent_run_id: Optional[uuid.UUID] = None, **kwargs: Any
    ) -> Any:
        """Run when LLM starts running."""
        self._post("steps", {
            "id": str(run_id),
            "runId": self.run_id,
            "parentId": str(parent_run_id),
            "type": "llm",
            "name": "LLM",
            "status": "running",
            "startTime": datetime.now().isoformat(),
            "inputs": {"prompts": prompts},
            "metadata": kwargs.get("metadata")
        })

    def on_llm_end(self, response: LLMResult, *, run_id: uuid.UUID, parent_run_id: Optional[uuid.UUID] = None, **kwargs: Any) -> Any:
        """Run when LLM ends running."""
        self._post("steps", {
            "id": str(run_id),
            "status": "completed",
            "endTime": datetime.now().isoformat(),
            "outputs": {"generations": [[g.text for g in gen] for gen in response.generations]}
        })

    def on_llm_error(self, error: Union[Exception, KeyboardInterrupt], *, run_id: uuid.UUID, parent_run_id: Optional[uuid.UUID] = None, **kwargs: Any) -> Any:
        """Run when LLM errors."""
        self._post("steps", {
            "id": str(run_id),
            "status": "failed",
            "endTime": datetime.now().isoformat(),
            "error": str(error)
        })
