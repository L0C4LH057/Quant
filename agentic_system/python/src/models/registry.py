"""
Model Registry — centralised model versioning and lifecycle management.

GAP-01 fix: Provides save / load / list / promote API so that trained models
are tracked with metadata instead of relying on raw file paths.

Usage:
    >>> registry = ModelRegistry()
    >>> registry.register("ppo_eurusd", model, {"sharpe": 1.5, "return": 0.12})
    >>> model = registry.load("ppo_eurusd", version="latest")
"""
import json
import logging
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from stable_baselines3.common.base_class import BaseAlgorithm

from ..config.base import get_config

logger = logging.getLogger(__name__)


@dataclass
class ModelVersion:
    """Metadata for a single model version."""

    version: int
    algorithm: str
    created_at: str
    metrics: Dict[str, float] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    path: str = ""
    promoted: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ModelRegistry:
    """
    File-system-backed model registry.

    Layout::

        storage/rl_models/registry/
          └─ <model_name>/
              ├─ manifest.json        # list of versions
              ├─ v1/
              │   └─ model.zip
              ├─ v2/
              │   └─ model.zip
              └─ promoted/
                  └─ model.zip        # symlink / copy of best version
    """

    def __init__(self, base_path: Optional[Path] = None):
        config = get_config()
        self.base_path = base_path or (config.model_save_path / "registry")
        self.base_path.mkdir(parents=True, exist_ok=True)

    # ── Write operations ─────────────────────────────────────────────────

    def register(
        self,
        name: str,
        model: BaseAlgorithm,
        metrics: Optional[Dict[str, float]] = None,
        tags: Optional[List[str]] = None,
    ) -> ModelVersion:
        """Save a new version of *name*."""
        model_dir = self.base_path / name
        model_dir.mkdir(parents=True, exist_ok=True)

        manifest = self._load_manifest(name)
        next_version = len(manifest) + 1

        version_dir = model_dir / f"v{next_version}"
        version_dir.mkdir(parents=True, exist_ok=True)
        model_path = version_dir / "model.zip"
        model.save(str(model_path))

        entry = ModelVersion(
            version=next_version,
            algorithm=type(model).__name__,
            created_at=datetime.now().isoformat(),
            metrics=metrics or {},
            tags=tags or [],
            path=str(model_path),
        )
        manifest.append(entry)
        self._save_manifest(name, manifest)

        logger.info(f"Registered {name} v{next_version} — metrics={metrics}")
        return entry

    def promote(self, name: str, version: Optional[int] = None) -> ModelVersion:
        """
        Mark a version as promoted (production-ready).

        If *version* is ``None`` the latest version is promoted.
        """
        manifest = self._load_manifest(name)
        if not manifest:
            raise ValueError(f"No versions found for {name}")

        target = manifest[-1] if version is None else self._find_version(manifest, version)

        # Copy model to promoted/
        promoted_dir = self.base_path / name / "promoted"
        promoted_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(target.path, promoted_dir / "model.zip")

        # Mark in manifest
        for v in manifest:
            v.promoted = v.version == target.version
        self._save_manifest(name, manifest)

        logger.info(f"Promoted {name} v{target.version}")
        return target

    # ── Read operations ──────────────────────────────────────────────────

    def load(
        self,
        name: str,
        version: str = "latest",
        algorithm_class: Optional[type] = None,
    ) -> BaseAlgorithm:
        """
        Load a model.

        Args:
            name: Registered model name.
            version: ``"latest"``, ``"promoted"``, or an integer version string.
            algorithm_class: SB3 algorithm class.  Auto-detected when possible.
        """
        manifest = self._load_manifest(name)
        if not manifest:
            raise ValueError(f"No versions found for {name}")

        if version == "promoted":
            path = self.base_path / name / "promoted" / "model.zip"
            if not path.exists():
                raise FileNotFoundError(f"No promoted model for {name}")
            entry = next((v for v in manifest if v.promoted), manifest[-1])
        elif version == "latest":
            entry = manifest[-1]
            path = Path(entry.path)
        else:
            entry = self._find_version(manifest, int(version))
            path = Path(entry.path)

        if algorithm_class is None:
            algorithm_class = self._resolve_algorithm(entry.algorithm)

        return algorithm_class.load(str(path))

    def list_models(self) -> List[str]:
        """Return all registered model names."""
        return sorted(
            d.name for d in self.base_path.iterdir()
            if d.is_dir() and (d / "manifest.json").exists()
        )

    def list_versions(self, name: str) -> List[ModelVersion]:
        """Return all versions for a given model."""
        return self._load_manifest(name)

    # ── Internal helpers ─────────────────────────────────────────────────

    def _load_manifest(self, name: str) -> List[ModelVersion]:
        path = self.base_path / name / "manifest.json"
        if not path.exists():
            return []
        data = json.loads(path.read_text())
        return [ModelVersion(**d) for d in data]

    def _save_manifest(self, name: str, manifest: List[ModelVersion]) -> None:
        path = self.base_path / name / "manifest.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps([v.to_dict() for v in manifest], indent=2))

    @staticmethod
    def _find_version(manifest: List[ModelVersion], version: int) -> ModelVersion:
        for v in manifest:
            if v.version == version:
                return v
        raise ValueError(f"Version {version} not found")

    @staticmethod
    def _resolve_algorithm(algorithm_name: str) -> type:
        from stable_baselines3 import A2C, DQN, PPO, SAC, TD3

        mapping = {"PPO": PPO, "SAC": SAC, "A2C": A2C, "TD3": TD3, "DQN": DQN}
        cls = mapping.get(algorithm_name)
        if cls is None:
            raise ValueError(
                f"Unknown algorithm: {algorithm_name}. "
                f"Pass algorithm_class explicitly."
            )
        return cls
