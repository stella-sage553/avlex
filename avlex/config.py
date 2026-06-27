"""Declarative pipeline configuration.

A :class:`PipelineConfig` names the components to assemble (by their registry
keys) so a whole audio-visual -> LLM pipeline can be described in a small YAML
file and rebuilt reproducibly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from avlex.tasks import Task


@dataclass
class ComponentConfig:
    """A registry ``name`` plus keyword ``options`` for one component."""

    name: str
    options: dict = field(default_factory=dict)

    @classmethod
    def coerce(cls, value: object) -> ComponentConfig:
        """Accept a bare string, a dict, or an existing :class:`ComponentConfig`."""
        if isinstance(value, ComponentConfig):
            return value
        if isinstance(value, str):
            return cls(name=value)
        if isinstance(value, dict):
            return cls(name=value["name"], options=dict(value.get("options", {})))
        raise TypeError(f"cannot read component config from {value!r}")

    def to_dict(self) -> dict:
        return {"name": self.name, "options": dict(self.options)}


def _visual() -> ComponentConfig:
    return ComponentConfig("motion_histogram")


def _audio() -> ComponentConfig:
    return ComponentConfig("mel_stat")


@dataclass
class PipelineConfig:
    """Everything needed to build a :class:`~avlex.pipeline.Pipeline`."""

    visual_encoder: ComponentConfig | None = field(default_factory=_visual)
    audio_encoder: ComponentConfig | None = field(default_factory=_audio)
    fusion: ComponentConfig = field(default_factory=lambda: ComponentConfig("concat"))
    bridge: ComponentConfig = field(default_factory=lambda: ComponentConfig("token"))
    llm: ComponentConfig = field(default_factory=lambda: ComponentConfig("template"))
    task: Task = Task.CAPTION

    @classmethod
    def from_dict(cls, data: dict) -> PipelineConfig:
        """Build a config from a plain dict (e.g. parsed YAML)."""

        def opt(key: str) -> ComponentConfig | None:
            if key not in data or data[key] is None:
                return None
            return ComponentConfig.coerce(data[key])

        kwargs: dict = {}
        if "visual_encoder" in data:
            kwargs["visual_encoder"] = opt("visual_encoder")
        if "audio_encoder" in data:
            kwargs["audio_encoder"] = opt("audio_encoder")
        for key in ("fusion", "bridge", "llm"):
            if key in data:
                kwargs[key] = ComponentConfig.coerce(data[key])
        if "task" in data:
            kwargs["task"] = Task(data["task"])
        return cls(**kwargs)

    def to_dict(self) -> dict:
        """Serialize to a plain dict suitable for YAML/JSON."""
        return {
            "visual_encoder": self.visual_encoder.to_dict()
            if self.visual_encoder
            else None,
            "audio_encoder": self.audio_encoder.to_dict()
            if self.audio_encoder
            else None,
            "fusion": self.fusion.to_dict(),
            "bridge": self.bridge.to_dict(),
            "llm": self.llm.to_dict(),
            "task": self.task.value,
        }

    @classmethod
    def from_yaml(cls, path: str | Path) -> PipelineConfig:
        """Load a config from a YAML file."""
        text = Path(path).read_text(encoding="utf-8")
        data = yaml.safe_load(text) or {}
        return cls.from_dict(data)

    def to_yaml(self, path: str | Path) -> None:
        """Write this config to a YAML file."""
        Path(path).write_text(
            yaml.safe_dump(self.to_dict(), sort_keys=False), encoding="utf-8"
        )
