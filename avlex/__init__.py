"""avlex: bridge audio-visual encoders to LLMs for video captioning and understanding."""

from avlex._version import __version__
from avlex.clip import ClipFeatures, synthetic_clip
from avlex.config import ComponentConfig, PipelineConfig
from avlex.pipeline import Pipeline
from avlex.result import PipelineResult
from avlex.tasks import Task
from avlex.types import Modality

__all__ = [
    "__version__",
    "ClipFeatures",
    "synthetic_clip",
    "PipelineConfig",
    "ComponentConfig",
    "Pipeline",
    "PipelineResult",
    "Task",
    "Modality",
]
