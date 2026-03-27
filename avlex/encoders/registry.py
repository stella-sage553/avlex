"""A small name -> encoder registry so configs can refer to encoders by string."""

from __future__ import annotations

from typing import Callable

from avlex.encoders.audio import EnergyEnvelopeEncoder, MelStatEncoder
from avlex.encoders.base import Encoder
from avlex.encoders.vision import ColorStatsEncoder, MotionHistogramEncoder

EncoderFactory = Callable[..., Encoder]

_REGISTRY: dict[str, EncoderFactory] = {
    "motion_histogram": MotionHistogramEncoder,
    "color_stats": ColorStatsEncoder,
    "mel_stat": MelStatEncoder,
    "energy_envelope": EnergyEnvelopeEncoder,
}


def register_encoder(name: str, factory: EncoderFactory) -> None:
    """Register ``factory`` under ``name`` (overwrites an existing entry)."""
    _REGISTRY[name] = factory


def get_encoder(name: str, **kwargs: object) -> Encoder:
    """Instantiate the encoder registered as ``name`` with ``kwargs``."""
    try:
        factory = _REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"unknown encoder {name!r}; available: {available_encoders()}"
        ) from None
    return factory(**kwargs)


def available_encoders() -> list[str]:
    """Sorted list of registered encoder names."""
    return sorted(_REGISTRY)
