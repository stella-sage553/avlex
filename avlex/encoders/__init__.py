"""Encoders: raw modality input -> temporal feature sequences."""

from avlex.encoders.audio import EnergyEnvelopeEncoder, MelStatEncoder
from avlex.encoders.base import Encoder
from avlex.encoders.registry import (
    available_encoders,
    get_encoder,
    register_encoder,
)
from avlex.encoders.vision import ColorStatsEncoder, MotionHistogramEncoder

__all__ = [
    "Encoder",
    "MotionHistogramEncoder",
    "ColorStatsEncoder",
    "MelStatEncoder",
    "EnergyEnvelopeEncoder",
    "get_encoder",
    "register_encoder",
    "available_encoders",
]
