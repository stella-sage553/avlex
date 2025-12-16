"""Bridges that connect audio-visual features to language models."""

from avlex.bridges.base import Bridge, BridgeInput, BridgeOutput
from avlex.bridges.projector import LinearProjector
from avlex.bridges.qformer import QFormerBridge
from avlex.bridges.registry import (
    available_bridges,
    get_bridge,
    register_bridge,
)
from avlex.bridges.resampler import PerceiverResampler
from avlex.bridges.token_bridge import TokenBridge

__all__ = [
    "Bridge",
    "BridgeInput",
    "BridgeOutput",
    "LinearProjector",
    "PerceiverResampler",
    "QFormerBridge",
    "TokenBridge",
    "get_bridge",
    "register_bridge",
    "available_bridges",
]
