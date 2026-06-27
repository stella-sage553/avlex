"""A small name -> bridge registry mirroring the encoder registry."""

from __future__ import annotations

from typing import Callable

from avlex.bridges.base import Bridge
from avlex.bridges.projector import LinearProjector
from avlex.bridges.qformer import QFormerBridge
from avlex.bridges.resampler import PerceiverResampler
from avlex.bridges.token_bridge import TokenBridge

BridgeFactory = Callable[..., Bridge]

_REGISTRY: dict[str, BridgeFactory] = {
    "linear": LinearProjector,
    "perceiver": PerceiverResampler,
    "qformer": QFormerBridge,
    "token": TokenBridge,
}


def register_bridge(name: str, factory: BridgeFactory) -> None:
    _REGISTRY[name] = factory


def get_bridge(name: str, **kwargs: object) -> Bridge:
    try:
        factory = _REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"unknown bridge {name!r}; available: {available_bridges()}"
        ) from None
    return factory(**kwargs)


def available_bridges() -> list[str]:
    return sorted(_REGISTRY)
