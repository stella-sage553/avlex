"""Fusion strategies for combining audio and visual streams."""

from __future__ import annotations

from typing import Callable

from avlex.fusion.base import Fusion
from avlex.fusion.strategies import ConcatFusion, GatedFusion, InterleaveFusion

_REGISTRY: dict[str, Callable[..., Fusion]] = {
    "concat": ConcatFusion,
    "interleave": InterleaveFusion,
    "gated": GatedFusion,
}


def get_fusion(name: str, **kwargs: object) -> Fusion:
    try:
        factory = _REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"unknown fusion {name!r}; available: {sorted(_REGISTRY)}"
        ) from None
    return factory(**kwargs)


def available_fusions() -> list[str]:
    return sorted(_REGISTRY)


__all__ = [
    "Fusion",
    "ConcatFusion",
    "InterleaveFusion",
    "GatedFusion",
    "get_fusion",
    "available_fusions",
]
