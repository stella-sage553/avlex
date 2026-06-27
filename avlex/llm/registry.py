"""A name -> LLM client registry."""

from __future__ import annotations

from typing import Callable

from avlex.llm.base import LLMClient
from avlex.llm.template_llm import TemplateLLM

LLMFactory = Callable[..., LLMClient]

_REGISTRY: dict[str, LLMFactory] = {
    "template": TemplateLLM,
}


def register_llm(name: str, factory: LLMFactory) -> None:
    _REGISTRY[name] = factory


def get_llm(name: str, **kwargs: object) -> LLMClient:
    try:
        factory = _REGISTRY[name]
    except KeyError:
        raise KeyError(f"unknown llm {name!r}; available: {available_llms()}") from None
    return factory(**kwargs)


def available_llms() -> list[str]:
    return sorted(_REGISTRY)
