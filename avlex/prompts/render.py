"""Render bridge ``words`` into the perception block of a prompt.

The token bridge emits tagged phrases like ``"visual: lots of movement"``. We
render them as a bullet list that the offline :class:`TemplateLLM` can parse back,
and that reads naturally to a real LLM too.
"""

from __future__ import annotations


def render_words(words: list[str]) -> str:
    """Render tagged phrases as a ``- tag: phrase`` bullet block."""
    if not words:
        return "- (no salient audio-visual cues detected)"
    return "\n".join(f"- {word}" for word in words)


def parse_perception(block: str) -> dict[str, str]:
    """Inverse of :func:`render_words`: bullet block -> ``{tag: phrase}``."""
    percept: dict[str, str] = {}
    for line in block.splitlines():
        line = line.strip()
        if not line.startswith("- "):
            continue
        body = line[2:]
        if ": " in body:
            tag, phrase = body.split(": ", 1)
            percept[tag.strip()] = phrase.strip()
    return percept


def render_description(words: list[str]) -> str:
    """A one-line prose flattening of the perception, for logging/inspection."""
    percept = parse_perception(render_words(words))
    return "; ".join(f"{tag} {phrase}" for tag, phrase in percept.items())
