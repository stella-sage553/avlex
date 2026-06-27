"""Prompt assembly: templates, perception rendering, and the built-in library."""

from avlex.prompts.library import CAPTION, SUMMARIZE, VIDEO_QA, get_template
from avlex.prompts.render import parse_perception, render_description, render_words
from avlex.prompts.template import PromptTemplate

__all__ = [
    "PromptTemplate",
    "render_words",
    "parse_perception",
    "render_description",
    "get_template",
    "CAPTION",
    "VIDEO_QA",
    "SUMMARIZE",
]
