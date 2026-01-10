"""Prompt templates with up-front slot validation.

A recurring bug in the reference implementations is silent prompt corruption:
a ``<ImageHere>`` placeholder that never gets filled, or a typo'd field. A
:class:`PromptTemplate` knows its slots and refuses to render with any missing.
"""

from __future__ import annotations

import string


class PromptTemplate:
    """A ``str.format`` template that validates its slots before rendering."""

    def __init__(self, template: str, system: str | None = None) -> None:
        self.template = template
        self.system = system
        self.fields = frozenset(
            name for _, name, _, _ in string.Formatter().parse(template) if name
        )

    def render(self, **values: object) -> str:
        """Fill the template, raising on any missing slot."""
        missing = self.fields - set(values)
        if missing:
            raise KeyError(f"missing prompt slots: {sorted(missing)}")
        return self.template.format(**values)

    def __repr__(self) -> str:
        return f"PromptTemplate(fields={sorted(self.fields)})"
