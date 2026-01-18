import pytest

from avlex.prompts import PromptTemplate, get_template, parse_perception, render_words
from avlex.tasks import Task


def test_template_detects_fields():
    tmpl = PromptTemplate("hello {name}, you are {role}")
    assert tmpl.fields == frozenset({"name", "role"})


def test_template_renders():
    tmpl = PromptTemplate("{greeting}, world")
    assert tmpl.render(greeting="hi") == "hi, world"


def test_missing_slot_raises():
    tmpl = PromptTemplate("{a} and {b}")
    with pytest.raises(KeyError):
        tmpl.render(a="x")


def test_extra_values_are_ignored():
    tmpl = PromptTemplate("{a}")
    assert tmpl.render(a="x", unused="y") == "x"


def test_render_words_round_trips():
    words = [
        "visual: lots of movement",
        "audio: loud sound",
        "timeline: it settles down",
    ]
    block = render_words(words)
    percept = parse_perception(block)
    assert percept["visual"] == "lots of movement"
    assert percept["audio"] == "loud sound"
    assert percept["timeline"] == "it settles down"


def test_empty_words_render_placeholder():
    assert "no salient" in render_words([])


def test_qa_template_requires_question():
    tmpl = get_template(Task.VIDEO_QA)
    assert "question" in tmpl.fields
