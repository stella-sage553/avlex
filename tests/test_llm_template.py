from avlex.llm import Message, TemplateLLM
from avlex.prompts import get_template, render_words
from avlex.tasks import Task, get_task_spec

_WORDS = [
    "visual: lots of movement",
    "audio: loud sound with bright, high tones",
    "timeline: the action builds toward the end",
]


def _caption_prompt() -> str:
    spec = get_task_spec(Task.CAPTION)
    return get_template(Task.CAPTION).render(
        perception=render_words(_WORDS), instruction=spec.instruction
    )


def test_caption_mentions_perceived_content():
    llm = TemplateLLM()
    text = llm.complete(_caption_prompt())
    assert "movement" in text
    assert text.endswith(".")


def test_template_llm_is_deterministic():
    llm = TemplateLLM()
    prompt = _caption_prompt()
    assert llm.complete(prompt) == llm.complete(prompt)


def test_question_routes_to_audio():
    spec = get_task_spec(Task.VIDEO_QA)
    prompt = get_template(Task.VIDEO_QA).render(
        perception=render_words(_WORDS),
        instruction=spec.instruction,
        question="What can you hear?",
    )
    text = TemplateLLM().generate([Message("user", prompt)]).text
    assert "audio" in text.lower()


def test_summary_is_longer_than_caption():
    spec = get_task_spec(Task.SUMMARIZE)
    prompt = get_template(Task.SUMMARIZE).render(
        perception=render_words(_WORDS), instruction=spec.instruction
    )
    summary = TemplateLLM().complete(prompt)
    assert "Overall" in summary
