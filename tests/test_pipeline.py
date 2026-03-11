import pytest

from avlex import ClipFeatures, Pipeline, PipelineConfig, Task


def test_caption_returns_a_sentence(pipeline, clip):
    result = pipeline.caption(clip)
    assert result.task == Task.CAPTION
    assert result.text.endswith(".")
    assert result.words
    assert result.meta["bridge"] == "TokenBridge"


def test_pipeline_is_deterministic(pipeline, clip):
    assert pipeline.caption(clip).text == pipeline.caption(clip).text


def test_summarize_has_multiple_sentences(pipeline, clip):
    text = pipeline.summarize(clip).text
    assert text.count(".") >= 2


def test_answer_uses_alias(pipeline, clip):
    result = pipeline.answer(clip, "What can you hear?")
    assert result.answer == result.text
    assert "audio" in result.text.lower()


def test_video_qa_requires_a_question(pipeline, clip):
    with pytest.raises(ValueError):
        pipeline.run(clip, Task.VIDEO_QA)


def test_pipeline_rejects_clip_with_no_encodable_modality(clip):
    # this pipeline has no audio encoder, so an audio-only clip is unusable
    pipe = Pipeline.from_config(PipelineConfig(audio_encoder=None))
    audio_only = ClipFeatures(audio=clip.audio)
    with pytest.raises(ValueError):
        pipe.caption(audio_only)
