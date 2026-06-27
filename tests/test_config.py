from avlex import Pipeline, PipelineConfig, Task
from avlex.config import ComponentConfig


def test_dict_round_trip():
    config = PipelineConfig()
    data = config.to_dict()
    assert PipelineConfig.from_dict(data).to_dict() == data


def test_from_dict_accepts_shorthand():
    config = PipelineConfig.from_dict(
        {"bridge": "perceiver", "llm": {"name": "template"}, "task": "summarize"}
    )
    assert config.bridge.name == "perceiver"
    assert config.llm.name == "template"
    assert config.task == Task.SUMMARIZE


def test_null_encoder_is_preserved():
    config = PipelineConfig.from_dict({"visual_encoder": None})
    assert config.visual_encoder is None


def test_component_options_carry_through():
    config = PipelineConfig.from_dict(
        {"bridge": {"name": "token", "options": {"include_timeline": False}}}
    )
    assert config.bridge == ComponentConfig("token", {"include_timeline": False})


def test_yaml_round_trip(tmp_path):
    config = PipelineConfig.from_dict(
        {"bridge": {"name": "token", "options": {"include_timeline": False}}}
    )
    path = tmp_path / "config.yaml"
    config.to_yaml(path)
    loaded = PipelineConfig.from_yaml(path)
    assert loaded.bridge.options == {"include_timeline": False}


def test_build_pipeline_from_config(clip):
    config = PipelineConfig.from_dict({"fusion": "gated"})
    pipe = Pipeline.from_config(config)
    assert pipe.fusion.name == "GatedFusion"
    assert pipe.caption(clip).text.endswith(".")
