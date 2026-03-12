import pytest

from avlex.bridges import TokenBridge, available_bridges, get_bridge
from avlex.encoders import MotionHistogramEncoder, available_encoders, get_encoder
from avlex.fusion import ConcatFusion, available_fusions, get_fusion
from avlex.llm import TemplateLLM, available_llms, get_llm


def test_encoder_registry():
    assert isinstance(get_encoder("motion_histogram"), MotionHistogramEncoder)
    assert {"mel_stat", "energy_envelope"} <= set(available_encoders())


def test_bridge_registry():
    assert isinstance(get_bridge("token"), TokenBridge)
    assert "qformer" in available_bridges()


def test_fusion_registry():
    assert isinstance(get_fusion("concat"), ConcatFusion)
    assert "gated" in available_fusions()


def test_llm_registry():
    assert isinstance(get_llm("template"), TemplateLLM)
    assert "template" in available_llms()


def test_unknown_name_raises():
    with pytest.raises(KeyError):
        get_encoder("does_not_exist")


def test_options_pass_through():
    bridge = get_bridge("perceiver", num_queries=4)
    assert bridge.num_queries == 4
