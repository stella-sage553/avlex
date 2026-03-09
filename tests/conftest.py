import pytest

from avlex import Pipeline, PipelineConfig, synthetic_clip


@pytest.fixture
def clip():
    return synthetic_clip(seed=7)


@pytest.fixture
def pipeline():
    return Pipeline.from_config(PipelineConfig())
