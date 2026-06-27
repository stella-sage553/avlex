# Usage

## Building a pipeline

The quickest way is the offline default:

```python
from avlex import Pipeline, PipelineConfig

pipe = Pipeline.from_config(PipelineConfig())
```

Or assemble it by hand to swap a component:

```python
from avlex import Pipeline
from avlex.encoders import MotionHistogramEncoder, MelStatEncoder
from avlex.fusion import GatedFusion
from avlex.bridges import TokenBridge
from avlex.llm import TemplateLLM

pipe = Pipeline(
    visual_encoder=MotionHistogramEncoder(n_bins=32),
    audio_encoder=MelStatEncoder(n_mels=40),
    fusion=GatedFusion(d_model=128),
    bridge=TokenBridge(),
    llm=TemplateLLM(),
)
```

## Running tasks

```python
from avlex import synthetic_clip

clip = synthetic_clip(seed=0)

pipe.caption(clip).text          # one-sentence caption
pipe.summarize(clip).text        # a couple of sentences
pipe.answer(clip, "What can you hear?").text   # video QA
```

Every call returns a `PipelineResult` carrying the `text`, the assembled
`prompt`, the bridge `words`/`tokens`, and a `meta` dict — handy for debugging
exactly what the LLM saw.

## Bringing your own clip

avlex does not decode media; you pass arrays the encoders understand. For the
bundled encoders that means grayscale frames `(T, H, W)` and a waveform `(N,)`:

```python
import numpy as np
from avlex import ClipFeatures

clip = ClipFeatures(
    visual=np.load("frames.npy"),   # (T, H, W) in [0, 1]
    audio=np.load("waveform.npy"),  # (N,)
    fps=8.0,
    sample_rate=16000,
)
```

## Soft bridges and multimodal LLMs

The token bridge targets text LLMs. To produce soft-prompt tokens for a
multimodal model, swap in a soft bridge:

```python
from avlex.bridges import PerceiverResampler

result_tokens = PerceiverResampler(num_queries=32, out_dim=4096)
```

The soft bridges output `(K, out_dim)` arrays from `bridge(...).tokens`. Feed
those to your model as input embeddings; avlex supplies the architecture, you
supply the trained projection weights.

## Configuration files

See [architecture.md](architecture.md) for the YAML schema; load with
`PipelineConfig.from_yaml(path)` and save with `config.to_yaml(path)`.
