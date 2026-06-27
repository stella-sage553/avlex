# avlex

[![CI](https://github.com/stella-sage553/avlex/actions/workflows/ci.yml/badge.svg)](https://github.com/stella-sage553/avlex/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)

**Bridge audio-visual encoders to LLMs for video captioning and understanding.**

`avlex` is a small, composable framework for the part of an audio-visual LLM that
everyone reimplements from scratch: the **bridge** that turns encoder features
into something a language model can read. Plug in an audio encoder and a visual
encoder, pick a bridge, point it at an LLM, and ask for a caption, a summary, or
an answer.

The core is pure NumPy — no GPU, no model downloads — so the whole pipeline runs
(and is tested) offline. Bring your own torch encoders and a hosted LLM when you
want quality.

## Why

Research code for audio-visual LLMs (Video-LLaMA, BLIP-2-style connectors,
Q-Former resamplers, MLP projectors) tends to hard-wire the encoder, the bridge,
the fusion, and the prompt into one model class. Swapping any piece means editing
the model. avlex pulls those four concerns apart into small, registry-backed
abstractions you can mix, match, and test in isolation.

## Install

```bash
pip install avlex            # core (numpy + pyyaml)
pip install "avlex[openai]"  # optional hosted-LLM backend
```

## Quickstart

```python
from avlex import Pipeline, PipelineConfig, synthetic_clip

pipe = Pipeline.from_config(PipelineConfig())   # the offline defaults
clip = synthetic_clip(seed=1)                   # a toy clip, no media needed

print(pipe.caption(clip).text)
# The clip shows a little movement and moderate sound with mid-range tones,
# and the action builds toward the end.

print(pipe.answer(clip, "What can you hear?").text)
# In the audio, there is moderate sound with mid-range tones.
```

## The five stages

```
encoders ──▶ fusion ──▶ bridge ──▶ prompt ──▶ llm
```

| Stage | Abstraction | Built-ins |
|-------|-------------|-----------|
| Encode | `Encoder` | `MotionHistogramEncoder`, `ColorStatsEncoder`, `MelStatEncoder`, `EnergyEnvelopeEncoder` |
| Fuse | `Fusion` | `ConcatFusion`, `InterleaveFusion`, `GatedFusion` |
| Bridge | `Bridge` | `TokenBridge`, `LinearProjector`, `PerceiverResampler`, `QFormerBridge` |
| Prompt | `PromptTemplate` | caption / video-QA / summarize templates |
| Generate | `LLMClient` | `TemplateLLM` (offline), `OpenAIClient` (optional) |

The **token bridge** is what makes the offline path work: instead of soft-prompt
embeddings, it reads interpretable descriptors out of the features ("a little
movement", "moderate sound", "the action builds") and drops them into the prompt
for any text LLM. The soft bridges (projector, resampler, Q-Former) produce the
`(K, D)` token arrays a multimodal LLM consumes — the architecture is here; bring
the trained weights.

## CLI

```bash
avlex demo                       # caption a built-in synthetic clip
avlex caption clip.npz           # caption an .npz of visual/audio arrays
avlex caption clip.npz --question "what is happening?"
avlex inspect clip.npz           # show feature shapes and the bridge output
```

## Configuration

Everything is describable in a small YAML file:

```yaml
visual_encoder: { name: motion_histogram, options: { n_bins: 32 } }
audio_encoder: mel_stat
fusion: concat
bridge: { name: token }
llm: template
task: caption
```

```python
from avlex import Pipeline, PipelineConfig
pipe = Pipeline.from_config(PipelineConfig.from_yaml("pipeline.yaml"))
```

## Documentation

- [docs/architecture.md](docs/architecture.md) — the five-stage design
- [docs/usage.md](docs/usage.md) — recipes and the public API
- [docs/design-notes.md](docs/design-notes.md) — why the bridges look the way they do
- [docs/api-reference.md](docs/api-reference.md) — module-by-module reference
- [examples/](examples/) — runnable scripts

## License

MIT — see [LICENSE](LICENSE).
