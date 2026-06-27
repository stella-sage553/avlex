# API reference

A module-by-module tour of the public surface. Import the headline objects from
the package root: `from avlex import Pipeline, PipelineConfig, ClipFeatures`.

## `avlex`

- `__version__` — package version string.
- `ClipFeatures(visual=None, audio=None, fps=1.0, sample_rate=16000, meta={})`
- `synthetic_clip(seed=0, n_frames=24, size=16, sample_rate=16000, duration=1.5)`
- `Pipeline` / `PipelineConfig` / `ComponentConfig`
- `PipelineResult` — `.text`, `.answer`, `.task`, `.prompt`, `.words`, `.tokens`, `.meta`
- `Task` — `CAPTION`, `VIDEO_QA`, `SUMMARIZE`
- `Modality` — `AUDIO`, `VISUAL`

## `avlex.Pipeline`

- `Pipeline(visual_encoder, audio_encoder, fusion, bridge, llm)`
- `Pipeline.from_config(config) -> Pipeline`
- `run(clip, task=Task.CAPTION, question=None) -> PipelineResult`
- `caption(clip)` / `summarize(clip)` / `answer(clip, question)`

## `avlex.encoders`

- `Encoder` (ABC: `encode(raw) -> (T, D)`, `output_dim`, `modality`)
- `MotionHistogramEncoder(n_bins=32)`, `ColorStatsEncoder(n_bins=16)`
- `MelStatEncoder(n_mels=40, ...)`, `EnergyEnvelopeEncoder(...)`
- `get_encoder(name, **opts)`, `register_encoder(name, factory)`, `available_encoders()`

## `avlex.fusion`

- `Fusion` (ABC: `fuse({modality: array}) -> BridgeInput`)
- `ConcatFusion`, `InterleaveFusion`, `GatedFusion` (all take `d_model`, `seed`)
- `get_fusion(name, **opts)`, `available_fusions()`

## `avlex.bridges`

- `Bridge` (ABC: `bridge(BridgeInput) -> BridgeOutput`)
- `BridgeInput(sequence, modalities, spans)`, `BridgeOutput(tokens, words, meta)`
- `TokenBridge(include_timeline=True)`
- `LinearProjector(out_dim, compression, factor, seed)`
- `PerceiverResampler(num_queries, out_dim, n_heads, depth, seed)`
- `QFormerBridge(num_queries, out_dim, n_heads, depth, seed)`
- `get_bridge(name, **opts)`, `register_bridge(name, factory)`, `available_bridges()`

Building blocks in `avlex.bridges.quantize`: `assign_codebook`, `kmeans`,
`temporal_segments`; attention primitives in `avlex.bridges.attention`.

## `avlex.prompts`

- `PromptTemplate(template, system=None)` — `.render(**slots)`, `.fields`
- `render_words(words)`, `parse_perception(block)`
- `get_template(task)`; templates `CAPTION`, `VIDEO_QA`, `SUMMARIZE`

## `avlex.llm`

- `LLMClient` (ABC: `generate(messages, config=None) -> LLMResponse`, `complete(prompt)`)
- `Message(role, content)`, `GenerationConfig(...)`, `LLMResponse(text, meta)`
- `TemplateLLM` — deterministic offline backend
- `get_llm(name, **opts)`, `register_llm(name, factory)`, `available_llms()`

## `avlex.cli`

- `main(argv=None) -> int` — entry point for the `avlex` console script.
