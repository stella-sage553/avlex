# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project aims to
follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-06-27

### Added

- Five-stage pipeline: encoders → fusion → bridge → prompt → LLM.
- NumPy encoders: `MotionHistogramEncoder`, `ColorStatsEncoder`,
  `MelStatEncoder`, `EnergyEnvelopeEncoder`.
- Fusion strategies: `ConcatFusion`, `InterleaveFusion`, `GatedFusion`.
- Bridges: `TokenBridge` (offline, textual) plus soft bridges `LinearProjector`,
  `PerceiverResampler`, and `QFormerBridge`.
- Prompt assembly with slot-validated `PromptTemplate` and a deterministic
  offline `TemplateLLM`; optional `OpenAIClient`.
- `PipelineConfig` with dict/YAML (de)serialization and string-keyed registries
  for every component.
- `avlex` CLI: `version`, `demo`, `caption`, `inspect`.
- Documentation, runnable examples, and a test suite at ~90% coverage.

[Unreleased]: https://github.com/stella-sage553/avlex/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/stella-sage553/avlex/releases/tag/v0.1.0

