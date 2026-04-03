# Architecture

avlex is organised around a single pipeline with five replaceable stages:

```
ClipFeatures
     │
     ▼
┌──────────┐   {modality: (T, D)}
│ encoders │ ─────────────────────────┐
└──────────┘                          ▼
                                 ┌──────────┐   BridgeInput
                                 │  fusion  │ ───────────────┐
                                 └──────────┘                ▼
                                                       ┌──────────┐  BridgeOutput
                                                       │  bridge  │ ──────────────┐
                                                       └──────────┘   tokens/words │
                                                                                   ▼
                                                                             ┌──────────┐
                                                                             │  prompt  │
                                                                             └──────────┘
                                                                                   │ messages
                                                                                   ▼
                                                                             ┌──────────┐
                                                                             │   llm    │ ─▶ text
                                                                             └──────────┘
```

## Stages

### 1. Encoders (`avlex.encoders`)

An `Encoder` maps one modality's raw input to a temporal feature sequence
`(T, D)`. The bundled encoders are deterministic NumPy descriptors:

- `MotionHistogramEncoder` — per-frame histogram of inter-frame change.
- `ColorStatsEncoder` — per-frame intensity histogram.
- `MelStatEncoder` — log-mel energies via an FFT + triangular mel filterbank.
- `EnergyEnvelopeEncoder` — RMS loudness and zero-crossing rate.

They share the `Encoder` interface, so a torch-backed Whisper/ViT encoder is a
drop-in: subclass `Encoder`, return an array.

### 2. Fusion (`avlex.fusion`)

A `Fusion` projects each stream to a common width and merges them into the
`BridgeInput.sequence` that soft bridges consume, while keeping the raw
per-modality features for the token bridge.

- `ConcatFusion` — streams occupy disjoint stretches of one timeline.
- `InterleaveFusion` — streams are aligned and zipped frame by frame.
- `GatedFusion` — streams are aligned and combined with a per-step energy gate.

### 3. Bridge (`avlex.bridges`)

The bridge is the crux. It reduces variable-length, fused features into a fixed,
LLM-ready representation. Two flavours:

- **soft bridges** emit `(K, D)` token arrays for a multimodal LLM:
  `LinearProjector` (with stack/avg-pool compression), `PerceiverResampler`
  (query tokens cross-attending the features), `QFormerBridge` (query tokens
  with interleaved self/cross-attention).
- **token bridge** emits textual descriptors for a text LLM.

### 4. Prompt (`avlex.prompts`)

`PromptTemplate` assembles the perception block and task instruction into a chat
prompt, validating that every slot is filled — no silently dropped placeholders.

### 5. LLM (`avlex.llm`)

`LLMClient` is a tiny chat interface. `TemplateLLM` is a deterministic offline
stand-in that reads the perception block and composes an answer; `OpenAIClient`
is an optional hosted backend.

## Configuration & registries

Each stage has a string-keyed registry, so a whole pipeline can be named in YAML
and rebuilt with `Pipeline.from_config(PipelineConfig.from_yaml(...))`.
