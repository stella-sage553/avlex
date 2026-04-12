# Design notes

A few decisions that shaped avlex, and the trade-offs behind them.

## Why a NumPy core

Audio-visual LLM repos are heavy: torch, CUDA, model checkpoints, often a
fairseq or transformers pin. That weight makes them hard to install, slow to
test, and impossible to run in CI. avlex keeps the *orchestration* and the
*bridge architecture* in plain NumPy, and treats torch encoders / hosted LLMs as
optional plug-ins. The result installs in seconds and tests in milliseconds.

## Untrained, seeded weights

The soft bridges (projector, resampler, Q-Former) need projection matrices and
query tokens. avlex draws them deterministically from a seed
(`avlex.utils.seeding`) rather than training them. This is honest about what the
framework provides — the *shape* of the computation, not learned parameters — and
keeps every run reproducible. Names are hashed into seeds with SHA-256 rather
than the builtin `hash`, which is salted per process.

## The token bridge

Soft prompts only help a model that was trained to consume them. To make the
offline path genuinely useful, the token bridge takes the other road: it reads
interpretable scalars from the features — motion level, loudness, spectral
centroid, and whether energy rises or falls over time — and renders them as short
phrases. Any text LLM can use those. The thresholds are calibrated for the
bundled encoders and are easy to override for your own.

## Variable length in, fixed tokens out

The recurring problem these systems solve is collapsing a variable number of
frames/segments into a fixed budget of LLM tokens. avlex offers the three common
answers: pool-and-project (cheap), perceiver resampling (a fixed query set cross-
attending the timeline), and a Q-Former (queries that also attend to each other).
All three are `Bridge` subclasses returning `(num_queries, out_dim)`.

## Validation where it bites

A silent prompt bug — an unfilled `<ImageHere>` placeholder — is a classic way to
quietly wreck a multimodal prompt. `PromptTemplate` knows its slots and refuses
to render with any missing, turning a silent corruption into a loud `KeyError`.

## Deliberate non-goals

avlex does not decode video/audio files, does not train anything, and does not
ship model weights. It is the connective tissue between encoders you already have
and an LLM you already use.
