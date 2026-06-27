# avlex

Bridge audio-visual encoders to LLMs for video captioning and understanding.

`avlex` is a small, composable framework: plug in an audio encoder and a visual
encoder, pick a *bridge* that turns their features into LLM-ready tokens, and let
a language model do the talking. Pure NumPy core, no GPU required.

Status: early work in progress.
