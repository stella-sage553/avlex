"""Register a custom bridge and use it by name through the registry."""

import numpy as np

from avlex import Pipeline, PipelineConfig, synthetic_clip
from avlex.bridges import BridgeInput, BridgeOutput, TokenBridge, register_bridge
from avlex.types import Modality


class WarmthBridge(TokenBridge):
    """Token bridge that also reports a rough colour-temperature of the audio."""

    def bridge(self, features: BridgeInput) -> BridgeOutput:
        out = super().bridge(features)
        words = list(out.words or [])
        audio = features.modalities.get(Modality.AUDIO)
        if audio is not None and audio.size:
            centroid = float(np.mean(np.argmax(audio, axis=1))) / audio.shape[1]
            words.append(
                "mood: warm and bright" if centroid > 0.5 else "mood: cool and low"
            )
        return BridgeOutput(words=words, meta=out.meta)


register_bridge("warmth", WarmthBridge)


def main() -> None:
    pipe = Pipeline.from_config(PipelineConfig.from_dict({"bridge": "warmth"}))
    result = pipe.caption(synthetic_clip(seed=5))
    print("words:", result.words)
    print("caption:", result.text)


if __name__ == "__main__":
    main()
