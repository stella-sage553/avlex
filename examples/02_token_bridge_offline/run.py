"""Inspect the perceptual words produced by the token bridge."""

from avlex import synthetic_clip
from avlex.bridges import TokenBridge
from avlex.encoders import MelStatEncoder, MotionHistogramEncoder
from avlex.fusion import ConcatFusion
from avlex.types import Modality


def main() -> None:
    clip = synthetic_clip(seed=4)

    features = {
        Modality.VISUAL: MotionHistogramEncoder().encode(clip.visual),
        Modality.AUDIO: MelStatEncoder().encode(clip.audio),
    }
    fused = ConcatFusion().fuse(features)
    output = TokenBridge().bridge(fused)

    print("perceptual words:")
    for word in output.words or []:
        print(" -", word)
    print("scalars:", output.meta)


if __name__ == "__main__":
    main()
