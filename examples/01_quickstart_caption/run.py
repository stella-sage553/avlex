"""Caption a synthetic clip with the offline default pipeline."""

from avlex import Pipeline, PipelineConfig, synthetic_clip


def main() -> None:
    pipe = Pipeline.from_config(PipelineConfig())
    clip = synthetic_clip(seed=1)
    result = pipe.caption(clip)
    print("caption:", result.text)
    print("perception:", result.words)


if __name__ == "__main__":
    main()
