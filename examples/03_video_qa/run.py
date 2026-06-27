"""Ask several questions about one clip."""

from avlex import Pipeline, PipelineConfig, synthetic_clip

QUESTIONS = [
    "Is anything moving?",
    "What can you hear?",
    "Does the clip get more or less active?",
]


def main() -> None:
    pipe = Pipeline.from_config(PipelineConfig())
    clip = synthetic_clip(seed=2)
    for question in QUESTIONS:
        print(f"Q: {question}")
        print(f"A: {pipe.answer(clip, question).text}\n")


if __name__ == "__main__":
    main()
