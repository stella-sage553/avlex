"""Command-line interface for avlex.

avlex does not decode media itself, so the CLI works on ``.npz`` files holding
``visual`` and/or ``audio`` arrays (whatever the configured encoders expect), or
on the built-in synthetic clip via ``avlex demo``.
"""

from __future__ import annotations

import argparse

import numpy as np

from avlex import __version__
from avlex.clip import ClipFeatures, synthetic_clip
from avlex.config import PipelineConfig
from avlex.pipeline import Pipeline
from avlex.tasks import Task


def _load_clip(path: str) -> ClipFeatures:
    data = np.load(path)
    visual = data["visual"] if "visual" in data.files else None
    audio = data["audio"] if "audio" in data.files else None
    if visual is None and audio is None:
        raise SystemExit(f"{path}: npz must contain a 'visual' and/or 'audio' array")
    extra: dict = {}
    if "fps" in data.files:
        extra["fps"] = float(data["fps"])
    if "sample_rate" in data.files:
        extra["sample_rate"] = int(data["sample_rate"])
    return ClipFeatures(visual=visual, audio=audio, **extra)


def _pipeline(config_path: str | None) -> Pipeline:
    config = PipelineConfig.from_yaml(config_path) if config_path else PipelineConfig()
    return Pipeline.from_config(config)


def _cmd_version(args: argparse.Namespace) -> int:
    print(__version__)
    return 0


def _cmd_demo(args: argparse.Namespace) -> int:
    pipeline = _pipeline(args.config)
    clip = synthetic_clip(seed=args.seed)
    print(pipeline.caption(clip).text)
    return 0


def _cmd_caption(args: argparse.Namespace) -> int:
    pipeline = _pipeline(args.config)
    clip = _load_clip(args.input)
    if args.question:
        result = pipeline.answer(clip, args.question)
    else:
        result = pipeline.run(clip, Task(args.task))
    print(result.text)
    return 0


def _cmd_inspect(args: argparse.Namespace) -> int:
    pipeline = _pipeline(args.config)
    clip = _load_clip(args.input)
    features = pipeline._encode(clip)
    for modality, feat in features.items():
        print(f"{modality}: {feat.shape}")
    fused = pipeline.fusion.fuse(features)
    print(f"fused: {fused.sequence.shape} via {pipeline.fusion.name}")
    output = pipeline.bridge.bridge(fused)
    print(f"bridge {pipeline.bridge.name}: {output.num_tokens} tokens")
    if output.words:
        for word in output.words:
            print(f"  - {word}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="avlex", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_version = sub.add_parser("version", help="print the installed version")
    p_version.set_defaults(func=_cmd_version)

    p_demo = sub.add_parser("demo", help="caption a built-in synthetic clip")
    p_demo.add_argument("--seed", type=int, default=0, help="synthetic clip seed")
    p_demo.add_argument("--config", default=None, help="pipeline config YAML")
    p_demo.set_defaults(func=_cmd_demo)

    p_caption = sub.add_parser("caption", help="caption/answer over an .npz clip")
    p_caption.add_argument("input", help="path to an .npz with visual/audio arrays")
    p_caption.add_argument(
        "--task",
        default="caption",
        choices=["caption", "summarize"],
        help="task to run (use --question for QA)",
    )
    p_caption.add_argument("--question", default=None, help="ask a question (video QA)")
    p_caption.add_argument("--config", default=None, help="pipeline config YAML")
    p_caption.set_defaults(func=_cmd_caption)

    p_inspect = sub.add_parser("inspect", help="show feature/bridge shapes for a clip")
    p_inspect.add_argument("input", help="path to an .npz with visual/audio arrays")
    p_inspect.add_argument("--config", default=None, help="pipeline config YAML")
    p_inspect.set_defaults(func=_cmd_inspect)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
