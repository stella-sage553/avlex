"""Command-line interface for avlex.

avlex does not decode media itself, so the CLI works on ``.npz`` files holding
``visual`` and/or ``audio`` arrays (whatever the configured encoders expect), or
on the built-in synthetic clip via ``avlex demo``.
"""

from __future__ import annotations

import argparse

from avlex import __version__
from avlex.clip import synthetic_clip
from avlex.config import PipelineConfig
from avlex.pipeline import Pipeline


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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="avlex", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_version = sub.add_parser("version", help="print the installed version")
    p_version.set_defaults(func=_cmd_version)

    p_demo = sub.add_parser("demo", help="caption a built-in synthetic clip")
    p_demo.add_argument("--seed", type=int, default=0, help="synthetic clip seed")
    p_demo.add_argument("--config", default=None, help="pipeline config YAML")
    p_demo.set_defaults(func=_cmd_demo)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
