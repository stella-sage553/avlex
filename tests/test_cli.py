import numpy as np

from avlex import synthetic_clip
from avlex.cli import main


def _write_clip(tmp_path) -> str:
    clip = synthetic_clip(seed=2)
    path = tmp_path / "clip.npz"
    np.savez(
        path,
        visual=clip.visual,
        audio=clip.audio,
        fps=clip.fps,
        sample_rate=clip.sample_rate,
    )
    return str(path)


def test_version(capsys):
    assert main(["version"]) == 0
    assert capsys.readouterr().out.strip()


def test_demo(capsys):
    assert main(["demo", "--seed", "1"]) == 0
    assert capsys.readouterr().out.strip().endswith(".")


def test_caption(tmp_path, capsys):
    assert main(["caption", _write_clip(tmp_path)]) == 0
    assert capsys.readouterr().out.strip().endswith(".")


def test_caption_question(tmp_path, capsys):
    path = _write_clip(tmp_path)
    assert main(["caption", path, "--question", "what can you hear?"]) == 0
    assert "audio" in capsys.readouterr().out.lower()


def test_inspect(tmp_path, capsys):
    assert main(["inspect", _write_clip(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "bridge" in out.lower()
    assert "timeline" in out


def test_missing_arrays_errors(tmp_path):
    path = tmp_path / "empty.npz"
    np.savez(path, other=np.zeros(3))
    try:
        main(["caption", str(path)])
    except SystemExit:
        return
    raise AssertionError("expected SystemExit for npz without visual/audio")
