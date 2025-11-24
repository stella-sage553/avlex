import numpy as np

from avlex.encoders import EnergyEnvelopeEncoder, MelStatEncoder


def _tone(freq: float, n: int = 4000, sr: int = 16000, amp: float = 0.5) -> np.ndarray:
    t = np.arange(n) / sr
    return amp * np.sin(2 * np.pi * freq * t)


def test_mel_encoder_shapes():
    enc = MelStatEncoder(n_mels=32)
    feats = enc.encode(_tone(220.0))
    assert feats.shape[1] == 32
    assert enc.output_dim == 32
    assert feats.shape[0] > 1


def test_mel_encoder_is_deterministic():
    enc = MelStatEncoder()
    wav = _tone(440.0)
    np.testing.assert_array_equal(enc.encode(wav), enc.encode(wav))


def test_louder_audio_has_more_energy():
    enc = MelStatEncoder()
    quiet = enc.encode(_tone(440.0, amp=0.05)).mean()
    loud = enc.encode(_tone(440.0, amp=0.9)).mean()
    assert loud > quiet


def test_energy_envelope_shapes_and_range():
    enc = EnergyEnvelopeEncoder()
    feats = enc.encode(_tone(440.0))
    assert feats.shape[1] == 2
    # zero-crossing rate lives in [0, 1]
    assert feats[:, 1].min() >= 0.0
    assert feats[:, 1].max() <= 1.0


def test_short_signal_is_padded():
    enc = MelStatEncoder(frame_length=400, hop_length=160)
    feats = enc.encode(np.array([0.1, -0.2, 0.3]))
    assert feats.shape[0] == 1
