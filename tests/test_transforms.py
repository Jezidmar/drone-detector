# unit tests for specaug
import random

import torch

import data.utils as utils


def _dummy_wave(sec=1.0, sr=utils.SR):
    t = torch.linspace(0, sec, int(sec * sr))
    return torch.sin(2 * torch.pi * 440 * t)  # 440 Hz sine


def test_specaugment_masks_when_forced(monkeypatch):
    fb = utils.wav_to_fbank(_dummy_wave(), sr=utils.SR)

    # |_ monkey‑patch random.random to always < p so masking happens
    monkeypatch.setattr(random, "random", lambda: 0.0)
    aug = utils.apply_spec_augment(fb, p=1.0)

    assert not torch.allclose(aug, fb)  # something got masked
    assert aug.shape == fb.shape


def test_key_adapt(monkeypatch):
    fb = utils.wav_to_fbank(_dummy_wave(), sr=utils.SR)

    # |_ monkey‑patch random.random to always < p so masking happens
    monkeypatch.setattr(random, "random", lambda: 0.0)
    aug = utils.apply_spec_augment(fb, p=1.0)

    assert not torch.allclose(aug, fb)  # something got masked
    assert aug.shape == fb.shape
