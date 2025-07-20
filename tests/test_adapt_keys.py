import re

import torch

from model.adapt_EAT_model_dict import convert


def test_convert_end_to_end(tmp_path, capsys):
    """Checking tensor patching"""

    # -- create a fake checkpoint -----------------------------------------
    def _t():  # make a 1‑element tensor quickly
        return torch.randn(1)

    ckpt = {
        "model": {
            "modality_encoders.audio.encoder.layer1.weight": _t(),
            "modality_encoders.video.decoder.layer2.bias": _t(),
            "modality_encoders.audio.context_encoder.fc.weight": _t(),
            "backbone.layer3.weight_ema": _t(),
            "backbone.layer3.bias": _t(),
            "cls_head.weight": _t(),
        },
        "_extra": "anything",  # make sure non‑model blocks are ignored
    }

    src = tmp_path / "orig.pt"
    dst = tmp_path / "patched.pt"
    torch.save(ckpt, src)

    # -- run convert -------------------------------------------------------
    convert(str(src), str(dst))

    # capture stdout
    out, _ = capsys.readouterr()
    assert re.search(r"Patched\s+3\s+tensors", out)  # exactly 3 kept

    # -- validate resulting state dict ------------------------------------
    patched = torch.load(dst, map_location="cpu")
    expected_keys = {
        "encoder.layer1.weight",  # trimmed
        "backbone.layer3.bias",  # unchanged
        "cls_head.weight",  # unchanged
    }
    assert set(patched.keys()) == expected_keys

    # spot‑check that values were copied, not re‑created
    assert torch.equal(
        patched["encoder.layer1.weight"],
        ckpt["model"]["modality_encoders.audio.encoder.layer1.weight"],
    )
