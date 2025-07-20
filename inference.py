# app.py
import argparse

import matplotlib.pyplot as plt
import soundfile as sf
import streamlit as st
import torch
import torchaudio

from model.load_model import load_EAT_model

p = argparse.ArgumentParser()
p.add_argument("--ckpt_path", required=True, help="pre-trained model checkpoint")
p.add_argument("--cfg_path", required=True, help="model config path")
args = p.parse_args()

# ------------------------------------------------------------------
# Constants
FS = 10
SR = 16_000
NORM_MEAN = -4.268
NORM_STD = 4.569
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ------------------------------------------------------------------


@st.cache_resource(show_spinner=False)
def load_model(ckpt_path: str, config_path):
    """Load the EAT model once and keep it in cache/session."""
    model = load_EAT_model(ckpt_path, config_path).to(DEVICE)
    model.eval()
    return model


def load_audio(file):
    """Read audio, return torch waveform (1‑D) resampled to SR."""
    wav, sr = sf.read(file)
    waveform = torch.tensor(wav).float()
    if sr != SR:
        waveform = torchaudio.functional.resample(waveform, sr, SR)
    return waveform


def to_mel(waveform):
    """Compute 128‑bin log‑Mel spectrogram."""
    waveform = waveform - waveform.mean()
    mel = torchaudio.compliance.kaldi.fbank(
        waveform.unsqueeze(0),
        htk_compat=True,
        sample_frequency=SR,
        use_energy=False,
        window_type="hanning",
        num_mel_bins=128,
        dither=0.0,
        frame_shift=FS,
    )
    return mel


def pad_or_truncate(mel, target_length):
    """Pad with zeros or truncate to target length (frames)."""
    n_frames = mel.shape[0]
    if n_frames < target_length:
        pad = torch.zeros(target_length - n_frames, mel.shape[1]).to(DEVICE)
        mel = torch.cat([mel, pad], dim=0)
    else:
        mel = mel[:target_length, :]
    return mel


# ------------------------- Streamlit UI ---------------------------
st.title("🎛️  Drone Detector")


target_len = st.slider("Target length (frames)", 256, 2048, 1024, step=128)

model = load_model(args.ckpt_path, args.cfg_path)


# Main area
audio_file = st.file_uploader("Upload .wav file", type=["wav"])

if audio_file is not None:
    # ---------- inference ----------
    waveform = load_audio(audio_file).to(DEVICE)
    mel = to_mel(waveform).to(DEVICE)
    mel = pad_or_truncate(mel, target_len)
    st.subheader("Mel‑spectrogram")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.imshow(mel.cpu().T, aspect="auto", origin="lower")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Mel bin")
    st.pyplot(fig)
    # Normalise & reshape for model (B × C × T × F)
    mel_input = ((mel - NORM_MEAN) / (NORM_STD * 2)).unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        logits = model(mel_input)
        prob = torch.sigmoid(logits).item()

    # ---------- display ----------
    st.subheader("Detection result")

    prob_pct = prob * 100

    # 2) Streamlit metric widget
    st.metric(label="Probability", value=f"{prob_pct:.1f} %")

    # Visual bar
    st.progress(prob)

    # Decision banner
    if prob > 0.5:
        st.success("🚁  **Drone detected**")
    else:
        st.info("🌳  **No drone detected**")
else:
    st.info("⬆️  Upload a .wav file to run detection.")
