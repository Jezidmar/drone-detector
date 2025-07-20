import os

import numpy as np
import sounddevice as sd
import torch
import torchaudio

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import argparse
import datetime
import json
import os
import uuid

import onnxruntime as ort
import soundfile as sf

N_MELS = 128  # #mel bins
NORM_MEAN = -4.268
NORM_STD = 4.569
FS = 10  # 10ms


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def wav_to_fbank(wav: torch.Tensor, sr: int) -> torch.Tensor:
    wav -= wav.mean()
    fb = torchaudio.compliance.kaldi.fbank(
        wav,
        htk_compat=True,
        sample_frequency=sr,
        num_mel_bins=N_MELS,
        use_energy=False,
        window_type="hanning",
        dither=0.0,
        frame_shift=FS,
    )  # [T , 128]
    fb = (fb - NORM_MEAN) / (2 * NORM_STD)
    return fb.unsqueeze(0)  # [1, T, 128]


def generate_filename(out_prob, prefix="data", extension=".wav"):
    unique_id = uuid.uuid4()
    return f"{prefix}_{unique_id}_{out_prob:.2f}_{extension}"


def load_session(model_path: str) -> ort.InferenceSession:
    return ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])


def run_inference(session: ort.InferenceSession, audio: np.ndarray, sr: int):
    audio = torch.from_numpy(audio.astype(np.float32)[None, :])  # (1, T)
    feats = wav_to_fbank(audio, sr)
    feats = feats[None, :, :].numpy()
    input_name = session.get_inputs()[0].name
    return session.run(None, {input_name: feats})[0]  # first output


def make_callback(session, sr, window_sec=1.02, save_path=None, rs=None):
    """Return a callback that keeps a 1.02‑second rolling buffer and
    runs the model every window_sec."""
    buf_len = int(sr * window_sec)
    buffer = np.zeros(buf_len, dtype=np.float32)

    def callback(indata, frames, time_info, status):
        nonlocal buffer
        if status:
            print("\033[KStatus:", status, flush=True)

        # add new audio and keep last `buf_len` samples
        buffer = np.concatenate([buffer, indata[:, 0]]).astype(np.float32)
        if buffer.shape[0] < buf_len:
            return
        buffer = buffer[-buf_len:]

        # Inference
        output = run_inference(session, buffer.copy(), sr)[0][
            0
        ]  # Function returns list[list[logit]]]
        output_prob = sigmoid(output)
        print(f"\rProbability of drone sound: {output_prob:.2f}", end="", flush=True)
        if output_prob > 0.03:  # fixed threshold to 0.5 for now
            print("==Drone detected==")
        # Optional: save the window that was just scored & add stats to recognition_summary.json
        if save_path:
            os.makedirs(save_path, exist_ok=True)
            fn = os.path.join(save_path, generate_filename(output_prob))
            sf.write(fn, buffer, sr)
            if rs is not None and (output_prob > 0.4 and output_prob < 0.6):
                rs.append(
                    {
                        "path": fn,
                        "prob": output_prob,
                        "time": datetime.datetime.now().isoformat(),
                    }
                )

    return callback


# Setup and run the streaming
def main(args):
    rec_summary = []
    summary_path = os.path.join(args.save_path, "recognition_summary.json")
    if args.save_path and os.path.exists(summary_path):
        with open(summary_path, "r") as f:
            rec_summary = json.load(f)
    session = load_session(args.model_path)
    cb = make_callback(
        session, sr=args.sample_rate, save_path=args.save_path, rs=rec_summary
    )

    print("#" * 80)
    print("Listening…  Press Ctrl‑C to quit.")
    print("#" * 80)

    try:
        with sd.InputStream(
            callback=cb,
            dtype="float32",
            channels=1,
            samplerate=args.sample_rate,
            blocksize=0,
        ):  # let PortAudio pick
            sd.sleep(int(1e9))
    except KeyboardInterrupt:
        print("\nStopped.")
        if args.save_path:
            with open(summary_path, "w") as f:
                json.dump(rec_summary, f, indent=4)
    except Exception as e:
        print(f"\nError: {e}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True, help="onnx model")
    p.add_argument(
        "--save_path", default="infer_samples/", help="Where to save inferred samples"
    )
    p.add_argument("--sample_rate", type=int, default=16000)
    args = p.parse_args()

    main(args)
