import random
from collections import Counter
from statistics import mean, stdev

import torch
import torchaudio
from tabulate import tabulate
from torch.utils.data import Dataset

SR = 16000
N_MELS = 128
NORM_MEAN = -4.268
NORM_STD = 4.569
MIN_SEC, MAX_SEC = 0.4, 15.0  # keep ONLY within this band
TARGET_SEC = 1.0  # 1‑second chunks
TARGET_FRAMES = int(TARGET_SEC * SR)
SEED = 42


def create_train_val_test_splits(data, args):
    # ---------------------- stratified two-stage split ---------------------
    # stage 1: separate out test
    first = data.train_test_split(
        test_size=args.test_frac,
        stratify_by_column="label",
        seed=SEED,
    )
    temp_train = first["train"]
    test_ds = first["test"]

    val_frac_within_temp = args.val_frac / (
        args.train_frac + args.val_frac
    )  # since test already removed
    second = temp_train.train_test_split(
        test_size=val_frac_within_temp,
        stratify_by_column="label",
        seed=SEED,
    )
    train_ds = second["train"]
    val_ds = second["test"]

    return train_ds, val_ds, test_ds


def pad_collate(batch):
    fbs, ys, lens = [], [], []
    for fb, y in batch:
        fbs.append(fb)
        ys.append(y)
        lens.append(fb.shape[1])

    T_max = max(lens)
    out = torch.zeros(len(batch), 1, T_max, N_MELS)
    for i, fb in enumerate(fbs):
        out[i, :, : fb.shape[1]] = fb
    return out, torch.tensor(ys)


def wav_to_fbank(wav: torch.Tensor, sr: int) -> torch.Tensor:
    if sr != SR:
        wav = torchaudio.functional.resample(wav, sr, SR)
    wav -= wav.mean()

    fb = torchaudio.compliance.kaldi.fbank(
        wav.unsqueeze(0),
        htk_compat=True,
        sample_frequency=SR,
        num_mel_bins=N_MELS,
        use_energy=False,
        window_type="hanning",
        dither=0.0,
        frame_shift=10,
    )  # [T , 128]
    fb = (fb - NORM_MEAN) / (2 * NORM_STD)
    return fb.unsqueeze(0)  # [1, T, 128]


def apply_spec_augment(
    fb: torch.Tensor,
    freq_mask_param: int = 24,
    time_mask_param: int = 80,
    num_freq_masks: int = 2,
    num_time_masks: int = 2,
    p: float = 0.5,
) -> torch.Tensor:
    """
    Randomly masks horizontal (time) and vertical (frequency) stripes on an
    fbank tensor [T, F].
    """
    if p == 0.0 or random.random() > p:
        return fb
    _, T, F = fb.shape
    fb_aug = fb.clone().squeeze()

    # --- Frequency masks ---------------------------------------------------
    for _ in range(num_freq_masks):
        f = random.randrange(0, freq_mask_param + 1)
        if f == 0 or f >= F:
            continue
        f0 = random.randrange(0, F - f)
        fb_aug[:, f0 : f0 + f] = 0.0

    # --- Time masks --------------------------------------------------------
    for _ in range(num_time_masks):
        t = random.randrange(0, time_mask_param + 1)
        if t == 0 or t >= T:
            continue
        t0 = random.randrange(0, T - t)
        fb_aug[t0 : t0 + t, :] = 0.0

    return fb_aug.unsqueeze(0)


class HFDatasetWrapper(Dataset):
    """
    Converts one HF example → fbank tensor + label.
    Works with DataLoader num_workers > 0.
    """

    def __init__(self, hf_ds, spec_aug=False):
        self.ds = hf_ds
        self.spec_aug = spec_aug

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        ex = self.ds[idx]  # ONE row
        wav = torch.from_numpy(ex["audio"]["array"]).float()
        sr = ex["audio"]["sampling_rate"]
        label = ex["label"]

        fb = wav_to_fbank(wav, sr)  # Tensor [T, F]

        if self.spec_aug:
            fb = apply_spec_augment(fb)

        return fb, label


def keep(example):
    dur = len(example["audio"]["array"]) / example["audio"]["sampling_rate"]
    return MIN_SEC <= dur <= MAX_SEC


# -----------------------------------------------------------------------
def filter_and_chunk(batch):
    new_audio, new_label = [], []
    sr = SR
    for wav, label in zip(batch["audio"], batch["label"]):
        dur_sec = len(wav["array"]) / sr
        if dur_sec < MIN_SEC:
            continue  # drop this clip entirely

        if label == 1:  # keep positives as‑is
            new_audio.append({"array": wav["array"], "sampling_rate": sr})
            new_label.append(label)
        else:  # chunk negatives
            n_chunks = len(wav["array"]) // TARGET_FRAMES
            for i in range(n_chunks):
                start = i * TARGET_FRAMES
                end = start + TARGET_FRAMES
                chunk = wav["array"][start:end]
                if len(chunk) == TARGET_FRAMES:
                    new_audio.append({"array": chunk, "sampling_rate": sr})
                    new_label.append(label)

    return {"audio": new_audio, "label": new_label}


def prepare_split(split: Dataset, num_proc: int) -> Dataset:
    return split.map(
        filter_and_chunk,
        batched=True,
        remove_columns=["audio", "label"],
        num_proc=num_proc,
    )


def split_stats(train_ds, val_ds, test_ds):
    """
    For every split show:
        #neg, #pos, total, pos‑ratio, mean‑seconds, std‑seconds, min, max
    Assumes each example has an 'audio' column decoded as HF Audio().
    """
    rows = []
    hdr = ["split", "neg", "pos", "total", "pos %", "mean s", "std s", "min s", "max s"]

    for name, split in [("train", train_ds), ("val", val_ds), ("test", test_ds)]:
        # ---------- label counts ----------
        cnt = Counter(split["label"])
        n_neg = cnt.get(0, 0)
        n_pos = cnt.get(1, 0)
        total = n_neg + n_pos

        # ---------- audio durations ----------
        # (array length) / (sampling rate) per clip
        durations = [
            len(row["audio"]["array"]) / row["audio"]["sampling_rate"] for row in split
        ]
        m, sd = mean(durations), (stdev(durations) if len(durations) > 1 else 0)
        rows.append(
            [
                name,
                n_neg,
                n_pos,
                total,
                f"{n_pos / total:.2%}",
                f"{m:.2f}",
                f"{sd:.2f}",
                f"{min(durations):.2f}",
                f"{max(durations):.2f}",
            ]
        )

    print(tabulate(rows, headers=hdr, tablefmt="github"))
