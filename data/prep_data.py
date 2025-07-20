import argparse
import random

import numpy as np
from datasets import Audio, DatasetDict, load_dataset

from data.utils import create_train_val_test_splits, keep, prepare_split


def prepare_geronimo_dataset(args):
    """
    End‑to‑end pipeline for the “drone‑audio‑detection‑samples” HF dataset.

    Steps
    -----
    1. Download & cache the raw training split.
    2. Resample/normalize the `audio` column to `args.sr` Hz.
    3. Chunk the waveform into uniform windows via `prepare_split`.
    4. Filter away unusable chunks (`keep`).
    5. Create train/val/test splits with `create_train_val_test_splits`.
    6. Persist as a `DatasetDict` on disk.

    Parameters
    ----------
    args.sr : int
        Target sampling rate.
    args.output_dir : Path | str
        Destination directory for `save_to_disk`.
    ... any other args you’re using ...
    """
    random.seed(args.seed)
    np.random.seed(args.seed)
    data = load_dataset("geronimobasso/drone-audio-detection-samples", split="train")
    data = data.cast_column("audio", Audio(sampling_rate=args.sr, decode=True))
    data_chunked = prepare_split(data, args.num_proc)
    data_chunked_filtered = data_chunked.filter(keep, num_proc=args.num_proc)
    train_ds, val_ds, test_ds = create_train_val_test_splits(
        data_chunked_filtered, args
    )
    ds = [train_ds, val_ds, test_ds]

    ds_dict = DatasetDict({split: d for split, d in zip(("train", "val", "test"), ds)})

    ds_dict.save_to_disk(str(args.output_dir))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", required=True, default=42)
    p.add_argument("--num_proc", default=4)
    p.add_argument("--seed", default=42)
    p.add_argument("--train_frac", default=0.8)
    p.add_argument("--val_frac", default=0.1)
    p.add_argument("--test_frac", default=0.1)
    p.add_argument("--sr", default=16000)
    args = p.parse_args()

    prepare_geronimo_dataset(args)
