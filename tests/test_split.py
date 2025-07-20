import random
from collections import Counter
from types import SimpleNamespace

import numpy as np
import pytest
from datasets import Audio, ClassLabel, Dataset, Features

from data.utils import create_train_val_test_splits

SR = 16000


def _rand_audio(sec: float):
    n = int(sec * SR)
    return {
        "array": np.random.randn(n).astype(np.float32),
        "sampling_rate": SR,
    }


@pytest.fixture(scope="session")
def tiny_hf_dataset():
    rng = random.Random(0)
    rows = [
        {"audio": _rand_audio(rng.uniform(0.5, 2.0)), "label": i % 2} for i in range(40)
    ]

    features = Features(
        {
            "audio": Audio(sampling_rate=SR),
            "label": ClassLabel(num_classes=2, names=["neg", "pos"]),
        }
    )

    return Dataset.from_list(rows, features=features)


def test_stratified_split_balanced(tiny_hf_dataset):
    args = SimpleNamespace(train_frac=0.7, val_frac=0.15, test_frac=0.15)
    tr, va, te = create_train_val_test_splits(tiny_hf_dataset, args)

    # --- size sanity ---
    assert len(tr) + len(va) + len(te) == len(tiny_hf_dataset)

    for split in [tr, va, te]:
        cnt = Counter(split["label"])
        # perfect balance preserved
        assert cnt[0] == cnt[1]

    # exact sizes (because dataset is small & balanced)
    assert len(te) == 6  # 15 % of 40
    assert len(va) == 6
    assert len(tr) == 28
