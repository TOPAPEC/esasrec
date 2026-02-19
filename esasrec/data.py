from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


@dataclass
class PreparedData:
    train_sequences: list[list[int]]
    val_targets: dict[int, int]
    test_targets: dict[int, int]
    n_items: int
    n_users: int


def prepare_ml20m(ratings_path: str, min_user: int = 5, min_item: int = 5, max_users: int | None = None) -> PreparedData:
    df = pd.read_csv(ratings_path, usecols=["userId", "movieId", "timestamp"])
    while True:
        uc = df["userId"].value_counts()
        ic = df["movieId"].value_counts()
        nxt = df[df["userId"].isin(uc[uc >= min_user].index) & df["movieId"].isin(ic[ic >= min_item].index)]
        if len(nxt) == len(df):
            break
        df = nxt
    if max_users is not None:
        users = df["userId"].value_counts().index[:max_users]
        df = df[df["userId"].isin(users)]

    df = df.sort_values(["userId", "timestamp", "movieId"])
    users = df["userId"].unique()
    items = df["movieId"].unique()
    u2i = {u: i + 1 for i, u in enumerate(users)}
    it2i = {it: i + 1 for i, it in enumerate(items)}

    seqs: dict[int, list[int]] = {}
    for row in df.itertuples(index=False):
        uid = u2i[row.userId]
        iid = it2i[row.movieId]
        seqs.setdefault(uid, []).append(iid)

    train_sequences: list[list[int]] = []
    val_targets: dict[int, int] = {}
    test_targets: dict[int, int] = {}
    for uid in sorted(seqs):
        s = seqs[uid]
        if len(s) < 3:
            continue
        train_sequences.append(s[:-2])
        val_targets[uid] = s[-2]
        test_targets[uid] = s[-1]
    return PreparedData(train_sequences, val_targets, test_targets, n_items=len(items), n_users=len(train_sequences))


class ShiftedSequenceDataset(Dataset):
    def __init__(self, train_sequences: list[list[int]], max_len: int):
        self.max_len = max_len
        self.data = []
        for s in train_sequences:
            s = s[-(max_len + 1):]
            inp = s[:-1]
            tgt = s[1:]
            self.data.append((inp, tgt))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        inp, tgt = self.data[idx]
        x = np.zeros(self.max_len, dtype=np.int64)
        y = np.zeros(self.max_len, dtype=np.int64)
        x[-len(inp):] = inp
        y[-len(tgt):] = tgt
        return torch.from_numpy(x), torch.from_numpy(y)


def save_meta(path: str | Path, prep: PreparedData) -> None:
    meta = {
        "n_items": prep.n_items,
        "n_users": prep.n_users,
        "val_targets": prep.val_targets,
        "test_targets": prep.test_targets,
    }
    Path(path).write_text(json.dumps(meta))
