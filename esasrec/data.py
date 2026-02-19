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


@dataclass
class PreparedRealisticData:
    train_sequences: list[list[int]]
    val_targets: dict[int, int]
    test_relevant_items: dict[int, set[int]]
    n_items: int
    n_users: int


def _iterative_kcore(df: pd.DataFrame, min_user: int, min_item: int) -> pd.DataFrame:
    while True:
        uc = df["userId"].value_counts()
        ic = df["movieId"].value_counts()
        nxt = df[df["userId"].isin(uc[uc >= min_user].index) & df["movieId"].isin(ic[ic >= min_item].index)]
        if len(nxt) == len(df):
            return df
        df = nxt


def prepare_ml20m(ratings_path: str, min_user: int = 5, min_item: int = 5, max_users: int | None = None) -> PreparedData:
    df = pd.read_csv(ratings_path, usecols=["userId", "movieId", "timestamp"])
    df = _iterative_kcore(df, min_user=min_user, min_item=min_item)
    if max_users is not None:
        users = df["userId"].value_counts().index[:max_users]
        df = df[df["userId"].isin(users)]

    df = df.sort_values(["userId", "timestamp", "movieId"])  # legacy path
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


def prepare_ml20m_realistic(
    ratings_path: str,
    test_days: int = 60,
    min_user: int = 2,
    min_item: int = 5,
    max_users: int | None = None,
    filter_train_pairs_from_test: bool = True,
) -> PreparedRealisticData:
    df = pd.read_csv(ratings_path, usecols=["userId", "movieId", "timestamp"])
    df = _iterative_kcore(df, min_user=min_user, min_item=min_item)
    if max_users is not None:
        users = df["userId"].value_counts().index[:max_users]
        df = df[df["userId"].isin(users)]

    df = df.reset_index(drop=False).rename(columns={"index": "orig_idx"})
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)

    split_ts = df["datetime"].max() - pd.Timedelta(days=test_days)
    train_df = df[df["datetime"] < split_ts].copy()
    test_df = df[df["datetime"] >= split_ts].copy()

    train_users = set(train_df["userId"].unique())
    train_items = set(train_df["movieId"].unique())
    test_df = test_df[test_df["userId"].isin(train_users) & test_df["movieId"].isin(train_items)]
    if filter_train_pairs_from_test:
        seen_pairs = set(zip(train_df["userId"], train_df["movieId"]))
        mask = [(u, i) not in seen_pairs for u, i in zip(test_df["userId"], test_df["movieId"])]
        test_df = test_df[mask]

    used_users = sorted(set(train_df["userId"]).intersection(set(test_df["userId"])))
    train_df = train_df[train_df["userId"].isin(used_users)].copy()
    test_df = test_df[test_df["userId"].isin(used_users)].copy()

    train_items_sorted = sorted(train_df["movieId"].unique())
    u2i = {u: i + 1 for i, u in enumerate(used_users)}
    it2i = {it: i + 1 for i, it in enumerate(train_items_sorted)}

    train_df["uid"] = train_df["userId"].map(u2i)
    train_df["iid"] = train_df["movieId"].map(it2i)
    test_df["uid"] = test_df["userId"].map(u2i)
    test_df["iid"] = test_df["movieId"].map(it2i)

    train_df = train_df.sort_values(["uid", "datetime", "orig_idx"])
    train_sequences_by_uid: dict[int, list[int]] = {}
    for row in train_df[["uid", "iid"]].itertuples(index=False):
        train_sequences_by_uid.setdefault(int(row.uid), []).append(int(row.iid))

    val_targets: dict[int, int] = {}
    train_sequences: list[list[int]] = []
    uid_order = sorted(train_sequences_by_uid)
    for uid in uid_order:
        seq = train_sequences_by_uid[uid]
        if len(seq) < 2:
            continue
        val_targets[uid] = seq[-1]
        train_sequences.append(seq[:-1])

    valid_uids = set(val_targets.keys())
    test_relevant_items: dict[int, set[int]] = {}
    for row in test_df[["uid", "iid"]].itertuples(index=False):
        uid = int(row.uid)
        if uid in valid_uids:
            test_relevant_items.setdefault(uid, set()).add(int(row.iid))

    kept_uids = sorted(set(test_relevant_items).intersection(valid_uids))
    uid_to_pos = {uid: i for i, uid in enumerate(uid_order)}
    train_sequences = [train_sequences[uid_to_pos[uid]] for uid in kept_uids]
    val_targets = {uid: val_targets[uid] for uid in kept_uids}
    test_relevant_items = {uid: test_relevant_items[uid] for uid in kept_uids}

    return PreparedRealisticData(
        train_sequences=train_sequences,
        val_targets=val_targets,
        test_relevant_items=test_relevant_items,
        n_items=len(train_items_sorted),
        n_users=len(train_sequences),
    )


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
