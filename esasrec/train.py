from __future__ import annotations
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import random
import numpy as np
import torch
import torch.nn.functional as F

from torch.utils.data import DataLoader
from tqdm import tqdm

from esasrec.data import ShiftedSequenceDataset, prepare_ml20m, prepare_ml20m_realistic
from esasrec.model import ESASRec, ModelConfig


ART_METRICS = {
    "ml20m_our_experiments": {
        "sasrec_ss": {"recall@10": 0.313, "ndcg@10": 0.183},
        "esasrec": {"recall@10": 0.329, "ndcg@10": 0.197},
    }
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sampled_softmax_loss(model: ESASRec, h: torch.Tensor, tgt: torch.Tensor, n_items: int, n_neg: int) -> torch.Tensor:
    m = tgt > 0
    if not torch.any(m):
        return h.new_zeros(())
    h = h[m]
    pos = tgt[m]
    h = h.float()
    pos_e = model.item_emb(pos).float()
    pos_logits = (h * pos_e).sum(-1, keepdim=True)

    neg = torch.randint(1, n_items + 1, (h.size(0), n_neg), device=h.device)
    clash = neg.eq(pos.unsqueeze(1))
    while clash.any():
        neg[clash] = torch.randint(1, n_items + 1, (int(clash.sum().item()),), device=h.device)
        clash = neg.eq(pos.unsqueeze(1))
    neg_e = model.item_emb(neg).float()
    neg_logits = torch.einsum("bd,bnd->bn", h, neg_e)
    logits = torch.cat([pos_logits, neg_logits], dim=1).clamp_(-50.0, 50.0)
    labels = torch.zeros(logits.size(0), dtype=torch.long, device=h.device)
    return F.cross_entropy(logits, labels)




def compute_topk_metrics(topk: torch.Tensor, targets: torch.Tensor, device: torch.device) -> dict[str, float]:
    hits = topk.eq(targets.unsqueeze(1))
    recall = float(hits.any(dim=1).float().mean().item())
    ndcg_vals = torch.zeros(topk.size(0), dtype=torch.float32, device=device)
    hit_pos = hits.float().argmax(dim=1)
    has_hit = hits.any(dim=1)
    ndcg_vals[has_hit] = 1.0 / torch.log2(hit_pos[has_hit].float() + 2.0)
    ndcg = float(ndcg_vals.mean().item())
    return {"recall@10": recall, "ndcg@10": ndcg, "metrics_backend": "full_catalog"}


def compute_sampled_metrics(ranks: np.ndarray) -> dict[str, float]:
    hr1 = float(np.mean(ranks <= 1))
    hr5 = float(np.mean(ranks <= 5))
    hr10 = float(np.mean(ranks <= 10))
    ndcg5 = float(np.mean(np.where(ranks <= 5, 1.0 / np.log2(ranks + 1.0), 0.0)))
    ndcg10 = float(np.mean(np.where(ranks <= 10, 1.0 / np.log2(ranks + 1.0), 0.0)))
    mrr = float(np.mean(1.0 / ranks))
    return {
        "hr@1": hr1,
        "hr@5": hr5,
        "hr@10": hr10,
        "ndcg@5": ndcg5,
        "ndcg@10": ndcg10,
        "mrr": mrr,
        "metrics_backend": "s3rec_sampled_100n" if len(ranks) > 0 else "s3rec_sampled",
    }

@torch.no_grad()
def evaluate_full_catalog(model: ESASRec, histories: list[list[int]], targets: dict[int, int], max_len: int, batch_size: int, device: torch.device, max_users: int | None = None) -> dict[str, float]:
    model.eval()
    users = list(range(1, len(histories) + 1))
    if max_users is not None:
        users = users[:max_users]
    item_w = model.item_emb.weight[1:]
    all_topk, all_targets = [], []

    for i in range(0, len(users), batch_size):
        ub = users[i:i + batch_size]
        x = torch.zeros((len(ub), max_len), dtype=torch.long, device=device)
        for j, uid in enumerate(ub):
            seq = histories[uid - 1][-max_len:]
            x[j, -len(seq):] = torch.tensor(seq, device=device)

        h = model.encode(x)
        idx = (x.ne(0).sum(1) - 1).clamp_min(0)
        u = h[torch.arange(h.size(0), device=device), idx]
        logits = u @ item_w.t()
        seen_mask = torch.zeros_like(logits, dtype=torch.bool)
        for j, uid in enumerate(ub):
            seen = torch.tensor(list(set(histories[uid - 1])), device=device) - 1
            seen_mask[j, seen] = True
        logits = logits.masked_fill(seen_mask, float('-inf'))
        topk = torch.topk(logits, k=10, dim=1).indices + 1
        all_topk.append(topk)
        all_targets.append(torch.tensor([targets[uid] for uid in ub], device=device))

    topk_all = torch.cat(all_topk, dim=0)
    targets_all = torch.cat(all_targets, dim=0)
    return compute_topk_metrics(topk_all, targets_all, device)


@torch.no_grad()
def evaluate_s3rec_sampled(model: ESASRec, histories: list[list[int]], targets: dict[int, int], max_len: int, batch_size: int, device: torch.device, n_items: int, n_sampled_negatives: int = 100, max_users: int | None = None, seed: int = 42) -> dict[str, float]:
    model.eval()
    rng = np.random.default_rng(seed)
    users = list(range(1, len(histories) + 1))
    if max_users is not None:
        users = users[:max_users]
    item_w = model.item_emb.weight
    ranks: list[int] = []

    for i in range(0, len(users), batch_size):
        ub = users[i:i + batch_size]
        x = torch.zeros((len(ub), max_len), dtype=torch.long, device=device)
        cands = torch.zeros((len(ub), n_sampled_negatives + 1), dtype=torch.long, device=device)
        for j, uid in enumerate(ub):
            seq = histories[uid - 1][-max_len:]
            x[j, -len(seq):] = torch.tensor(seq, device=device)
            gt = int(targets[uid])
            seen = set(histories[uid - 1])
            seen.add(gt)
            negs = []
            while len(negs) < n_sampled_negatives:
                cand = int(rng.integers(1, n_items + 1))
                if cand not in seen:
                    negs.append(cand)
                    seen.add(cand)
            cand_list = [gt] + negs
            cands[j] = torch.tensor(cand_list, dtype=torch.long, device=device)

        h = model.encode(x)
        idx = (x.ne(0).sum(1) - 1).clamp_min(0)
        u = h[torch.arange(h.size(0), device=device), idx]
        cand_emb = item_w[cands]
        scores = torch.einsum('bd,bkd->bk', u, cand_emb)
        order = torch.argsort(scores, dim=1, descending=True)
        pos = (order == 0).nonzero(as_tuple=False)[:, 1] + 1
        ranks.extend(pos.detach().cpu().tolist())

    return compute_sampled_metrics(np.asarray(ranks, dtype=np.float64))




@torch.no_grad()
def evaluate_realistic_time_split(model: ESASRec, histories: list[list[int]], test_relevant_items: dict[int, set[int]], max_len: int, batch_size: int, device: torch.device, n_items: int, max_users: int | None = None) -> dict[str, float]:
    model.eval()
    users = list(sorted(test_relevant_items.keys()))
    if max_users is not None:
        users = users[:max_users]
    item_w = model.item_emb.weight[1:]
    unique_recs: set[int] = set()
    ndcgs = []

    for i in range(0, len(users), batch_size):
        ub = users[i:i + batch_size]
        x = torch.zeros((len(ub), max_len), dtype=torch.long, device=device)
        for j, uid in enumerate(ub):
            seq = histories[uid - 1][-max_len:]
            x[j, -len(seq):] = torch.tensor(seq, device=device)

        h = model.encode(x)
        idx = (x.ne(0).sum(1) - 1).clamp_min(0)
        u = h[torch.arange(h.size(0), device=device), idx]
        logits = u @ item_w.t()
        seen_mask = torch.zeros_like(logits, dtype=torch.bool)
        for j, uid in enumerate(ub):
            seen = torch.tensor(list(set(histories[uid - 1])), device=device) - 1
            seen_mask[j, seen] = True
        logits = logits.masked_fill(seen_mask, float('-inf'))
        topk = torch.topk(logits, k=10, dim=1).indices + 1

        for j, uid in enumerate(ub):
            recs = topk[j].detach().cpu().numpy().tolist()
            unique_recs.update(recs)
            rel = test_relevant_items[uid]
            dcg = 0.0
            for rank, it in enumerate(recs, start=1):
                if it in rel:
                    dcg += 1.0 / np.log2(rank + 1.0)
            ideal_len = min(len(rel), 10)
            idcg = sum(1.0 / np.log2(r + 1.0) for r in range(1, ideal_len + 1))
            ndcgs.append((dcg / idcg) if idcg > 0 else 0.0)

    coverage = float(len(unique_recs) / max(n_items, 1))
    return {"ndcg@10": float(np.mean(ndcgs) if ndcgs else 0.0), "coverage@10": coverage, "metrics_backend": "realistic_time_split"}

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ratings-path", type=str, required=True)
    p.add_argument("--workdir", type=str, default="./artifacts")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-len", type=int, default=200)
    p.add_argument("--emb-dim", type=int, default=256)
    p.add_argument("--n-heads", type=int, default=8)
    p.add_argument("--n-blocks", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--ff-mult", type=int, default=4)
    p.add_argument("--n-neg", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--max-train-users", type=int, default=None)
    p.add_argument("--max-eval-users", type=int, default=None)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--eval-protocol", type=str, default="realistic", choices=["full", "s3rec", "realistic"])
    p.add_argument("--eval-sampled-negatives", type=int, default=100)
    p.add_argument("--test-days", type=int, default=60)
    args = p.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    if args.eval_protocol == "realistic":
        prep = prepare_ml20m_realistic(args.ratings_path, test_days=args.test_days, max_users=args.max_train_users)
    else:
        prep = prepare_ml20m(args.ratings_path, max_users=args.max_train_users)
    ds = ShiftedSequenceDataset(prep.train_sequences, args.max_len)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=device.type == "cuda", persistent_workers=args.num_workers > 0)

    cfg = ModelConfig(n_items=prep.n_items, max_len=args.max_len, emb_dim=args.emb_dim, n_heads=args.n_heads, n_blocks=args.n_blocks, dropout=args.dropout, ff_mult=args.ff_mult)
    model = ESASRec(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")

    best_val = -1.0
    best_state = None

    global_step = 0
    loss_checkpoints = []
    for ep in range(1, args.epochs + 1):
        model.train()
        bar = tqdm(dl, desc=f"epoch {ep}/{args.epochs}")
        losses = []
        for x, y in bar:
            global_step += 1
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=args.amp and device.type == "cuda"):
                h = model.encode(x)
                loss = sampled_softmax_loss(model, h, y, prep.n_items, args.n_neg)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at epoch={ep} step={global_step}")
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(opt)
            scaler.update()
            loss_val = float(loss.item())
            losses.append(loss_val)
            if global_step % args.log_every == 0:
                recent = float(np.mean(losses[-args.log_every:]))
                loss_checkpoints.append((global_step, recent))
                print(f"loss_check step={global_step} mean_last={recent:.6f} finite=1")
            bar.set_postfix(loss=f"{np.mean(losses):.4f}")

        if not losses:
            raise RuntimeError("all training batches produced non-finite loss")
        print(f"epoch_done {ep} loss_mean={float(np.mean(losses)):.6f}")

        val_hist = prep.train_sequences
        if args.eval_protocol == "s3rec":
            val = evaluate_s3rec_sampled(model, val_hist, prep.val_targets, args.max_len, args.batch_size, device, prep.n_items, args.eval_sampled_negatives, args.max_eval_users, args.seed + ep)
            print(f"val: hr@10={val['hr@10']:.4f} ndcg@10={val['ndcg@10']:.4f} mrr={val['mrr']:.4f}")
            val_key = val["ndcg@10"]
        else:
            val = evaluate_full_catalog(model, val_hist, prep.val_targets, args.max_len, args.batch_size, device, args.max_eval_users)
            print(f"val: recall@10={val['recall@10']:.4f} ndcg@10={val['ndcg@10']:.4f}")
            val_key = val["ndcg@10"]
        if val_key > best_val:
            best_val = val["ndcg@10"]
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    loss_trend_ok = True
    if len(loss_checkpoints) >= 2:
        loss_trend_ok = loss_checkpoints[-1][1] <= loss_checkpoints[0][1]
    print(f"loss_trend_ok={int(loss_trend_ok)} checkpoints={len(loss_checkpoints)}")
    if args.eval_protocol == "realistic":
        test = evaluate_realistic_time_split(model, prep.train_sequences, prep.test_relevant_items, args.max_len, args.batch_size, device, prep.n_items, args.max_eval_users)
        print(f"test: ndcg@10={test['ndcg@10']:.4f} coverage@10={test['coverage@10']:.4f}")
    else:
        test_hist = [s + [prep.val_targets[i + 1]] for i, s in enumerate(prep.train_sequences)]
        if args.eval_protocol == "s3rec":
            test = evaluate_s3rec_sampled(model, test_hist, prep.test_targets, args.max_len, args.batch_size, device, prep.n_items, args.eval_sampled_negatives, args.max_eval_users, args.seed + 10_000)
            print(f"test: hr@1={test['hr@1']:.4f} hr@5={test['hr@5']:.4f} hr@10={test['hr@10']:.4f} ndcg@10={test['ndcg@10']:.4f} mrr={test['mrr']:.4f}")
        else:
            test = evaluate_full_catalog(model, test_hist, prep.test_targets, args.max_len, args.batch_size, device, args.max_eval_users)
            print(f"test: recall@10={test['recall@10']:.4f} ndcg@10={test['ndcg@10']:.4f}")

    report = {
        "config": vars(args),
        "model": asdict(cfg),
        "test_metrics": {k: float(v) for k, v in test.items() if k != "metrics_backend"},
        "metrics_backend": test.get("metrics_backend", "unknown"),
        "article_reference": ART_METRICS,
        "loss_checkpoints": [{"step": s, "mean_last": v} for s, v in loss_checkpoints],
        "loss_trend_ok": bool(loss_trend_ok),
    }
    (workdir / "metrics.json").write_text(json.dumps(report, indent=2))
    torch.save(model.state_dict(), workdir / "esasrec.pt")
    print("train_finished")


if __name__ == "__main__":
    main()
