from __future__ import annotations
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import random
import numpy as np
import torch
import torch.nn.functional as F

try:
    from torchmetrics.retrieval import RetrievalRecall, RetrievalNormalizedDCG
except Exception:
    RetrievalRecall = None
    RetrievalNormalizedDCG = None
from torch.utils.data import DataLoader
from tqdm import tqdm

from esasrec.data import ShiftedSequenceDataset, prepare_ml20m
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
    if RetrievalRecall is None or RetrievalNormalizedDCG is None:
        hits = topk.eq(targets.unsqueeze(1))
        recall = float(hits.any(dim=1).float().mean().item())
        ndcg_vals = torch.zeros(topk.size(0), dtype=torch.float32, device=device)
        hit_pos = hits.float().argmax(dim=1)
        has_hit = hits.any(dim=1)
        ndcg_vals[has_hit] = 1.0 / torch.log2(hit_pos[has_hit].float() + 2.0)
        ndcg = float(ndcg_vals.mean().item())
        return {"recall@10": recall, "ndcg@10": ndcg, "metrics_backend": "manual_fallback"}

    n_users, k = topk.shape
    indexes = torch.arange(n_users, device=device).repeat_interleave(k)
    preds = torch.arange(k, 0, -1, dtype=torch.float32, device=device).repeat(n_users)
    target = topk.eq(targets.unsqueeze(1)).reshape(-1).int()
    recall_metric = RetrievalRecall(top_k=k).to(device)
    ndcg_metric = RetrievalNormalizedDCG(top_k=k).to(device)
    recall = float(recall_metric(preds, target, indexes=indexes).item())
    ndcg = float(ndcg_metric(preds, target, indexes=indexes).item())
    return {"recall@10": recall, "ndcg@10": ndcg, "metrics_backend": "torchmetrics"}

@torch.no_grad()
def evaluate(model: ESASRec, histories: list[list[int]], targets: dict[int, int], max_len: int, batch_size: int, device: torch.device, max_users: int | None = None) -> dict[str, float]:
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
            seen = set(histories[uid - 1])
            seen = torch.tensor(list(seen), device=device) - 1
            seen_mask[j, seen] = True
        logits = logits.masked_fill(seen_mask, float('-inf'))
        topk = torch.topk(logits, k=10, dim=1).indices + 1
        all_topk.append(topk)
        all_targets.append(torch.tensor([targets[uid] for uid in ub], device=device))

    topk_all = torch.cat(all_topk, dim=0)
    targets_all = torch.cat(all_targets, dim=0)
    return compute_topk_metrics(topk_all, targets_all, device)


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
    args = p.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)

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
        val = evaluate(model, val_hist, prep.val_targets, args.max_len, args.batch_size, device, args.max_eval_users)
        print(f"val: recall@10={val['recall@10']:.4f} ndcg@10={val['ndcg@10']:.4f}")
        if val["ndcg@10"] > best_val:
            best_val = val["ndcg@10"]
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    loss_trend_ok = True
    if len(loss_checkpoints) >= 2:
        loss_trend_ok = loss_checkpoints[-1][1] <= loss_checkpoints[0][1]
    print(f"loss_trend_ok={int(loss_trend_ok)} checkpoints={len(loss_checkpoints)}")
    test_hist = [s + [prep.val_targets[i + 1]] for i, s in enumerate(prep.train_sequences)]
    test = evaluate(model, test_hist, prep.test_targets, args.max_len, args.batch_size, device, args.max_eval_users)
    print(f"test: recall@10={test['recall@10']:.4f} ndcg@10={test['ndcg@10']:.4f}")

    report = {
        "config": vars(args),
        "model": asdict(cfg),
        "test_metrics": {"recall@10": test["recall@10"], "ndcg@10": test["ndcg@10"]},
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
