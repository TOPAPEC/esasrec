import argparse
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from sasrec.data import (
    download_and_preprocess,
    load_data,
    split_leave_one_out,
    CausalLMDataset,
    PaddingCollateFn,
)
from sasrec.losses import compute_sampled_ce_loss
from sasrec_reasoning.model import SASRecReasoning
from sasrec_reasoning.losses import compute_erl_sampled_loss, compute_prl_sampled_loss


def train_one_epoch_reasoning(model, dataloader, optimizer, device, num_items,
                               mode='erl', reason_loss_weight=0.1,
                               num_neg_reason=256,
                               kl_weight=0.05, kl_temperature=1.0,
                               temp_scale=5.0, pl_weight=1.0):
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)
        negatives = batch['negatives'].to(device)

        optimizer.zero_grad()

        hidden, reasoning_out = model(input_ids)

        base_loss = compute_sampled_ce_loss(hidden, labels, negatives, model.item_emb)

        valid_mask = (labels != -100)
        last_idx = valid_mask.long().cumsum(dim=1).eq(
            valid_mask.long().cumsum(dim=1).max(dim=1, keepdim=True).values) & valid_mask
        last_targets = labels[last_idx]
        B = reasoning_out.shape[0]

        r_negs = torch.randint(1, num_items + 1, (B, num_neg_reason), device=device)
        clash = r_negs.eq(last_targets.unsqueeze(1))
        while clash.any():
            r_negs[clash] = torch.randint(1, num_items + 1, (int(clash.sum()),), device=device)
            clash = r_negs.eq(last_targets.unsqueeze(1))

        if mode == 'erl':
            reason_loss, _ = compute_erl_sampled_loss(
                reasoning_out, last_targets, r_negs, model.item_emb,
                kl_weight=kl_weight, kl_temperature=kl_temperature)
        else:
            reason_loss, _ = compute_prl_sampled_loss(
                reasoning_out, last_targets, r_negs, model.item_emb,
                temp_scale=temp_scale, pl_weight=pl_weight)

        loss = base_loss + reason_loss_weight * reason_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


@torch.no_grad()
def evaluate_reasoning(model, user_sequences, user_targets, num_items, max_length,
                       device, mode='erl', k=10, batch_size=256, filter_seen=True):
    from torch.nn.utils.rnn import pad_sequence

    model.eval()
    hit, ndcg, mrr = 0.0, 0.0, 0.0
    num_users = 0

    user_ids = list(user_targets.keys())
    for start in range(0, len(user_ids), batch_size):
        batch_users = user_ids[start:start + batch_size]

        input_seqs = []
        targets = []
        histories = []
        for uid in batch_users:
            seq = user_sequences[uid]
            input_seq = seq[-max_length:]
            input_seqs.append(torch.tensor(input_seq, dtype=torch.long))
            targets.append(user_targets[uid])
            histories.append(set(seq))

        input_ids = pad_sequence(input_seqs, batch_first=True, padding_value=0).to(device)

        _, reasoning_out = model(input_ids)

        if mode == 'erl':
            last_hidden = reasoning_out.mean(dim=1)
        else:
            last_hidden = reasoning_out[:, -1, :]

        all_scores = torch.matmul(last_hidden, model.item_emb.weight.T)
        all_scores = all_scores.cpu().numpy()

        for i, uid in enumerate(batch_users):
            scores = all_scores[i].copy()
            target = targets[i]

            if filter_seen:
                seen = np.array(list(histories[i]), dtype=np.intp)
                scores[seen] = -np.inf
            scores[0] = -np.inf

            target_score = all_scores[i][target]
            rank = (scores > target_score).sum() + 1

            if rank <= k:
                hit += 1.0
                ndcg += 1.0 / np.log2(rank + 1)
                mrr += 1.0 / rank

            num_users += 1

        if (start // batch_size) % 50 == 0:
            print(f"  Evaluated {num_users}/{len(user_ids)} users...", flush=True)

    return {
        f'HR@{k}': hit / num_users,
        f'NDCG@{k}': ndcg / num_users,
        f'MRR@{k}': mrr / num_users,
    }


@torch.no_grad()
def validate_fast_reasoning(model, val_sequences, val_targets, num_items, max_length,
                            device, mode='erl', k=10, batch_size=256, max_users=10000):
    user_ids = list(val_targets.keys())
    if len(user_ids) > max_users:
        user_ids = np.random.choice(user_ids, size=max_users, replace=False).tolist()

    subset_targets = {uid: val_targets[uid] for uid in user_ids}
    return evaluate_reasoning(model, val_sequences, subset_targets, num_items, max_length,
                              device, mode=mode, k=k, batch_size=batch_size, filter_seen=True)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path', type=str, default=None)
    parser.add_argument('--dataset', type=str, default='ml-20m',
                        choices=['ml-1m', 'ml-20m'])
    parser.add_argument('--data_dir', type=str, default='data')

    parser.add_argument('--hidden_units', type=int, default=256)
    parser.add_argument('--num_blocks', type=int, default=2)
    parser.add_argument('--num_heads', type=int, default=1)
    parser.add_argument('--dropout_rate', type=float, default=0.1)
    parser.add_argument('--max_length', type=int, default=200)

    parser.add_argument('--mode', type=str, default='erl', choices=['erl', 'prl'])
    parser.add_argument('--reason_steps', type=int, default=2)
    parser.add_argument('--reason_loss_weight', type=float, default=0.1)
    parser.add_argument('--num_neg_reason', type=int, default=256)
    parser.add_argument('--kl_weight', type=float, default=0.05)
    parser.add_argument('--kl_temperature', type=float, default=1.0)
    parser.add_argument('--temp_scale', type=float, default=5.0)
    parser.add_argument('--pl_weight', type=float, default=1.0)

    parser.add_argument('--num_negatives', type=int, default=3000)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--patience', type=int, default=10)

    parser.add_argument('--val_size', type=int, default=10000)
    parser.add_argument('--top_k', type=int, nargs='+', default=[10])

    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_dir', type=str, default='checkpoints_reasoning')

    return parser.parse_args()


def main():
    args = parse_args()
    print("=" * 70)
    print(f"SASRec + ReaRec ({args.mode.upper()}) Training")
    print("=" * 70)

    if args.data_path is None:
        args.data_path = download_and_preprocess(
            dataset_name=args.dataset, output_dir=args.data_dir)

    print(f"\nConfig:")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")
    print()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    user_sequences, num_items = load_data(args.data_path)
    (train_sequences, val_sequences, val_targets,
     test_sequences, test_targets) = split_leave_one_out(user_sequences)

    train_seqs_list = list(train_sequences.values())

    train_dataset = CausalLMDataset(
        train_seqs_list,
        max_length=args.max_length,
        num_negatives=args.num_negatives,
        full_negative_sampling=False,
        num_items=num_items,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=PaddingCollateFn(),
        pin_memory=True,
    )

    model = SASRecReasoning(
        item_num=num_items,
        maxlen=args.max_length,
        hidden_units=args.hidden_units,
        num_blocks=args.num_blocks,
        num_heads=args.num_heads,
        dropout_rate=args.dropout_rate,
        reason_steps=args.reason_steps,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    reason_params = args.reason_steps * args.hidden_units
    print(f"\nModel parameters: {total_params:,} (reasoning RPE adds {reason_params:,})")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    os.makedirs(args.save_dir, exist_ok=True)
    best_path = os.path.join(args.save_dir, f'best_model_{args.mode}.pt')
    best_ndcg = -1.0
    epochs_no_improve = 0

    print(f"\n{'='*70}")
    print(f"Starting {args.mode.upper()} training with {args.reason_steps} reasoning steps...")
    print(f"{'='*70}\n")

    start_time = time.time()

    for epoch in range(1, args.max_epochs + 1):
        epoch_start = time.time()

        avg_loss = train_one_epoch_reasoning(
            model, train_loader, optimizer, device,
            num_items=num_items,
            mode=args.mode,
            reason_loss_weight=args.reason_loss_weight,
            num_neg_reason=args.num_neg_reason,
            kl_weight=args.kl_weight,
            kl_temperature=args.kl_temperature,
            temp_scale=args.temp_scale,
            pl_weight=args.pl_weight,
        )

        epoch_time = time.time() - epoch_start

        val_metrics = validate_fast_reasoning(
            model, val_sequences, val_targets, num_items, args.max_length,
            device, mode=args.mode, k=10, batch_size=256, max_users=args.val_size)

        val_ndcg = val_metrics['NDCG@10']
        val_hr = val_metrics['HR@10']

        improved = val_ndcg > best_ndcg
        if improved:
            best_ndcg = val_ndcg
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_path)
            marker = " * (saved)"
        else:
            epochs_no_improve += 1
            marker = ""

        print(f"Epoch {epoch:3d}/{args.max_epochs} | "
              f"Loss: {avg_loss:.4f} | "
              f"Val HR@10: {val_hr:.4f} | "
              f"Val NDCG@10: {val_ndcg:.4f} | "
              f"Time: {epoch_time:.1f}s{marker}")

        if epochs_no_improve >= args.patience:
            print(f"\nEarly stopping at epoch {epoch} (patience={args.patience})")
            break

    total_time = time.time() - start_time
    print(f"\nTotal training time: {total_time:.1f}s ({total_time/60:.1f}min)")
    print(f"Best validation NDCG@10: {best_ndcg:.4f}")

    print(f"\n{'='*70}")
    print("Final evaluation on test set (full-catalog ranking)...")
    print(f"{'='*70}\n")

    model.load_state_dict(torch.load(best_path, map_location=device))

    for k in args.top_k:
        print(f"\n--- Test metrics @{k} ({args.mode.upper()}) ---")
        test_metrics = evaluate_reasoning(
            model, test_sequences, test_targets, num_items, args.max_length,
            device, mode=args.mode, k=k, batch_size=256, filter_seen=True)
        for name, value in test_metrics.items():
            print(f"  {name}: {value:.4f}")

    print(f"\n{'='*70}")
    print("Done!")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
