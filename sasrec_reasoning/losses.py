import torch
import torch.nn.functional as F


def compute_erl_sampled_loss(reasoning_outputs, labels, negatives, item_emb,
                             kl_weight=0.05, kl_temperature=1.0):
    B, num_steps, D = reasoning_outputs.shape

    seq_emb = reasoning_outputs.mean(dim=1)

    pos_emb = item_emb(labels)
    pos_logits = (seq_emb * pos_emb).sum(dim=-1, keepdim=True)

    neg_emb = item_emb(negatives)
    neg_logits = torch.bmm(neg_emb, seq_emb.unsqueeze(-1)).squeeze(-1)

    logits = torch.cat([pos_logits, neg_logits], dim=-1)
    logits = logits.clamp(-50.0, 50.0)
    targets = torch.zeros(B, dtype=torch.long, device=logits.device)
    ce_loss = F.cross_entropy(logits, targets)

    kl_loss = torch.tensor(0.0, device=reasoning_outputs.device)
    if num_steps > 1 and kl_weight > 0:
        all_items = torch.cat([labels.unsqueeze(1), negatives], dim=1)
        all_emb = item_emb(all_items)

        step_logits = torch.bmm(
            reasoning_outputs,
            all_emb.transpose(1, 2)
        ) / kl_temperature
        step_logits = step_logits.clamp(-50.0, 50.0)

        step_log_probs = F.log_softmax(step_logits, dim=-1)

        kl_sum = torch.tensor(0.0, device=reasoning_outputs.device)
        count = 0
        for i in range(num_steps):
            for j in range(num_steps):
                if i != j:
                    kl = F.kl_div(
                        step_log_probs[:, j, :],
                        step_log_probs[:, i, :],
                        reduction='batchmean',
                        log_target=True,
                    )
                    kl_sum += kl
                    count += 1
        kl_loss = kl_sum / count if count > 0 else kl_sum

    total_loss = ce_loss - kl_weight * kl_loss

    return total_loss, {
        'ce_loss': ce_loss.item(),
        'kl_loss': kl_loss.item(),
        'total_loss': total_loss.item(),
    }


def compute_prl_sampled_loss(reasoning_outputs, labels, negatives, item_emb,
                             temp_scale=5.0, pl_weight=1.0):
    B, num_steps, D = reasoning_outputs.shape

    pos_emb = item_emb(labels)
    neg_emb = item_emb(negatives)
    all_items_emb = torch.cat([pos_emb.unsqueeze(1), neg_emb], dim=1)

    final_emb = reasoning_outputs[:, -1, :]
    final_logits = torch.bmm(
        all_items_emb,
        final_emb.unsqueeze(-1)
    ).squeeze(-1)
    final_logits = final_logits.clamp(-50.0, 50.0)
    targets = torch.zeros(B, dtype=torch.long, device=final_logits.device)
    main_loss = F.cross_entropy(final_logits, targets)

    pl_loss = torch.tensor(0.0, device=reasoning_outputs.device)
    if num_steps > 1 and pl_weight > 0:
        intermediate = reasoning_outputs[:, :-1, :]
        K = num_steps - 1

        temp_scales = temp_scale ** torch.arange(K, 0, -1, dtype=torch.float,
                                                  device=reasoning_outputs.device)

        inter_logits = torch.bmm(
            intermediate,
            all_items_emb.transpose(1, 2)
        )
        inter_logits = inter_logits / temp_scales.view(1, K, 1)
        inter_logits = inter_logits.clamp(-50.0, 50.0)

        inter_logits_flat = inter_logits.reshape(B * K, -1)
        inter_targets = targets.unsqueeze(1).expand(B, K).reshape(B * K)
        pl_loss = F.cross_entropy(inter_logits_flat, inter_targets)

    total_loss = main_loss + pl_weight * pl_loss

    return total_loss, {
        'main_loss': main_loss.item(),
        'pl_loss': pl_loss.item(),
        'total_loss': total_loss.item(),
    }
