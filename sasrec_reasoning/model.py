import torch
import torch.nn as nn


class PointWiseFeedForward(nn.Module):
    def __init__(self, hidden_units, dropout_rate):
        super().__init__()
        self.conv1 = nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.conv1(inputs.transpose(-1, -2))
        outputs = self.relu(self.dropout1(outputs))
        outputs = self.conv2(outputs)
        outputs = self.dropout2(outputs)
        outputs = outputs.transpose(-1, -2)
        outputs += inputs
        return outputs


class SASRecReasoning(nn.Module):
    def __init__(self, item_num, maxlen=200, hidden_units=256, num_blocks=2,
                 num_heads=1, dropout_rate=0.1, initializer_range=0.02,
                 reason_steps=2):
        super().__init__()

        self.item_num = item_num
        self.maxlen = maxlen
        self.hidden_units = hidden_units
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.initializer_range = initializer_range
        self.reason_steps = reason_steps

        self.item_emb = nn.Embedding(item_num + 1, hidden_units, padding_idx=0)
        self.pos_emb = nn.Embedding(maxlen, hidden_units)
        self.emb_dropout = nn.Dropout(dropout_rate)

        self.attention_layernorms = nn.ModuleList()
        self.attention_layers = nn.ModuleList()
        self.forward_layernorms = nn.ModuleList()
        self.forward_layers = nn.ModuleList()
        self.last_layernorm = nn.LayerNorm(hidden_units, eps=1e-8)

        for _ in range(num_blocks):
            self.attention_layernorms.append(nn.LayerNorm(hidden_units, eps=1e-8))
            self.attention_layers.append(
                nn.MultiheadAttention(hidden_units, num_heads, dropout_rate))
            self.forward_layernorms.append(nn.LayerNorm(hidden_units, eps=1e-8))
            self.forward_layers.append(PointWiseFeedForward(hidden_units, dropout_rate))

        if reason_steps > 0:
            self.reason_pos_emb = nn.Embedding(reason_steps, hidden_units)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def _forward_block(self, seqs, attn_mask, block_idx, kv_cache=None):
        seqs_t = seqs.transpose(0, 1)
        Q = self.attention_layernorms[block_idx](seqs_t)

        if kv_cache is not None:
            cached_k, cached_v = kv_cache
            K_full = torch.cat([cached_k, seqs_t], dim=0)
            V_full = torch.cat([cached_v, seqs_t], dim=0)
        else:
            K_full = seqs_t
            V_full = seqs_t

        mha_out, _ = self.attention_layers[block_idx](
            Q, K_full, V_full, attn_mask=attn_mask)
        seqs_t = Q + mha_out
        seqs = seqs_t.transpose(0, 1)

        seqs = self.forward_layernorms[block_idx](seqs)
        seqs = self.forward_layers[block_idx](seqs)

        new_kv = (K_full, V_full)
        return seqs, new_kv

    def encode_base(self, input_ids):
        seqs = self.item_emb(input_ids)
        seqs *= self.hidden_units ** 0.5

        positions = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)
        seqs += self.pos_emb(positions)
        seqs = self.emb_dropout(seqs)

        timeline_mask = (input_ids == 0)
        seqs = seqs * (~timeline_mask).unsqueeze(-1).float()

        tl = seqs.shape[1]
        attn_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=seqs.device))

        kv_caches = []
        for i in range(self.num_blocks):
            seqs, new_kv = self._forward_block(seqs, attn_mask, i)
            seqs = seqs * (~timeline_mask).unsqueeze(-1).float()
            kv_caches.append(new_kv)

        seqs = self.last_layernorm(seqs)
        return seqs, kv_caches

    def reasoning_step(self, last_hidden, step_idx, kv_caches):
        rpe = self.reason_pos_emb(
            torch.tensor([step_idx], device=last_hidden.device))
        token = last_hidden + rpe.unsqueeze(0)

        seqs = token
        new_kv_caches = []

        for i in range(self.num_blocks):
            cached_k, cached_v = kv_caches[i]
            full_len = cached_k.shape[0] + 1
            attn_mask = torch.zeros((1, full_len), dtype=torch.bool,
                                    device=seqs.device)

            seqs, new_kv = self._forward_block(seqs, attn_mask, i, kv_cache=kv_caches[i])
            new_kv_caches.append(new_kv)

        seqs = self.last_layernorm(seqs)
        return seqs, new_kv_caches

    def forward(self, input_ids, reason_steps=None):
        K = reason_steps if reason_steps is not None else self.reason_steps

        hidden, kv_caches = self.encode_base(input_ids)

        lengths = (input_ids != 0).sum(dim=1) - 1
        rows = torch.arange(input_ids.shape[0], device=input_ids.device)
        r_0 = hidden[rows, lengths].unsqueeze(1)

        reasoning_outputs = [r_0]

        if K > 0:
            current = r_0
            for step in range(K):
                current, kv_caches = self.reasoning_step(current, step, kv_caches)
                reasoning_outputs.append(current)

        return hidden, torch.cat(reasoning_outputs, dim=1)
