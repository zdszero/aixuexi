---
title: 算子融合
type: docs
weight: 30
---

### norm & splitkv

### 选专家算子

### MoE 数据准备

Input Params:

- `recv_x_fp8`: `[B, T, 7168]`, original consecutive token embeddings without padding
- `recv_x_fp8_scale`: `[B, T, 1]`, fp8 scale
- `recv_topk_idx`: `[B, T, top_k]`, element is expert id
- `num_recv_tokens_per_expert_list`: `[E, 1]`

Prepare Data for deep_gemm:

- `num_recv_tokens_per_expert_list_128`: 每个专家的数量的 128 整数倍
- `num_recv_tokens_per_expert_list_128_start`: 每个专家在 recv_x_fp8_all 中的起始下标
- `recv_x_fp8_all`: `[B, T, 7168]`, recv_x_fp8 加上 128 padding 后的数据
- `recv_x_fp8_scale_all`: `[B, T, 1]`, recv_x_fp8_scale_all 加上 128 padding 后的数据

```
num_recv_tokens_per_expert_list:
    [188, 64, 309, 33, ..., 12]
                    ↓
num_recv_tokens_per_expert_list_128:
    [256, 128, 384, ..., 128]
num_recv_tokens_per_expert_list_128_start:
    [0, 256, 256+128, 256+128+384, ..., prev sum]

recv_x_fp8:
    [t1, t2, ..., t100]
            ↓
recv_x_fp8_all:
    [expert1_tokens, expert2_tokens, ...]

m_idx: the current token to filled in each expert group
token_counts: number of experts each tokens choose
token_to_m: map [token_id, expert_idx] to m_idxx
```

```python
for token_idx in range(num_tokens):
    for j in range(topk):
        if (recv_topk_idx[token_idx, j] != -1):
            expert = recv_topk_idx[token_idx, j].item()
            m_idx = num_recv_tokens_per_expert_list_128_start[expert]
            recv_x_all_fp8[m_idx, :] = recv_x_fp8[token_idx, :]
            recv_x_all_scale[m_idx, :] = recv_x_scale[token_idx, :]
            token_to_m[token_idx, token_counts[token_idx]] = m_idx
            token_counts[token_idx] += 1
            token_weights[m_idx] = recv_topk_weights[token_idx][j]
            num_recv_tokens_per_expert_list_128_start[expert] += 1
```
