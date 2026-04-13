---
title: GQA
type: docs
weight: 17
---

### 连续分组

GQA 按 **“连续块（contiguous block）”分组，而不是“交错（strided）分组”**

假设 q heads = 8，kv heads = 2，那么：

```
Q0,1,2,3 → KV0
Q4,5,6,7 → KV1
```

这种 连续分组设计，本质上是为 TP / 分布式推理服务的

```
KV0 ← Q0~3
KV1 ← Q4~7
```

这样可以直接 GPU0 计算 KV0，GPU1 计算 KV1 的部分。

### QKVParallelLinear

**核心逻辑**：在 weight load 的过程中就处理好每个 rank 需要的那一部分 qkv_proj 权重。

> QKVParallelLinear 在 weight_loader 中，将 fused QKV weight 按 [Q | K | V] 分段，对每一段分别根据 TP rank 和 KV replication 规则计算 shard_id，通过 narrow 截取对应 slice，并写入本 rank 的参数，从而构造出只包含本 rank 所需 Q/K/V 的局部 weight。

假设 q_heads=8，kv_heads=2，tp_size=2，那么：

```python
def weight_loader(...):
    # shard 逻辑：定义本 rank 应该拿多少 Q/K/V
    # 例如：
    # rank0:
    #     Q: 4 heads
    #     K: 1 head
    #     V: 1 head
    if loaded_shard_id == "q":
        shard_offset = 0
        shard_size = self.num_heads * self.head_size
    elif loaded_shard_id == "k":
        shard_offset = self.num_heads * self.head_size
        shard_size = self.num_kv_heads * self.head_size
    elif loaded_shard_id == "v":
        shard_offset = (self.num_heads + self.num_kv_heads) * self.head_size
        shard_size = self.num_kv_heads * self.v_head_size

    # 决定“拿哪一块”
    # 例如：rank 0 [ Q0 Q1 Q2 Q3 | K0 | V0 ]
    #       rank 1 [ Q4 Q5 Q6 Q7 | K1 | V1 ]
    if loaded_shard_id == "q":
        shard_id = self.tp_rank
    else:
        shard_id = self.tp_rank // self.num_kv_head_replicas

    # 切分 tensor
    start_idx = shard_id * shard_size
    loaded_weight = loaded_weight.narrow(
        output_dim, start_idx, shard_size
    )
    # 写入本 rank 的参数
    param_data.copy_(loaded_weight)
```

### Torch GQA

记忆几个点，方便面试速写：

- 用 s 代表 q_len
- 用 t 代表 kv_len
- Q @ K 的 einsum 写作 `b s h d, b t h d -> b h s t`
- attn_scores @ V 的 einsum 写作 `b h s t, b t h d -> b s h d`
- attn_scores 计算记得 `/ (self.d_qk ** 0.5)`


```python
class GQA(nn.Module):
    def __init__(self):
        supert().__init__()
        self.q_size = h * d_qk
        self.k_size = h_kv * d_qk
        self.v_size = h_kv * d_v

    def forward(self, x):
        # x: [B, S, D]
        B, S, D = x.shape

        # ---- 1. qkv proj ----
        qkv, _ = self.qkv_proj(x)
        q, k, v = qkv.split(
            [self.q_size, self.k_size, self.v_size], dim=-1
        )

        # ---- 2. reshape ----
        q = q.reshape([B, S, h, d_qk])
        k = k.reshape([B, S, h_kv, d_qk])
        v = v.reshape([B, S, h_kv, d_v])

        # ---- 3. expand ----
        repeat_factor = h // h_kv
        k = k.repeat_interleave(repeat_factor, dim=2) # [b,s,h,d]
        v = v.repeat_interleave(repeat_factor, dim=2) # [b,s,h,d]

        # ---- 4. attention ----
        attn_scores = torch.einsum(
            "b t h d, b s h d -> b h t s",
            q, k
        ) / (d_qk ** 0.5)

        if mask is not None:
            out =  torch.einsum(
                "b h t s, b s h d -> b t h d",
                attn_scores, v
            )

        # ---- 4. o proj ----
        out = out.reshape([B, S, h_kv * d_v])
        out = self.o_proj(out)
        return out
```
