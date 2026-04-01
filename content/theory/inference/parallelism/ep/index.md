---
title: EP
type: docs
description: Expert Parallelism 并行详解
weight: 10
---

### DeepSeek 两阶段专家

#### 高层概述

256 个 routed experts 被分成 **8 组，每组 32 个**。选择过程分两步：先从 8 组中选 4 组，再从这 4 组（128 个候选）中选 8 个专家。

#### 为什么要分组

如果直接从 256 中选 8 个，很可能所有 8 个都集中在某几个"热门"组里，导致：
1. **专家利用不均衡**——部分组过载，部分组闲置
2. **多样性不足**——不同组可能编码了不同的知识模式，只从少数组选会丢失多样性

分组限制（从 8 组中选 4 组）保证了每个 token 的专家**至少分散在 4 个组**中，促进负载均衡和知识覆盖。

#### 完整流程

```
hidden_states: (num_tokens, 7168)
```

##### 1: Gate 投影

```python
logits = F.linear(hidden_states, W_gate)   # W_gate: (256, 7168)
## logits: (num_tokens, 256)
```

每个 token 对每个专家产生一个标量 logit。

##### 2: Sigmoid 打分

```python
scores = sigmoid(logits)   # (num_tokens, 256)，每个值在 [0, 1]
```

V3 用 sigmoid 而非 softmax——每个专家独立打分，不互斥。

##### 3: 加 correction bias（用于选择）

```python
scores_for_choice = scores + correction_bias   # correction_bias: (256,) 可学习参数
```

这是 V3 的 **noaux_tc** 策略：不用辅助负载均衡 loss，而是用一个可学习的 per-expert bias 来隐式调节路由均衡。**bias 只影响选择，不影响最终权重**。

##### 4: 计算组分数

```python
## 256 个专家 reshape 为 8 组 × 32 个
group_view = scores_for_choice.view(n, 8, 32)

## 每组取 top-2 分数求和，作为该组的得分
group_scores = group_view.topk(2, dim=-1)[0].sum(dim=-1)   # (n, 8)
```

用 top-2 之和而非 max，能更稳定地反映该组对当前 token 的整体相关性。

##### 5: 选 4 个组

```python
group_idx = topk(group_scores, k=4)   # (n, 4)
```

从 8 组中选出得分最高的 **4 组**，剩下 4 组的所有专家被 mask 为 `-inf`。

##### 6: 从候选中选 8 个专家

```python
## 被淘汰组的专家设为 -inf
tmp_scores = scores_for_choice.masked_fill(非选中组, -inf)

## 从剩余 ~128 个候选中选 top-8
topk_ids = topk(tmp_scores, k=8)      # (n, 8) 专家编号
topk_weights = scores.gather(topk_ids) # 用原始 sigmoid 分数（不含 bias）
```

关键：**选择用的是加了 bias 的分数，但最终路由权重用的是原始 sigmoid 分数**，防止 bias 扭曲专家输出的加权。

##### 7: 归一化 + scaling

```python
topk_weights = topk_weights / topk_weights.sum()  # 归一化到和为 1
topk_weights *= 2.5                                 # routed_scaling_factor
```

`routed_scaling_factor=2.5` 是 V3 的超参，放大 routed experts 的贡献（相对于 shared expert）。

##### 8: 加权求和

```python
output = 2.5 × Σ(weight_i × expert_i(x)) + shared_expert(x)
```

#### 流程图

```
hidden (7168)
    │
    ▼
Gate: linear → logits (256)
    │
    ▼
sigmoid → scores (256)         ← 原始分数，用于最终权重
    │
    ├── + bias → scores_for_choice (256)  ← 加偏置，仅用于选择
    │       │
    │       ▼
    │   reshape (8 组 × 32 专家)
    │       │
    │       ▼
    │   每组 top-2 求和 → group_scores (8)
    │       │
    │       ▼
    │   选 top-4 组 → 淘汰 4 组 (mask=-inf)
    │       │
    │       ▼
    │   从剩余候选选 top-8 专家 → topk_ids (8)
    │
    ▼
gather 原始 scores → topk_weights (8)
    │
    ▼
normalize + × 2.5 → 最终权重
    │
    ▼
Σ weight_i × FFN_expert_i(x) + shared_expert(x)
```
