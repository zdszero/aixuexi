---
title: DSA
type: docs
weight: 30
---

### Indexer

Indexer 在结构上类似一个极简的 attention 前半段：它使用少量 head 的低维 \(QK^T\) 计算 query–key 的相关性分数，但不进行 softmax，也不计算 value 聚合。  
其输出是一个索引分数，用于评估每个 query token 与历史 token 的相关性，从而筛选出需要进入真实 attention 计算的候选 token。  

其中，不同 head 的相关性分数通过由 query token 自适应生成的 gating 权重进行加权融合。

\[
I_{t,s} = \sum_{j=1}^{H^l} w_{t,j}^l \cdot ReLU(q_{t,j}^l \cdot k_s^l)
\]

其中：

- \(H^l\)：索引器头数（数量少，保证轻量化）；
- \(w_{t,j}^l\)：由当前查询 token \(h_t\) 生成的权重；
- \(q_{t,j}^l / k_s^l\)：分别由查询 token、前文 token 生成的查询向量与键向量；
- \(ReLU\)：激活函数。

### Shape

- \(W Q^{T}\): `[bs, q_len, kv_len, n_heads, 1]`
- weight proj: `[bs, n_heads]`
- index score: `[bs, q_len, kv_len]`

{{< svg "/images/v32_indexer_workflow.drawio.svg" >}}

### 计算量分析

#### decode

attention 计算量在 decode 阶段线性增长，遵循如下公式：

\[4BTSNH\]

其中：
- \(B\): batch size
- \(T\): q_len
- \(S\): kv_len
- \(N\): number of q heads
- \(H\): head dim

实现 DSA 之后，当 kv_len 长度超过 2048 的时候，稀疏注意力部分的计算量就不再增长了，但是 indexer 阶段的注意力还是线性增长的，不过这个斜率比较小，因为 \(N=64, H=64\)，这个参数还是比较小的。
