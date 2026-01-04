---
title: TP4 流程优化
type: docs
weight: 10
---

### 优化前

优化前的 TP/DP 规划存在以下问题：

- **重复计算**：q_a_and_kv_a_proj 与 layernorm 在已经切分为 TP4 的情况下，实际上可以按照单机 DP2 的维度进行计算。目前仍按 DP8 的方式执行，产生了不必要的冗余计算。

- **通信效率不足**：当前仅在 attention 阶段进行了 TP 切分，后续使用 AllReduce 操作。由于 TP4 相比 TP1 减少了 4 倍的 DP 数量，而 EP 的 rank 数量保持不变，每个 EP 内部请求维度的 batch size 仍应切分。

  这里可以考虑用 **ReduceScatter** 替代当前的 AllReduce 加序列切分的操作，以提升通信效率。

{{< svg "/images/deepseek_tp4_before.drawio.svg" "120%" >}}

### 优化版本1

{{< svg "/images/deepseek_tp4_ver1.drawio.svg" "120%" >}}

---

- **优化点 1：对 q_a_and_kv_a_proj 按照 TP 维度也进行切分，不在每个 rank 内重复进行计算**

假设 prefill 总 token 数量为 \(T\)，每张卡的计算量减少为原来的 1/4，减少的计算量（单卡、单层）为
\[\frac{3}{4} \times T \times 7168 \times 2112 \times 2\]

假设单卡算力为 \(C\) TFlops，模型总共 \(L\) 层，则切分带来的时延收益为
\[\frac{3}{4} \times T \times 7168 \times 2112 \times 2 / (C \times \text{MFU}) / 1e12 \times L \text{ (s)}\]

之后有一个额外的 AllGather 操作，引入的通信量（单卡、单层）为

\[\frac{3}{4} \times T \times 7168\]

假设单卡 nvlink 带宽为 \(W\) GBps，模型总共 \(L\) 层，则带来的额外通信时间延为：

\[\frac{3}{4} \times T \times 7168 / (W \times \text{MBU}) / 1e9 \times L \text{ (s)}\]

对于长序列，这里设定 \( T = 16k \)，模型共有 61 层 \( L = 61 \)。在 H100 上，其算力约为 2000 TFlops，NVLink 带宽为 900 GB/s，假设 \(\text{MFU} = \text{MBU} = 50\%\)。

在 ttft 中，减少的 q 与 kv 投影计算时间约为 \( 22.7 \text{ms} \)，而增加的 AllGather 通信时间约为 \( 11.9 \text{ms} \)，切分是有实际收益的。

- **优化点 2：用 ReduceScatter 替代 AllReduce + 序列切分，减少通信量**

AllReduce 的通信量为 \(T \times 7186 \times 2 \times \text{tp\_size - 1}\)

ReduceScatter 的通信量为 \(T \times 7168 \times \text{(tp\_size - 1)}\)

减少的通信量为 \(T \times 7168 \times \text{(tp\_size -1 1)}\)

按照前文假设，在 16k 输入 TP4 的情况下，预估减少的时间为 \(4 \times 1024 \times 7168 \times 3 / 450 / 1e9 * 1000 = 19.5 \text{ ms}\)

### 优化版本2

{{< svg "/images/deepseek_tp4_ver2.drawio.svg" "120%" >}}

该优化和前面其实差不多，不过用用 All2All 替代了 ReduceScatter，该版本
