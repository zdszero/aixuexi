---
title: Ring
type: docs
description: Ring Communication
weight: 10
---

### ring attention

ring attention 的 ring 通信机制是如何实现的？

一般是 **P2P + 手写 ring pipeline**

底层用

- ncclSend / ncclRecv
- 或 CUDA P2P（NVLink / PCIe）

在 pytorch 中对应：

```
torch.distributed.isend
torch.distributed.irecv
```

这个模式其实更准确叫：

> **Pipelined P2P Ring Communication**

而不是 NCCL 那种“黑盒 ring”

### 通信原语

#### all gather ring

AllGather 可以基于 ring 或者 tree 的通信方式实现：

{{< svg "/images/allgather_ring_mechanism.svg" >}}

#### reduce scatter ring

{{< svg "/images/reduce_scatter_tp4.drawio.svg" "110%" >}}

以 TP4 为例，在 3 轮 ring 通信过后，假设完整数据分为 4 个 Shard A, B, C, D：

- GPU 3 持有 A 的 reduce 结果
- GPU 2 持有 B 的 reduce 结果
- GPU 1 持有 C 的 reduce 结果
- GPU 0 持有 D 的 reduce 结果

#### all2all ring

{{< svg "/images/all2all_tp4.drawio.svg" "90%" >}}

#### latency estimate

{{< anchored id="AllGather-ReduceScatter-All2All-估算" block="true" >}}AllGather, ReduceScatter, All2All 估算{{< /anchored >}}

**AllGather, ReduceScatter, All2All 的 ring 机制都类似**

\[\boxed{T_{\text{AllGather}} = T_{\text{All2All}} = T_{\text{ReduceScatter}} = (P-1) \times \frac{\text{local\_bytes}}{\text{BW\_unidirectional}}}\]

{{< anchored id="AllReduce 估算" block="true" >}}AllReduce 估算{{< /anchored >}}

**系数 2 来自 AllReduce**，它分两个阶段：

\[\boxed{T_{\text{AllReduce}} = 2(P-1) \times \frac{\text{local\_bytes}}{\text{BW\_unidirectional}}}\]

- Phase 1（ReduceScatter）：\((P-1)\) 步，每步传 \(M/P\)
- Phase 2（AllGather）：\((P-1)\) 步，每步传 \(M/P\)
- 合计乘 2

公式中的 **2 代表 AllReduce 的两个阶段**，不是 send/recv 双向。

{{< region note >}}
**一个细节：BW 用单向还是双向？**

ring 上每个 GPU 同时在 send（到右邻居）和 recv（从左邻居），用的是两条**独立**链路，所以 `BW` 填的是**单向链路带宽**（unidirectional），不需要除以 2。NVLink 的 `600 GB/s` 是双向总带宽，单向是 `300 GB/s`，填公式时要用 `300`。
{{< /region >}}
