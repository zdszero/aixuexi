---
title: Topology
type: docs
description: Topology
weight: 10
---

### NVLink mesh

{{< svg "/images/nvlink_fullmesh_8gpu.svg" >}}

#### bandwidth

需要注意区分单向（unidirectional）和双向带宽（bidirectional）：

| 规格 | 双向总带宽 | 单向带宽（填公式用这个） |
|---|---|---|
| NVLink 3（A100） | 600 GB/s | 300 GB/s |
| NVLink 4（H100） | 900 GB/s | 450 GB/s |
| NVLink 5（B200） | 1800 GB/s | 900 GB/s |

NCCL 文档和 `nccl-tests` 的 busbw 换算也是基于单向带宽——`nccl-tests` 输出的 `busbw` 列实际上是把测量到的 algobw 乘以 `(P-1)/P`（AllGather 系数），对应的就是单向链路利用率。

### Fat Tree
