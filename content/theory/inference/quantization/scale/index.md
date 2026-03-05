---
title: 缩放因子
type: docs
description: scale in quantization
weight: 10
---

### group size

一句话总结核心：

> group size 是为了解决 scale 存储成本和 kernel 访存的问题。

存储 scale 的时候不能为为一个被压缩过的数据单元都存储一个，不然就失去意义了：

因为 quantization 本质上就是为了少存储一些 weight，减少显存占用，每个都存一个独立的 factor，那么显存占用反而增大了。

所以是 tensor 中的连续多个数据共用一个 factor，连续的数量就是 group size。

### quant process

__分组 g_idx 如何确定？__

流程一般是：

1. 用 calibration dataset 跑一遍 forward
2. 统计 activation 分布
3. 根据 activation 分布，推导权重 scale（或直接统计权重最大值）
4. 固定下来

所有 expert group_size、分组维度、排列方式都是一样的。

### inference

方案一：**动态索引（最慢）**  
在计算每一个乘法时，都需要查询一张表（`g_idx`）来确认“第 13 列属于哪个组？它的 Scale 在哪？”。这导致 GPU 的 Tensor Core 无法高效并行。Tensor Core 适合大块数据的连续吞吐，而频繁查表会严重破坏并行性，使性能降至 FP16 的几分之一。

方案二：**重排输入（Shuffle）**  
在矩阵乘法开始前，将输入 \(x\) 按照 act-order 的顺序重排，使其对齐乱序后的权重。  
对于 **MoE 架构** 而言：  
- 专家（Expert）之间切换极快。  
- 若在每个专家的 `w13` 和 `w2` 计算前都进行一次大规模内存重排，所带来的延迟（Latency）往往会超过量化所节省的时间。
