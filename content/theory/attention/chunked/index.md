---
title: chunked prefill
type: docs
weight: 100
---

### 背景

在大模型推理中，**prefill 阶段** 是计算密集型，尤其是在真实业务中，prompt 往往是 **1K～4K tokens** 甚至更长。一个直觉上的做法是：

> 既然是 prefill，就一次性把所有 prompt token 喂给模型。

但实践中会遇到一个问题：**一次性塞更多 token，并不会无限提升吞吐率**。

#### Prefill 阶段

**关键观察 1**：Prefill 吞吐存在“边际收益递减”

在固定模型和 GPU 的情况下，prefill 阶段的吞吐率会在某个 token 数附近达到上限。比如下图是 FlashAttention 3 在 H200 上的性能测试结果（这里采用定长序列测试，行是序列长度，列是 batch size）：

{{< img "/images/fa3_why_chunk.png" "80%" >}}

可以观察到，在某个序列长度下，随着 batch size 的提升，attention kernel 的 FLOPS 不进一步提升，反而会逐步下降。所以这个时候进一步提升输入的 batch size 是没有意义的，反而会增加请求的 TTFT。

而且一个很重要的现象是：

> **模型越大（hidden size 越大），达到算力饱和所需的 token 数反而越小**

也就是说，只要 chunk 大小选得合理，就已经可以充分利用 GPU 计算能力。

---

#### Decode 阶段

decode 阶段（逐 token 生成）也有类似现象：

* 随着 batch size / decode token 总数增加，吞吐率持续上升
* 当总 token 数接近 **512** 左右时，也会进入 **compute-bound**
* 再继续加 batch，对性能帮助不大

这意味着：
**prefill 和 decode 本质上都不需要“越大越好”的一次性计算单元。**

---

### 目标

综合来看，chunked prefill 的目标是解决两个现实问题：

1️⃣ **超长 prompt 的计算调度问题**

真实业务中：

* 一个请求可能带 **2K / 4K token prompt**
* 如果一次性 prefill：

  * 显存占用大
  * kernel 执行时间长
  * 调度不灵活，容易阻塞其他请求

而 **chunked prefill** 允许我们把一个长 prompt 拆成多个**计算友好的小单元**。

---

2️⃣ **提高系统层面的并发与公平性**

在 serving 场景中：

* 很多请求是「长 prompt + 短输出」
* 如果长 prompt 一次性 prefill：

  * GPU 会被单个请求“霸占”较长时间
* 拆成 chunk 后：

  * 每个 chunk 都是一个可调度的计算单元
  * 可以和其他请求交错执行
  * 系统整体 latency 和 tail latency 更可控

---

### 长序列分治

**Chunked Prefill 的核心思想**就是“化整为零”：将长的输入序列（`K, V`）切分成若干个较小的块（Chunks），然后让查询（`Q`）逐个与这些块进行计算，最后像“拼图”一样，将每个块的结果合并成完整的注意力输出。这就像你无法一口吃完一个大蛋糕，但可以把它切成小块慢慢吃。

将长序列分成多个 chunk（块），逐块处理：

```
输入序列: [chunk1, chunk2, chunk3, ...]
处理流程:
1. 处理 chunk1 → 得到部分结果
2. 处理 chunk2 → 与 chunk1 的结果合并
3. 处理 chunk3 → 与前两个 chunk 的结果合并
...
```

