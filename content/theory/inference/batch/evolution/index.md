---
title: 静态动态
type: docs
description: 组 Batch 的方案
weight: 10
---

### 早期的 attn 算子

在早期实现（比如最初的 HF 推理实现）里：

attention kernel 通常假设：

```text
K: [B, n_heads, seq_len, head_dim]
V: [B, n_heads, seq_len, head_dim]
```

并且：

> 每个样本的 KV 在物理上是连续的一段内存

原因很简单：

* kernel 用 pointer arithmetic
* stride 是固定的
* 通过 base_ptr + offset 访问

如果 KV 是碎片化的：

```text
page0
page3
page8
page2
```

传统 kernel 没法高效访问。

### Static batching

特征：

* 一次性收集 N 个请求
* 组成一个 batch
* 一起执行到结束

中途：

* 不加入新请求
* 不移除老请求

典型形态：

```text
batch_size = 8
一直 decode 到 8 个都完成
```

### Dynamic batching

Dynamic Batching 也叫做 Continuous Batching

特征：

* 运行过程中不断有请求加入
* 有请求结束立即移除
* 每一轮 decode 都重新组 batch

例如：

```text
step 0: req1 req2 req3
step 5: req4 加入
step 8: req2 结束
step 9: req5 加入
```

#### Paged Attn

**dynamic batching 在没有 paged 机制时为何难以实现？**

没有 paged attention 时：

每个 request 的 KV 是一整块连续内存：

```text
| req1 0~1023 |
| req2 0~511  |
| req3 0~2047 |
```

当 req2 结束：

```text
| req1 | 空洞 | req3 |
```

这会导致两个问题：

🔴 1️⃣ 内存碎片

你不能把 req4 放进“空洞”里
因为 KV 必须连续。

🔴 2️⃣ 批处理困难

如果 dynamic batching：

你想把 req1 + req3 + req4 拼一起算

但：

* 它们的 KV 在不同连续块
* kernel 要为每个样本单独传 pointer

传统实现里：

通常会变成：

```text
for each request:
    launch attention kernel
```

而不是：

```text
一个大 kernel 处理整个 batch
```

吞吐会大幅下降。

---

**paged attention 真正改变了什么？**

它改变的不是：

> 是否支持不同 kv_len

而是：

> 是否允许 KV 在物理上离散

Paged attention 允许：

```text
req1 = page 7, 2, 19
req2 = page 1, 3
req3 = page 8, 9, 10, 11
```

然后通过 page table：

```text
logical token index -> physical page
```

kernel 内部遍历 page table。

这意味着：

✔ 内存可以复用
✔ 不再需要连续大块
✔ 请求可以随时加入
✔ 请求可以随时删除

于是：

> dynamic batching 变得可行

{{< details "paged attention 机制和 dynamic batching 的关联" >}}
* Static batching：batch 固定
* Dynamic batching：batch 动态变化
* Paged attention：让 KV 可以离散存储，从而支持高效 dynamic batching

它们是三层不同的概念：

```text
调度层（static/dynamic batching）
        ↓
内存层（paged / contiguous KV）
        ↓
kernel 层（如何遍历 KV）
```

{{< /details >}}
