---
title: Cuda Graph
type: docs
description: cuda graph 解析
weight: 30
---

### 原理

### Attention

一般情况下（以 FlashAttention / paged attention 为例）：

* ✅ grid/block **通常依赖 q_len**
* ❌ grid/block **通常不依赖 kv_len**
* kv_len 多数作为 **kernel 内部循环长度**

但这不是数学必然，而是工程选择。

看 attention 的核心计算：

\[
softmax(QK^T)V
\]

在 decode 阶段：

```text
Q: [bs, 1, n_heads, head_dim]
K: [bs, kv_len, n_heads, head_dim]
```

对每个 query token，都要算一整行 attention。

大多数实现的并行策略是：

```text
grid.x  ≈  bs * n_heads * q_len
```

也就是说：

> 一个 thread block 负责一个 query block

所以：

* 如果 q_len 从 1 变成 8
* grid 规模会扩大 8 倍

这会改变 launch 参数 → 破坏 CUDA Graph

因此：decode 能 graph 的核心原因之一是：

> q_len 固定为 1

为什么 grid 通常不依赖 kv_len？

因为 kv_len 很大（几千甚至几万），如果：

```cpp
grid.x = kv_len
```

那 grid 会爆炸。

更常见做法是：

* 每个 block 处理一个 query
* 在 block 内部循环 kv_len
* 分 tile 读 K/V

伪代码：

```cpp
for (int i = 0; i < kv_len; i += TILE) {
    load K tile
    accumulate
}
```

这里：

* kv_len 是循环上界
* 不是 grid 维度

因此：

* grid 固定
* 只要 kv_cache 物理大小不变
* graph 就可以 replay

---

prefill：

```text
Q: [bs, seq_len, hidden]
```

此时：

```text
q_len = seq_len
```

如果 prompt 长度不同：

* q_len 变
* grid 变
* launch 参数变
* graph 失效

这才是 prefill 最大的问题。

不是 kv_len。

---

> 必须 kernel 数量一样
> 执行顺序一样
> grid/block 一样
> 所有 tensor 地址一样

我们逐条对照 decode attention：

1️⃣ kernel 数量

decode 每步固定
→ OK

2️⃣ 执行顺序

固定
→ OK

3️⃣ grid/block

只要：

* q_len 固定为 1
* batch 被 padding 到固定版本

→ grid 固定
→ OK

4️⃣ tensor 地址

KV cache 预分配
Q buffer 预分配
workspace 预分配
→ 地址固定
→ OK

因此 decode 可以 graph。

---

> attention 是四维张量 (bs, q_len, kv_len, hidden)
> 那 grid 应该跟 kv_len 有关吧？

但 GPU kernel 的并行维度是：

> 由“怎么分配工作”决定的
> 不是由“数学维度”决定的

工程上可以选择：

* 并行在 q 上
* 或并行在 kv 上
* 或混合

为了 graph 友好和 occupancy 稳定：

→ decode 实现通常不把 kv_len 放进 grid

这是设计选择，不是理论必然。

---

真正决定 graph 能不能复用的不是：

> shape 变不变

而是：

> launch topology 变不变

只要：

* block 数量
* 每个 block 处理多少 query
* kernel 数量

保持一致，

内部循环次数变是无所谓的。

---

* grid/block 通常依赖 q_len
* 很少依赖 kv_len
* decode 固定 q_len=1 → grid 固定
* kv_len 只是循环长度 → graph 可复用

### Prefill

__Prefill 阶段是否需要使用 cuda graph?__

cuda graph 主要省下来的时候

### Piecewise Cuda Graph
