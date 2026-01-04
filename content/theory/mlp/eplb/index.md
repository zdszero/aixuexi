---
title: EPLB
type: docs
weight: 40
---

在 Mixture-of-Experts（MoE）架构中，不同专家接收的 token 数量存在显著差异，这直接导致了专家间的计算负载不均衡。当这些专家被分配到不同 GPU 上时，负载差异会进一步引发设备间的计算不均衡问题。

这种不均衡会降低 MoE 的整体效率。MoE 系统类似于一个木桶，其容量取决于最短的那块木板。均衡有两个方面：

- 不同 rank 之间的均衡
- 不同 expert 之间的均衡

以下是一个简单示例，用于说明这一问题。假设模型中有四个专家，分别部署在两张 GPU 上。其中，GPU0 上的专家负载较高，需要处理 \(75\%\) 的输入数据；而 GPU1 上的专家负载较低，仅需处理 \(25\%\) 的数据。

### Workflow

- 逻辑专家（logical expert）：模型原本的所有专家
- 冗余专家（redundant expert）：为高负载专家额外设置的冗余部署
- 物理专家（physical expert）：逻辑 + 冗余

```
在所有 node 中均衡放置所有逻辑专家
    ↓
决定哪些专家应该被 replicate
    ↓
在所有 node 中均衡放置所有物理专家
```

### Balanced Pack

这里本质上是一个装箱问题：

* `num_packs = m`
  * 要把 n 个 item 分到 m 个 pack（bin）里

**硬约束：**

1. 每个 pack **恰好装 `n / m` 个 item**
2. 每个 item 只能进一个 pack

**优化目标：**

* 让每个 pack 的**总 weight 尽量均衡**
* 理想情况：每个 pack 的 weight 和接近 `sum(weight) / m`

__算法思路：__

> **Largest-First Greedy + Least-Loaded Bin**

也就是：

1. **先处理“最重的 item”**
   * 重的东西最容易造成不均衡
2. **每次放到当前最轻的 bin 里**
   * 平衡局部 load
3. **但每个 bin 有容量上限（items 数）**
   * 防止一个 bin 被塞爆


**核心贪心循环（算法本体）**

```python
for group in indices[i]:
    pack = min(
        (i for i in range(num_packs) if pack_items[i] < groups_per_pack),
        key=pack_weights.__getitem__
    )
```

这句话是整个算法的精髓：**在“还有空位”的 pack 里，选当前 weight 最小的那个**

即：

* 过滤掉已经装满的 pack
* 在剩下的 pack 中，选 `pack_weights` 最小的

这正是：**Least-Loaded Feasible Bin**

### Replicate Expert

核心逻辑：


```python
for i in range(num_log, num_phy):
    redundant_indices = (weight / logcnt).max(dim=-1).indices
```


> **在当前副本数 logcnt 下，谁的“单副本负载”最大，就给谁再加一个副本**

这是一个经典的 **water-filling / load-splitting** 削峰策略，每次选取最大的峰值对半削弱。

举个例子：


假设 node 内有 4 个 logical experts：

\[
\begin{array}{|c|c|}
\hline
\text{expert} & \text{load} \\
\hline
E0 & 100 \\
\hline
E1 & 80 \\
\hline
E2 & 30 \\
\hline
E3 & 20 \\
\hline
\end{array}
\]

需要加 2 个冗余（num_phy = 6）。


```text
Round 1:
effective load = load / replicas = [100, 80, 30, 20]
→ 给 E0

Round 2
replicas = [2,1,1,1]
effective load = [50, 80, 30, 20]
→ 给 E1

Finally:
replicas = [2,2,1,1]
```
