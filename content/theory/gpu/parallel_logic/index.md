---
title: 并行逻辑
type: docs
description: GPU 中的不同维度的并行抽象
weight: 10
---

在 GPU（以 NVIDIA CUDA 架构为例）中，**grid、block、warp** 是三个不同层级的并行组织单位，它们的关系可以理解为：

> **grid > block > warp > thread**

下面我给你从“抽象层级 + 硬件层级 + 性能影响”三个角度讲清楚。

---

### Grid

Grid 指的是 **一次 kernel 启动所包含的所有线程块（block）的集合。**

```cpp
kernel<<<gridDim, blockDim>>>(...)
```

* `gridDim` = block 的数量
* 一个 grid 只对应一次 kernel 调用

__特点__

* block 之间 **默认不能同步**
* block 之间 **不能共享 shared memory**
* 可以在不同 SM 上并行执行

你可以把 grid 理解为：

> “一个任务被拆成很多 block，全部一起提交给 GPU”

---

### Block

一个 block 是一组线程（thread）的集合，**Block 是资源分配的基本单位**，在一个 SM 上运行。

例如：

```cpp
blockDim = 256
```

表示一个 block 有 256 个线程。

**特点**

* 同一个 block 内：
  * 可以使用 `__syncthreads()` 同步
  * 可以共享 **shared memory**
* 一个 block 只能运行在 **一个 SM 上**
* 一个 SM 可以同时驻留多个 block

> **资源分配是以 block 为单位的**

比如：

* shared memory 分配
* register 分配
* 调度单位

如果 block 太大，会降低 occupancy。

---

### Wrap

**warp 是 GPU 真正的执行单位。**

在 NVIDIA GPU 中：

> 1 warp = 32 个线程

这是硬件固定的。

* warp 内的 32 个线程：
  * 同一时间执行同一条指令（SIMT）
  * 共享一个 program counter
* 如果出现分支：

```cpp
if (threadIdx.x % 2 == 0)
```

会发生：

> warp divergence（分支发散）

GPU 会串行执行两个分支。

### 层级图

```
Kernel Launch
    ↓
Grid
    ↓
Block (多个)
    ↓
Warp (block 内自动划分)
    ↓
Thread
```

例如：

```cpp
<<<grid=4, block=128>>>
```

意味着：

* 4 个 block
* 每个 block 128 线程
* 每个 block 里有：

```
128 / 32 = 4 warp
```


| 维度                 | Grid | Block | Warp    |
| ------------------ | ---- | ----- | ------- |
| 层级                 | 最高   | 中间    | 最底层执行单位 |
| 是否硬件概念             | 否    | 半抽象   | 是       |
| 是否可同步              | ❌    | ✅     | 不需要     |
| 是否共享 shared memory | ❌    | ✅     | 自动共享    |
| 调度单位               | ❌    | ❌     | ✅       |
| 是否跨 SM             | 是    | 否     | 否       |

---

### 硬件视角理解

真正执行的是：

> SM 调度 warp

而不是调度 thread，也不是调度 block。

执行流程是：

1. SM 接收一个 block
2. block 被拆成多个 warp
3. warp 被调度执行
4. 每个时钟周期发射 warp 指令

所以：

> warp 才是 GPU 性能调优的核心单位

---

**常见误区**

❌ 误区1：线程是执行单位

不是。
线程只是“逻辑单位”，真正执行的是 warp。

❌ 误区2：block 是执行单位

不是。
block 是资源分配单位。

❌ 误区3：warp 可以跨 block

绝对不行。
warp 永远属于某一个 block。

---

你在做 MoE / fused kernel / tensor parallel 时，真正影响性能的是：

* warp 利用率
* warp divergence
* memory coalescing（warp 内）
* occupancy（block 资源）

大模型 kernel 优化，本质都是：

> 让 warp 高效、连续、无发散地执行

---

总结

> **grid 是任务级，block 是资源级，warp 是执行级。**
